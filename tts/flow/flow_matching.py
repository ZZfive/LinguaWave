# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import threading
import torch
import torch.nn.functional as F
from matcha.models.components.flow_matching import BASECFM


class ConditionalCFM(BASECFM):
    def __init__(self, in_channels, cfm_params, n_spks=1, spk_emb_dim=64, estimator: torch.nn.Module = None):
        super().__init__(
            n_feats=in_channels,
            cfm_params=cfm_params,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )
        self.t_scheduler = cfm_params.t_scheduler
        self.training_cfg_rate = cfm_params.training_cfg_rate
        self.inference_cfg_rate = cfm_params.inference_cfg_rate
        in_channels = in_channels + (spk_emb_dim if n_spks > 0 else 0)
        # Just change the architecture of the estimator here
        self.estimator = estimator
        self.lock = threading.Lock()

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None, prompt_len=0, flow_cache=torch.zeros(1, 80, 0, 2)):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """

        z = torch.randn_like(mu).to(mu.device).to(mu.dtype) * temperature  # 生成初始噪声，shape: [batch_size, n_feats, mel_timesteps]，如[1, 80, 942]
        cache_size = flow_cache.shape[2]
        # fix prompt and overlap part mu and z
        if cache_size != 0:
            z[:, :, :cache_size] = flow_cache[:, :, :, 0]
            mu[:, :, :cache_size] = flow_cache[:, :, :, 1]
        z_cache = torch.concat([z[:, :, :prompt_len], z[:, :, -34:]], dim=2)
        mu_cache = torch.concat([mu[:, :, :prompt_len], mu[:, :, -34:]], dim=2)
        flow_cache = torch.stack([z_cache, mu_cache], dim=-1)

        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == 'cosine':  # 使用余弦调度器
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond), flow_cache

    def solve_euler(self, x, t_span, mu, mask, spks, cond):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder；speech tokens经过encoder编码后的输出
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes；参考音频的mel谱图特征
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        t = t.unsqueeze(dim=0)  # 如[1]

        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        sol = []

        # Do not use concat, it may cause memory format changed and trt infer with wrong results!
        # 此处所有输入都翻倍的原因是同时进行有条件和无条件预测
        x_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=x.dtype)  # 如[2, 80, 110]
        mask_in = torch.zeros([2, 1, x.size(2)], device=x.device, dtype=x.dtype)  # 如[2, 1, 110]
        mu_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=x.dtype)  # 如[2, 80, 110]
        t_in = torch.zeros([2], device=x.device, dtype=x.dtype)  # 如[2]
        spks_in = torch.zeros([2, 80], device=x.device, dtype=x.dtype)  # 如[2, 80]
        cond_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=x.dtype)  # 如[2, 80, 110]
        for step in range(1, len(t_span)):
            # Classifier-Free Guidance inference introduced in VoiceBox
            x_in[:] = x  # 将x复制到x_in中，会将x重复两次
            mask_in[:] = mask  # 将mask复制到mask_in中，会将mask重复两次
            mu_in[0] = mu  # 将mu赋值给mu_in[0]，mu_in[1]为0，对应于无条件预测
            t_in[:] = t.unsqueeze(0)  # 将t复制到t_in中，会将t重复两次
            spks_in[0] = spks  # 将spks赋值给spks_in[0]，spks_in[1]为0，对应于无条件预测
            cond_in[0] = cond  # 将cond赋值给cond_in[0]，cond_in[1]为0，对应于无条件预测
            dphi_dt = self.forward_estimator(
                x_in, mask_in,
                mu_in, t_in,
                spks_in,
                cond_in
            )  # 如[2, 80, 110]
            dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)  # 将dphi_dt拆分为两个部分，分别对应有条件和无条件预测，shape均如[1, 80, 110]
            dphi_dt = ((1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt)  # 如[1, 80, 110]；cfg训练综合考虑有条件和无条件预测得到最终预测的向量场
            x = x + dt * dphi_dt  # 基于OT-CFM的欧拉法更新x，如[1, 80, 110]
            t = t + dt  # 更新时间步
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return sol[-1].float()  # 返回最终的x，shape如[1, 80, 110]

    def forward_estimator(self, x, mask, mu, t, spks, cond):
        if isinstance(self.estimator, torch.nn.Module):
            return self.estimator.forward(x, mask, mu, t, spks, cond)  # 如[1, 80, 110]
        else:
            with self.lock:
                self.estimator.set_input_shape('x', (2, 80, x.size(2)))
                self.estimator.set_input_shape('mask', (2, 1, x.size(2)))
                self.estimator.set_input_shape('mu', (2, 80, x.size(2)))
                self.estimator.set_input_shape('t', (2,))
                self.estimator.set_input_shape('spks', (2, 80))
                self.estimator.set_input_shape('cond', (2, 80, x.size(2)))
                # run trt engine
                self.estimator.execute_v2([x.contiguous().data_ptr(),
                                           mask.contiguous().data_ptr(),
                                           mu.contiguous().data_ptr(),
                                           t.contiguous().data_ptr(),
                                           spks.contiguous().data_ptr(),
                                           cond.contiguous().data_ptr(),
                                           x.data_ptr()])
            return x

    def compute_loss(self, x1, mask, mu, spks=None, cond=None):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            spks (torch.Tensor, optional): speaker embedding. Defaults to None.
                shape: (batch_size, spk_emb_dim)

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = mu.shape

        # random timestep
        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == 'cosine':
            t = 1 - torch.cos(t * 0.5 * torch.pi)
        # sample noise p(x_0)
        z = torch.randn_like(x1)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1  # 此时的y对应于flow matching中的x_t
        u = x1 - (1 - self.sigma_min) * z  # flow matching中的目标向量场

        # during training, we randomly drop condition to trade off mode coverage and sample fidelity; 在训练过程中，我们随机丢弃条件以权衡模式覆盖率和样本保真度
        if self.training_cfg_rate > 0:
            cfg_mask = torch.rand(b, device=x1.device) > self.training_cfg_rate  # 随机生成一个mask，用于控制条件注入的概率
            mu = mu * cfg_mask.view(-1, 1, 1)  # 将mu乘以cfg_mask，使得在训练过程中随机丢弃条件
            spks = spks * cfg_mask.view(-1, 1)  # 将spks乘以cfg_mask，使得在训练过程中随机丢弃条件
            cond = cond * cfg_mask.view(-1, 1, 1)  # 将cond乘以cfg_mask，使得在训练过程中随机丢弃条件

        pred = self.estimator(y, mask, mu, t.squeeze(), spks, cond)  # 预测的向量场
        loss = F.mse_loss(pred * mask, u * mask, reduction="sum") / (torch.sum(mask) * u.shape[1])  # flow matching的损失函数
        return loss, y


class CausalConditionalCFM(ConditionalCFM):
    def __init__(self, in_channels, cfm_params, n_spks=1, spk_emb_dim=64, estimator: torch.nn.Module = None):
        super().__init__(in_channels, cfm_params, n_spks, spk_emb_dim, estimator)
        self.rand_noise = torch.randn([1, 80, 50 * 300])  # 初始化时就随机生成一个固定噪声，shape为[1, 80, 15000]

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """

        z = self.rand_noise[:, :, :mu.size(2)].to(mu.device).to(mu.dtype) * temperature  # 以mu的长度从固定噪声中截取推理时要使用的噪声，如[1, 80, 110]
        # fix prompt and overlap part mu and z
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == 'cosine':
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond), None