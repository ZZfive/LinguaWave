# CosyVoice原理解析
在实际训练之前，需要更进一步对CosyVoice的架构、训练过程等分析，加强理解，便于后续的复现。

 - [CosyVoice1](#CosyVoice1)
 - [CosyVoice2](#CosyVoice2)
 - [注意事项](#注意事项)


## CosyVoice1
### 架构细节
&emsp;&emsp;在项目入口的[README.md](../README.md)中简短介绍了Cosyvoice架构，具体架构图下图所示。架构图(a)为CosyVoice中提出的以有监督语义tokens通过在编码器后插入Vector Quantizer/向量量化模块的多语言speech tokenizer架构，Vector Quantizer的码本大小为4096。图(a)中的模型本质是阿里内部专用SenseVoice ASR模型的微调版本(类Whisper模型)，虚线部分只在speech tokenizer训练过程中会使用，训练任务就是ASR。训练结束之后图(a)中的$Encoder_1$和Vector Quantizer组合为speech tokenizer，主要用于从音频文件中提取对应的speech tokens序列。

![enter image description here](CosyVoice_structure.png?raw=true)

&emsp;&emsp;架构图(b)是完整的CosyVoice预览图，从下向上看，输入包括x-Vex v、Text Y和Speech X，其中x-vec v是从音频中提取的说话人特征embedding，Text Y就是CosyVoice推理时设置的文本输入，Speech X就是推理时的音频输入(因为CosyVoice支持音色克隆)。Text Encoder将文本和speech tokens的语音空间对齐，因为Text-to-token LM中本质是speech tokens的隐空间，其预测的也是speech tokens；Speech Tokenizer从音频中提取speech tokens；Text-to-token LM模块主要由Transformer blocks堆叠组成，输出经过Embedding layer转换为与Contional flow matching/CFM模块训练的条件向量，最终预测对应内容的mel谱图特征，然后使用HiFiGAN生成最终音频(上图中没有显示HiFiGAN)。

&emsp;&emsp;CosyVoice是使用大型语言模型来学习整个文本编码和语音标记序列，将 TTS 重新定义为给定文本作为提示的自回归序列生成问题。

&emsp;&emsp;架构图(c)是CFM模块的放大视图，整体结构是类Unet模型，每层是以Transformer block为主，配合一维的ResNet层。CFM模块通过条件流匹配的方式在各类条件的引导下，将一个先验分布--标准正态分布开始预测出目标分布--目标mel谱图。

### 训练细节
#### LM模块
&emsp;&emsp;LM训练使用强制教师方式，左移一步，使用上一步的输出作为当前步的输入，进行训练。在训练期间只考虑语音标记和预测结束token E 的交叉熵损失。

&emsp;&emsp;CosyVoice具有zeor-shot上下文学习能力，允许仅使用简短参考音频样本复刻任意音色，即音色克隆；但此过程需要仔细构建LM模块的输入序列，如下图所示。图(a)是参考音频语种与目标语种相同时的输入构建方式，图(b)是参考音频语种与目标语种不同时的输入构建方式；主要区别是当语种不一致时，只会使用说话人特征embedding，而不会使用参考音频的文本和音频。

![enter image description here](CosyVoice_input.png?raw=true)

&emsp;&emsp;为了扩展CosyVoice的控制能力，还尝试了额外的指令微调。具体来说，它支持对各种方面的可控性，例如说话者身份（即说话者的特征）、说话风格（包括情感、性别、语速和音高）和细粒度的副语言特征。这些特征包括在笑时插入笑声、呼吸、说话和强调某些单词的能力。

#### CFM模块
&emsp;&emsp;基于最优传输构建OT-CFM进行条件生成，条件除了speech tokens(包含输入的和预测的speech tokens)，还包括说话人特征embedding，掩码Mel谱图特征$\tilde{X}$(对真实目标Mel谱图随机设置的起点开始直到结束设置为0)和连续时间$t \in [0,1]$。

&emsp;&emsp;考虑到开始时的生成更难，使用时间步$t$使用余弦调度器$t:=1-\cos(\frac{1}{2}\pi t)$，开始阶段时间步更密集，结束阶段时间步更稀疏。同时训练过程也使用了CFG，以20%的固定概率随机丢掉条件，可同时学习到条件流和无条件流。

#### 数据集
&emsp;&emsp;对 LibriTTS 语料库进行实验，该语料库包含来自 2,456 个英语说话者的 585 小时；遵循官方数据分区，其中“train-clean-100”、“trainclean-360”和“train-other-500”被合并用于训练，“dev-clean”用于模型选择，“test-clean”用于构建评估集。

&emsp;&emsp;开发人员使用内部工具进行语言检测、信噪比/SNR估计，说话人二值化等操作，搜集了一个巨大的多语言数据集用于训练，详情如下表所示。
| Language | Duration (hr) |
|:--------:|:-------------:|
|    ZH    |    130,000    |
|    EN    |     30,000    |
|   Yue    |     5,000     |
|    JP    |     4,600     |
|    KO    |     2,200     |

#### 训练设置
&emsp;&emsp;CosyVoice论文中只是公布Speech Tokenizer和CosyVoice训练的简短信息。对于在LibriTTS 语料库上的实验性训练，是时使用ESPNet Conformer ASR模型为backbone，在第六个encoder layers后插入Vector Quantizer层，未加载预训练权重，从头开始训练了50个epochs。对于完整体数据集，使用SenseVoice-Large模型作为backbone，也是从第六个encoder layers后插入Vector Quantizer层，但加载了预训练的权重，并在8张A800 GPU上全量微调210000步。

&emsp;&emsp;与两种体量的数据集和Speech Tokenizer对应，CosyVoice也训练了两个体量，细节如下表所示。Tiny模型在LibriTTS训练集上训练50个epoch，使用了4张V100-32M GPU，而Normal模型在完整体内部数据集上训练800,000步，使用64张 V100-32M GPU。Tiny和Normal模型的学习率分别是$10^{-3},10^{-4}$，预热步数都是10000步。
| Settings           | Tiny  | Normal |
|:------------------:|:-----:|:------:|
|           |**Text Encoder**|        |
| Layers             |   6   |    6   |
| Attention Dim.     |  512  | 1,024  |
| Attention Heads    |   8   |   16   |
| Linear Units       | 2,048 | 4,096  |
|       |**Language Encoder**|        |
| Layers             |  12   |   14   |
| Attention Dim.     |  512  | 1,024  |
| Attention Heads    |   8   |   16   |
| Linear Units       | 2,048 | 4,096  |

&emsp;&emsp;论文中公布的训练细节并不详细，但CosyVoice中提供了比较详细的训练脚本和代码，具体细节可参考。

## CosyVoice2
&emsp;&emsp;CosyVoice的成功，证明了LM+CFM的TTS架构的高效，开发团队趁热打铁，在CosyVoice基础上改进了流式能力，优化性能，推出了CosyVoice2。

&emsp;&emsp;CosyVoice2的架构如下图所示，整体架构与CosyVoice1类似，但增加了流式推理能力，并使用多任务学习来提高性能。主要从以下三个方面优化：
1. 用有限标量量化(Finite scalar quantization，FSQ)替换语音标记器中的矢量量化(VQ)，提高码本利用率，捕获更多的语音信息
2. 简化LM模块，移除了text encoder和speaker embedding，允许直接使用预训练的LLM作为backbone，增强上下文理解
3. 开发了一个块感知的因果流匹配模型支持各种合成场景，能在单模型推理中启动流式和非流式合成

### 架构细节
![enter image description here](CosyVoice2_structure.png?raw=true)

&emsp;&emsp;CosyVoice2架构图如上所示，整体结构与CosyVoice变化不大，主要是各个模块内的存在一些细节区别。

#### Text Tokenizer
&emsp;&emsp;CosyVoice2直接使用原始文本作为输入，通过BPE文本分词器进行分词，消除了“字素到音素转换过程”所需的前端模型，简化了数据预处理工作流程，还使模型能够以端到端的方式学习各种上下文单词的发音。与文本 LLM 中常用的分词器不同，CosyVoice 2 掩盖了一对多的标记。这可以防止单个token发音变得过长，并减少数据稀疏性引起的极端情况。

#### Speech Tokenizer
&emsp;&emsp;CosyVoice2使用FSQ替换VQ，在FSQ中经过$Encoder_1$编码后中间特征先隐射到D维的低秩空间，然后对向量的每个维度的值通过有界舍入操作ROUND量化到[-K,K]范围，然后再将量化后的低秩表征投影会原始维度。训练阶段使用straight-through estimator(STE)近似FSQ和$Encoder_1$的梯度，通过(2K + 1)进制系统将量化后的值转换为单个token，公式如下；$Encoder_1$、FSQ模块、有界ROUND运算和索引计算形成了CosyVoice2的Speech Tokenizer。
$$\mu_i=\sum_{j=0}^{D-1} \bar{h}_{i,j}(2K+1)^j$$

#### Text-Speech Language Model
&emsp;&emsp;CosyVoice2使用预训练的Qwen2.5-0.5B作为backbone，以文本为提示吃自回归生成speech tokens。实验发现在LM中使用speaker embedding会造成信息泄露，且其包含的语言和副语言信息会损害LM的韵律自然都和跨语言能力。此外，因为Qwen2.5-0.5B模型的想打能力，其可以对齐文本和specch tokens，故移除了Text Encoder。

&emsp;&emsp;LM模块简化后，可以建立一个同时支持流式和非流式的的生成模型。流式是表示输入文本在连续流中接受，而不是预先要获取整个完整文本句子。CosyVoice2中流式和非流式的区别支持LM的输出序列的构建方式不同：
1. 流式--下图上半部分；LM输入中以一个预定义的比例N:M将文本tokens和音频tokens混合，即每N个文本tokens后面接M个音频tokens。所过下一个token是文本token，则希望模型预测出Filling token，而不是文本token。一旦文本tokens耗尽，turn of speech token和剩余的语音tokens顺序拼接，形成流模式下的混合文本-语音tokens序列。
2. 非流式--下图下半部分；LM的输入依次由start of sequence token、所有的文本tokens、turn of speech token、所有的音频tokens、end of sequence token组成。

![enter image description here](CosyVoice2_input.png?raw=true)

&emsp;&emsp;通过同时在以上两种序列上训练text-speech LM，可以在单个统一模型中执行流式和非流式语音生成。

#### Chunk-aware Flow Matching
&emsp;&emsp;CosyVoice2的FM模块架构图和和用于流式生成的四类Mask如下图所示。FM模块结构与CosyVoice中实现不同，但训练思路基本一致。训练阶段时间步长服从均匀分布$U[0,1]$，但推理时与CosyVoice一致，使用余弦调度器为生成早期提供更多推理步数，也使用CFG方式进行训练。为了改善流式生成性能，CosyVoice2将多步流估计视为一个堆叠更深的神经网络；通过将神经网络因果展开，可以将其应用于流式生成，为此构建了四种掩码来满足不同的应用情况。
1. Non-causal Mask--非因果掩码，用于离线模式，可通过关注所有条件帧来实现最佳性能，适用于对延迟不敏感的场景
2. Full-causal Mask--全因果掩码，为需要极低延迟场景涉及，当前帧只关注过去帧
3. Chunk-M Mask--延迟和性能之间的权衡，可以利用过去帧和M个未来帧的信息，适合要求第一个chunk生成较快的场景
4. Chunk-2M Mask--牺牲更多的实时性来获得更好的性能，更接近离线模式

![enter image description here](CosyVoice2_cfm.png?raw=true)

&emsp;&emsp;对于小批量中的每个训练样例，从上述四种掩码中按均匀分布随机采样一个掩码。通过这种方式，一个流匹配模型可以兼容不同的场景，降低部署的复杂性。这种分块感知训练的另一个优点是，具有更多上下文的掩码可以作为具有较少上下文的掩码的教师，是一种隐式的自蒸馏方案。

### 训练细节
#### Multi-Speaker Fine-tuning
&emsp;&emsp;目前发布的CosyVoice2-0.5B模型可看作预训练的base模型，在其基础上进行特定的说话人/speakers微调可以进一步提高生成质量和相似度。CosyVoice2提出了多说话疼监督微调/mSFT，即同时再多个说话人数据上微调预训练模型。这种方法确保了跨多个说话者的全面韵律和发音覆盖率，并减轻了预训练模型的潜在灾难性遗忘。为了避免各种speakers之间的音色混淆，将speaker提示标签“Speaker A<|endofprompt|>”添加到特定说话者的输入文本中，类似于Instruct prompt。如果训练样本中没有提供speaker标签，则使用默认的s“unknown<|endofprompt|>”。在整个mSFT过程中，学习率设置为 1e-5。

#### Reinforcement Learning for SFT
&emsp;&emsp;为提高CosyVoice2性能，使用speaker相似度/SS和ASR系统的识别单次错误率WER作为奖励函区分preferred sample $x^w$和rejected sample $x^l$，使用DPO进行优化，即：
$$L_{DPO}(\pi_{\theta};\pi_{ref})=-\log \sigma(\beta \log \frac{\pi_{\theta}(\mu^w|y)}{\pi_{ref}(\mu^w|y)}-\beta \log \frac{\pi_{\theta}(\mu^l|y)}{\pi_{ref}(\mu^l|y)})$$
其中$\mu^w$和$\mu^l$分别是从preferred sample $x^w$和rejected sample $x^l$中提取的speech tokens。

&emsp;&emsp;然而这种方法既耗时，计算量又大，因为它会反复通过TTS系统合成音频，以获得可区分的偏好和被拒绝的样本；训练过程中一个训练步骤需要四次前向操作。为了简化流程，将LM预测的speech tokens $\mu_i \in \lbrace 0,1,...,(2K+1)^D-1 \rbrace$恢复为低质空间向量$\bar{H}$，然后直接使用speech tokenizer后半部分的ASR后端预测出输入文本；预测的对数后验概率可视为优化text-speech LM的ASR奖励函数。训练过程中，ASR后端冻结，
$$
\begin{align*}
\bar{h}_{i,j} & = [\frac{\mu_i}{(2K+1)^j}] \mod (2K+1) \\
\hat{H} & = Proj_{up}(\bar{H}) \\
L_{ASR} & = -\log p(Y|\hat{H};\theta_{ASR})
\end{align*}
$$
其中$Y$是输入文本，$\hat{H}$是从预测的speech tokens中恢复的低秩表征。由于$u_i \sim P(u_i|u_{1:i-1},Y;\theta_{LM})$的样本操作仍然阻止直接优化模型，使用 gumbel softmax 采样使其可微，然后通过 $L_{ASR}$ 优化 $\theta_{LM}$。

#### 数据集
&emsp;&emsp;基于开源ASR数据集、内部数据和TTS生成的数据构建了一个200000小时的数据集用于训练speech tokenizer，虽然此数据集只包含中文和英文，但实验发现训练后的speech tokenizer对其他语言有zeto-shot能力，其可以用于日语和韩语等语言生成。speech tokenizer训练构成如下表所示，而CosyVoice2使用了与CosyVoice相同的训练数据，具体见上述部分。
| Language | Duration (hr) |
|:--------:|:-------------:|
| Chinese  |    110,884    |
| English  |     99,918    |

## 注意事项
1. CosyVoice2中lm模块为Qwen2.5-0.5B模型，该模型正常情况下使用内部的embed_tokens层将输入的文本ids序列转换为embeddings嵌入后进行后续计算；但CosyVoice2中的输入除了文本外，还有音频，为了复用Qwen2.5-0.5B模型中的embed_tokens层并且同时能处理speech tokens，CosyVoice2的做法是先单独使用embed_tokens层对文本ids序列进行嵌入转换操作，然后再使用额外的speech_embedding层对speech tokens进行嵌入转换操作，speech_embedding层的输出维度要与embed_tokens层的输出维度一致，Qwen2.5-0.5B中是896。完成嵌入转换操作后，将所有的嵌入向量拼接得到lm_input，transfomers库中实现的Qwen2Model的forward函数支持直接传入inputs_embeds，即lm_input，就实现了在复用了Qwen2.5-0.5B模型中的embed_tokens层的同时，还能处理speech tokens。