# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
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
import logging
import random

import pyarrow.parquet as pq
from io import BytesIO
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import pyworld as pw


AUDIO_FORMAT_SETS = {'flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'}


def parquet_opener(data, mode='train', tts_data={}):
    """ Give url or local file, return file descriptor
        Inplace operation.

        Args:
            data(Iterable[str]): url or local file list

        Returns:
            Iterable[{src, stream}]
    """
    for sample in data:
        assert 'src' in sample
        url = sample['src']
        try:
            for df in pq.ParquetFile(url).iter_batches(batch_size=64):
                df = df.to_pandas()
                for i in range(len(df)):
                    if mode == 'inference' and df.loc[i, 'utt'] not in tts_data:
                        continue
                    sample.update(dict(df.loc[i]))
                    if mode == 'train':
                        # NOTE do not return sample directly, must initialize a new dict
                        yield {**sample}
                    else:
                        for index, text in enumerate(tts_data[df.loc[i, 'utt']]):
                            yield {**sample, 'tts_index': index, 'tts_text': text}
        except Exception as ex:
            logging.warning('Failed to open {}, ex info {}'.format(url, ex))


def filter(data,
           max_length=10240,
           min_length=10,
           token_max_length=200,
           token_min_length=1,
           min_output_input_ratio=0.0005,
           max_output_input_ratio=1,
           mode='train'):
    """ Filter sample according to feature and label length
        Inplace operation.

        Args::
            data: Iterable[{key, wav, label, sample_rate}]
            max_length: drop utterance which is greater than max_length(10ms)
            min_length: drop utterance which is less than min_length(10ms)
            token_max_length: drop utterance which is greater than
                token_max_length, especially when use char unit for
                english modeling
            token_min_length: drop utterance which is
                less than token_max_length
            min_output_input_ratio: minimal ration of
                token_length / feats_length(10ms)
            max_output_input_ratio: maximum ration of
                token_length / feats_length(10ms)

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        sample['speech'], sample['sample_rate'] = torchaudio.load(BytesIO(sample['audio_data']))  # 加载音频数据
        sample['speech'] = sample['speech'].mean(dim=0, keepdim=True)  # 计算音频数据的平均值
        del sample['audio_data']  # 删除音频数据
        # sample['wav'] is torch.Tensor, we have 100 frames every second
        num_frames = sample['speech'].size(1) / sample['sample_rate'] * 100  # 计算音频数据帧数
        if num_frames < min_length:  # 如果帧数小于最小长度，则跳过该样本
            continue
        if num_frames > max_length:  # 如果帧数大于最大长度，则跳过该样本
            continue
        if len(sample['text_token']) < token_min_length:  # 如果文本token长度小于最小长度，则跳过该样本
            continue
        if len(sample['text_token']) > token_max_length:  # 如果文本token长度大于最大长度，则跳过该样本
            continue
        if len(sample['speech_token']) == 0:  # 如果音频token长度为0，则跳过该样本
            continue
        if num_frames != 0:  # 如果帧数不为0
            if len(sample['text_token']) / num_frames < min_output_input_ratio:
                continue
            if len(sample['text_token']) / num_frames > max_output_input_ratio:
                continue
        yield sample


def resample(data, resample_rate=22050, min_sample_rate=16000, mode='train'):
    """ Resample data.
        Inplace operation.

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            resample_rate: target resample rate

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'speech' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['speech']
        if sample_rate != resample_rate:
            if sample_rate < min_sample_rate:  # 如果采样率小于min_sample_rate，则跳过该样本
                continue
            sample['sample_rate'] = resample_rate
            sample['speech'] = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=resample_rate)(waveform)  # 重采样
        max_val = sample['speech'].abs().max()
        if max_val > 1:
            sample['speech'] /= max_val  # 归一化
        yield sample


def truncate(data, truncate_length=24576, mode='train'):
    """ Truncate data.

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            truncate_length: truncate length

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        waveform = sample['speech']
        if waveform.shape[1] > truncate_length:  # 如果音频数据长度大于truncate_length，则随机截取一段音频数据
            start = random.randint(0, waveform.shape[1] - truncate_length)
            waveform = waveform[:, start: start + truncate_length]
        else:
            waveform = torch.concat([waveform, torch.zeros(1, truncate_length - waveform.shape[1])], dim=1)  # 如果音频数据长度小于truncate_length，则填充0
        sample['speech'] = waveform
        yield sample


def compute_fbank(data,
                  feat_extractor,
                  mode='train'):
    """ Extract fbank

        Args:
            data: Iterable[{key, wav, label, sample_rate}]

        Returns:
            Iterable[{key, feat, label}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'speech' in sample
        assert 'utt' in sample
        assert 'text_token' in sample
        waveform = sample['speech']
        mat = feat_extractor(waveform).squeeze(dim=0).transpose(0, 1)  # 提取mel谱图特征
        sample['speech_feat'] = mat
        yield sample


def compute_f0(data, sample_rate, hop_size, mode='train'):
    """ Extract f0

        Args:
            data: Iterable[{key, wav, label, sample_rate}]

        Returns:
            Iterable[{key, feat, label}]
    """
    frame_period = hop_size * 1000 / sample_rate
    for sample in data:
        assert 'sample_rate' in sample
        assert 'speech' in sample
        assert 'utt' in sample
        assert 'text_token' in sample
        waveform = sample['speech']
        _f0, t = pw.harvest(waveform.squeeze(dim=0).numpy().astype('double'), sample_rate, frame_period=frame_period)
        if sum(_f0 != 0) < 5: # this happens when the algorithm fails
            _f0, t = pw.dio(waveform.squeeze(dim=0).numpy().astype('double'), sample_rate, frame_period=frame_period) # if harvest fails, try dio
        f0 = pw.stonemask(waveform.squeeze(dim=0).numpy().astype('double'), _f0, t, sample_rate)
        f0 = F.interpolate(torch.from_numpy(f0).view(1, 1, -1), size=sample['speech_feat'].shape[0], mode='linear').view(-1)
        sample['pitch_feat'] = f0
        yield sample


def parse_embedding(data, normalize, mode='train'):
    """ Parse utt_embedding/spk_embedding

        Args:
            data: Iterable[{key, wav, label, sample_rate}]

        Returns:
            Iterable[{key, feat, label}]
    """
    for sample in data:
        sample['utt_embedding'] = torch.tensor(sample['utt_embedding'], dtype=torch.float32)  # 将utt_embedding转换为torch.Tensor
        sample['spk_embedding'] = torch.tensor(sample['spk_embedding'], dtype=torch.float32)  # 将spk_embedding转换为torch.Tensor
        if normalize:
            sample['utt_embedding'] = F.normalize(sample['utt_embedding'], dim=0)  # 归一化
            sample['spk_embedding'] = F.normalize(sample['spk_embedding'], dim=0)  # 归一化
        yield sample


def tokenize(data, get_tokenizer, allowed_special, mode='train'):
    """ Decode text to chars or BPE
        Inplace operation

        Args:
            data: Iterable[{key, wav, txt, sample_rate}]

        Returns:
            Iterable[{key, wav, txt, tokens, label, sample_rate}]
    """
    tokenizer = get_tokenizer()
    for sample in data:
        assert 'text' in sample
        sample['text_token'] = tokenizer.encode(sample['text'], allowed_special=allowed_special)
        if mode == 'inference':
            sample['tts_text_token'] = tokenizer.encode(sample['tts_text'], allowed_special=allowed_special)
        yield sample


def shuffle(data, shuffle_size=10000, mode='train'):
    """ Local shuffle the data

        Args:
            data: Iterable[{key, feat, label}]
            shuffle_size: buffer size for shuffle

        Returns:
            Iterable[{key, feat, label}]
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= shuffle_size:
            random.shuffle(buf)
            for x in buf:
                yield x
            buf = []
    # The sample left over
    random.shuffle(buf)
    for x in buf:
        yield x


def sort(data, sort_size=500, mode='train'):
    """ Sort the data by feature length.
        Sort is used after shuffle and before batch, so we can group
        utts with similar lengths into a batch, and `sort_size` should
        be less than `shuffle_size`

        Args:
            data: Iterable[{key, feat, label}]
            sort_size: buffer size for sort

        Returns:
            Iterable[{key, feat, label}]
    """

    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= sort_size:
            buf.sort(key=lambda x: x['speech_feat'].size(0))  # 根据speech_feat的长度排序
            for x in buf:
                yield x
            buf = []
    # The sample left over
    buf.sort(key=lambda x: x['speech_feat'].size(0))  # 根据speech_feat的长度排序
    for x in buf:
        yield x


def static_batch(data, batch_size=16):
    """ Static batch the data by `batch_size`

        Args:
            data: Iterable[{key, feat, label}]
            batch_size: batch size

        Returns:
            Iterable[List[{key, feat, label}]]
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= batch_size:
            yield buf
            buf = []
    if len(buf) > 0:  # 返回最后一批数量不足batch_size的数据
        yield buf


def dynamic_batch(data, max_frames_in_batch=12000, mode='train'):
    """ Dynamic batch the data until the total frames in batch
        reach `max_frames_in_batch`

        Args:
            data: Iterable[{key, feat, label}]
            max_frames_in_batch: max_frames in one batch

        Returns:
            Iterable[List[{key, feat, label}]]
    """
    buf = []
    longest_frames = 0
    for sample in data:
        assert 'speech_feat' in sample
        assert isinstance(sample['speech_feat'], torch.Tensor)
        new_sample_frames = sample['speech_feat'].size(0)
        longest_frames = max(longest_frames, new_sample_frames)  # 更新当前batch中最长音频数据的长度
        frames_after_padding = longest_frames * (len(buf) + 1)  # 计算当前batch中所有音频数据的总长度；同一batch中所有音频数据都会通过padding到最长长度
        if frames_after_padding > max_frames_in_batch:  # 如果当前batch中所有音频数据的总长度大于max_frames_in_batch，则返回当前batch
            yield buf  # 返回当前batch，此时buf中还没有包含当前sample
            buf = [sample]  # 将当前sample添加到buf中
            longest_frames = new_sample_frames  # 更新当前batch中最长音频数据的长度
        else:
            buf.append(sample)
    if len(buf) > 0:  # 返回最后一批数量不足max_frames_in_batch的数据
        yield buf


def batch(data, batch_type='static', batch_size=16, max_frames_in_batch=12000, mode='train'):
    """ Wrapper for static/dynamic batch
    """
    if mode == 'inference':
        return static_batch(data, 1)
    else:
        if batch_type == 'static':
            return static_batch(data, batch_size)
        elif batch_type == 'dynamic':
            return dynamic_batch(data, max_frames_in_batch)
        else:
            logging.fatal('Unsupported batch type {}'.format(batch_type))


def padding(data, use_spk_embedding, mode='train', gan=False):
    """ Padding the data into training data

        Args:
            data: Iterable[List[{key, feat, label}]]

        Returns:
            Iterable[Tuple(keys, feats, labels, feats lengths, label lengths)]
    """
    for sample in data:
        assert isinstance(sample, list)  # 此处的sample是batch函数返回的一个batch
        speech_feat_len = torch.tensor([x['speech_feat'].size(1) for x in sample],
                                       dtype=torch.int32)  # 获取当前batch中所有音频mel谱图特征的长度
        order = torch.argsort(speech_feat_len, descending=True)  # 根据mel谱图特征的长度排序

        utts = [sample[i]['utt'] for i in order]  # 获取当前batch中所有音频的utt
        speech = [sample[i]['speech'].squeeze(dim=0) for i in order]  # 获取当前batch中所有音频的speech
        speech_len = torch.tensor([i.size(0) for i in speech], dtype=torch.int32)  # 获取当前batch中所有音频的speech长度
        speech = pad_sequence(speech, batch_first=True, padding_value=0)  # 将当前batch中所有音频的speech填充到同一长度
        speech_token = [torch.tensor(sample[i]['speech_token']) for i in order]  # 获取当前batch中所有音频的speech_token
        speech_token_len = torch.tensor([i.size(0) for i in speech_token], dtype=torch.int32)  # 获取当前batch中所有音频的speech_token长度
        speech_token = pad_sequence(speech_token,
                                    batch_first=True,
                                    padding_value=0)  # 将当前batch中所有音频的speech_token填充到同一长度
        speech_feat = [sample[i]['speech_feat'] for i in order]  # 获取当前batch中所有音频的speech_feat
        speech_feat_len = torch.tensor([i.size(0) for i in speech_feat], dtype=torch.int32)  # 获取当前batch中所有音频的speech_feat长度
        speech_feat = pad_sequence(speech_feat,
                                   batch_first=True,
                                   padding_value=0)  # 将当前batch中所有音频的speech_feat填充到同一长度
        text = [sample[i]['text'] for i in order]  # 获取当前batch中所有音频的text
        text_token = [torch.tensor(sample[i]['text_token']) for i in order]  # 获取当前batch中所有音频的text_token序列
        text_token_len = torch.tensor([i.size(0) for i in text_token], dtype=torch.int32)  # 获取当前batch中所有音频的text_token序列长度
        text_token = pad_sequence(text_token, batch_first=True, padding_value=0)  # 将当前batch中所有音频的text_token序列填充到同一长度
        utt_embedding = torch.stack([sample[i]['utt_embedding'] for i in order], dim=0)
        spk_embedding = torch.stack([sample[i]['spk_embedding'] for i in order], dim=0)
        batch = {
            "utts": utts,
            "speech": speech,
            "speech_len": speech_len,
            "speech_token": speech_token,
            "speech_token_len": speech_token_len,
            "speech_feat": speech_feat,
            "speech_feat_len": speech_feat_len,
            "text": text,
            "text_token": text_token,
            "text_token_len": text_token_len,
            "utt_embedding": utt_embedding,
            "spk_embedding": spk_embedding,
        }
        if gan is True:
            # in gan train, we need pitch_feat
            pitch_feat = [sample[i]['pitch_feat'] for i in order]
            pitch_feat_len = torch.tensor([i.size(0) for i in pitch_feat], dtype=torch.int32)
            pitch_feat = pad_sequence(pitch_feat,
                                      batch_first=True,
                                      padding_value=0)
            batch["pitch_feat"] = pitch_feat
            batch["pitch_feat_len"] = pitch_feat_len
        else:
            # only gan train needs speech, delete it to save memory
            del batch["speech"]
            del batch["speech_len"]
        if mode == 'inference':
            tts_text = [sample[i]['tts_text'] for i in order]
            tts_index = [sample[i]['tts_index'] for i in order]
            tts_text_token = [torch.tensor(sample[i]['tts_text_token']) for i in order]
            tts_text_token_len = torch.tensor([i.size(0) for i in tts_text_token], dtype=torch.int32)
            tts_text_token = pad_sequence(tts_text_token, batch_first=True, padding_value=-1)
            batch.update({'tts_text': tts_text,
                          'tts_index': tts_index,
                          'tts_text_token': tts_text_token,
                          'tts_text_token_len': tts_text_token_len})
        if use_spk_embedding is True:  # 默认use_spk_embedding为False，当进行sft时，设置为True
            batch["embedding"] = batch["spk_embedding"]  # 此为该音频中说话人所有音频中提取出的embedding的平均值
        else:
            batch["embedding"] = batch["utt_embedding"]  # 此为单个音频中提取出的说话人embedding
        yield batch
