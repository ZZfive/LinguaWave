from functools import partial
from typing import List, Callable, Tuple

import torch
from torch.utils.data import DataLoader
from librosa.filters import mel as librosa_mel_fn

from tts.dataset.dataset import Dataset
from tts.tokenizer.tokenizer import get_tokenizer
from tts.dataset.processor import parquet_opener, tokenize, filter, resample, compute_fbank, parse_embedding, shuffle, sort, batch, padding


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global mel_basis, hann_window  # pylint: disable=global-statement
    if f"{str(fmax)}_{str(y.device)}" not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax) + "_" + str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode="reflect"
    )
    y = y.squeeze(1)

    spec = torch.view_as_real(
        torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=hann_window[str(y.device)],
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(mel_basis[str(fmax) + "_" + str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def init_pipeline(mode: str = 'train', multilingual: bool = True, num_languages: int = 100, language: str = 'en',
                  task: str = 'transcribe',allowed_special: str = 'all', max_length: int = 40960, min_length: int = 0,
                  token_max_length: int = 200, token_min_length: int = 1, sample_rate: int = 22050, n_fft: int = 1024,
                  num_mels: int = 80, hop_size: int = 256, win_size: int = 1024, fmin: int = 0, fmax: int = 8000,
                  center: bool = False, normalize: bool = True, shuffle_size: int = 1000, sort_size: int = 500,
                  batch_type: str = 'dynamic', max_frames_in_batch: int = 2000, use_spk_embedding: bool = False) -> List[Callable]:
    pipeline = []
    pipeline.append(parquet_opener)

    init_tokenizer = partial(get_tokenizer, multilingual=multilingual, num_languages=num_languages, language=language, task=task)
    pipeline.append(partial(tokenize, get_tokenizer=init_tokenizer, allowed_special=allowed_special, mode=mode))

    pipeline.append(partial(filter, max_length=max_length, min_length=min_length, token_max_length=token_max_length,
                            token_min_length=token_min_length))

    pipeline.append(partial(resample, resample_rate=sample_rate))

    feat_extractor = partial(mel_spectrogram, n_fft=n_fft, num_mels=num_mels, sampling_rate=sample_rate,
                             hop_size=hop_size, win_size=win_size, fmin=fmin, fmax=fmax, center=center)

    pipeline.append(partial(compute_fbank, feat_extractor=feat_extractor, mode=mode))

    pipeline.append(partial(parse_embedding, normalize=normalize))

    pipeline.append(partial(shuffle, shuffle_size=shuffle_size))

    pipeline.append(partial(sort, sort_size=sort_size))

    pipeline.append(partial(batch, batch_type=batch_type, max_frames_in_batch=max_frames_in_batch))

    pipeline.append(partial(padding, use_spk_embedding=use_spk_embedding))
    return pipeline


def init_dataset_and_dataloader(data_list_file, gan: bool = False, pin_memory: bool = False, num_workers: int = 2,
                                prefetch: int = 100) -> Tuple[Dataset, DataLoader]:
    data_pipeline = init_pipeline()
    dataset = Dataset(data_list_file, data_pipeline=data_pipeline, mode='train', gan=gan, shuffle=True, partition=True)

    # do not use persistent_workers=True, as whisper tokenizer opens tiktoken file each time when the for loop starts
    train_data_loader = DataLoader(dataset,
                                   batch_size=None,
                                   pin_memory=pin_memory,
                                   num_workers=num_workers,
                                   prefetch_factor=prefetch)
    
    return dataset, train_data_loader


if __name__ == '__main__':
    data_list_file = "data.list"
    dataset, data_loader = init_dataset_and_dataloader(data_list_file)
    for batch_idx, batch_dict in enumerate(data_loader):
        print(batch_dict)
        break
