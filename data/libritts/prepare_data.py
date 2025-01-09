import argparse
import logging
import glob
import os
from tqdm import tqdm


logger = logging.getLogger()


def main():
    wavs = list(glob.glob('{}/*/*/*wav'.format(args.src_dir)))  # 获取所有wav文件

    utt2wav, utt2text, utt2spk, spk2utt = {}, {}, {}, {}
    for wav in tqdm(wavs):
        txt = wav.replace('.wav', '.normalized.txt')  # 获取文本文件
        if not os.path.exists(txt):
            logger.warning('{} do not exsist'.format(txt))
            continue
        with open(txt) as f:
            content = ''.join(l.replace('\n', '') for l in f.readline())  # 读取文本文件
        utt = os.path.basename(wav).replace('.wav', '')  # 获取utt，即使不带文件格式后缀的文件名
        spk = utt.split('_')[0]  # 获取说话人id，utt的第一个下划线前的字符串
        utt2wav[utt] = wav  # wav文件路径
        utt2text[utt] = content  # 文本
        utt2spk[utt] = spk  # 说话人id
        if spk not in spk2utt:
            spk2utt[spk] = []
        spk2utt[spk].append(utt)  # 记录是当前说话人的所有utt

    with open('{}/wav.scp'.format(args.des_dir), 'w') as f:  # 记录utt与wav文件路径的映射
        for k, v in utt2wav.items():  
            f.write('{} {}\n'.format(k, v))
    with open('{}/text'.format(args.des_dir), 'w') as f:  # 记录utt与文本的映射
        for k, v in utt2text.items():
            f.write('{} {}\n'.format(k, v))
    with open('{}/utt2spk'.format(args.des_dir), 'w') as f:  # 记录utt与说话人id的映射
        for k, v in utt2spk.items():
            f.write('{} {}\n'.format(k, v))
    with open('{}/spk2utt'.format(args.des_dir), 'w') as f:  # 记录说话人id与其所有utt的映射
        for k, v in spk2utt.items():
            f.write('{} {}\n'.format(k, ' '.join(v)))
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir',
                        type=str)
    parser.add_argument('--des_dir',
                        type=str)
    args = parser.parse_args()
    main()
