# 数据构建
本路径下主要记录数据集构建相关内容

 - [LibriTTS](#LibriTTS)
 - [](#)

## LibriTTS
 - 下载：
   - [download_and_untar.sh](./libritts/download_and_untar.sh)--从CosyVoice项目中复制的下载脚本文件，可分别对dev-clean, test-clean, dev-other, test-other, train-clean-100, train-clean-360, train-other-500进行下载
   - [huggingface](https://huggingface.co/datasets/mythicinfinity/libritts)--Huggingface上数据集地址，也可直接下载
 - 数据预处理：
   1. [prepare_data.py](./libritts/prepare_data.py)--从CosyVoice项目中复制；将下载的LibriTTS数据集转换为类kaldi格式，生成utt2spk, wav.scp, text, spk2utt等文件；下文中的utt就表示一个音频数据，utt本质就是不带文件格式后缀的音频文件名
      - wav.scp--utt与wav文件路径的映射
      - text--utt与音频对应的文本的映射
      - utt2spk--utt与对应的说话人id的映射；每条音频都有一个说话人，对应一个speaker is
      - spk2utt--说话人id与其所有utt的映射；音频数据集构建时，同一个说话人会不只生成一条音频，所以spk2utt中会记录一个说话人对应的所有utt
   2. [extract_embedding.py](./libritts/extract_embedding.py)--使用campplus.onnx模型从音频数据中抽取speaker embedding，并保存为utt2embedding.pt和spk2embedding.pt
      - utt2embedding.pt--记录从每条utt对应的音频文件中抽取的speaker embedding
      - spk2embedding.pt--一个说话人对应多个utt，将每个speaker is对应的所有embedding求平均，得到一个平均embedding，记录为spk2embedding.pt
   3. [extract_speech_token.py](./libritts/extract_speech_token.py)--使用CosyVoice中提供的speech tokenizer模型，从音频数据中抽取speech token，并保存为utt2speech_token.pt
   4. [make_parquet_list.py](./libritts/make_parquet_list.py)--