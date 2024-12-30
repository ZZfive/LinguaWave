# LinguaWave
一个探讨“将LLM与文本音频转换任务结合”的实验性项目

 - [简介](#简介)
 - [音频大语言模型](#音频大语言模型)
 - [语音生成](#语音生成)


## 简介
&emsp;&emsp;随着技术发展，多模态大模型工作越来越多，但大多是基于视觉的，然而最近将音频和大语言模型结合的工作越来越多，最近发布的[mini-omni](https://github.com/gpt-omni/mini-omni)、[GLM-4-Voice](https://github.com/THUDM/GLM-4-Voice)、[CosyVoice](https://github.com/FunAudioLLM/CosyVoice)、[InspireMusic](https://github.com/FunAudioLLM/InspireMusic)等模型已在开发工作中有所应用，目前本人更看好音频大语言模型，因此本项目旨在探索LLM与音频结合的开源项目，尝试从头复现一些有意思的工作，做一些实验性质的尝试。

&emsp;&emsp;GLM-4-Voice基本基于CosyVoice的架构，最近刚发布的Cosyvoice2也是进一步将之前定制化的LM模块简化为Qwen架构，GLM-4-Voice中的LM模块是自家的GLM-4-9B，然后使用Flow match训练。结构基本一致，一个用于TTS任务，一个用于语音聊天，其实本质上是训练数据性质的不同，就和mini-omni一样，同时支持ASR、TTS、语音聊天能力。本项目将基于此架构进行深入探究。

## 音频大语言模型
To be continued...

## 语音生成
To be continued...
