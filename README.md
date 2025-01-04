# LinguaWave
一个探讨“将LLM与文本音频转换任务结合”的实验性项目

 - [简介](#简介)
 - [语音生成](#语音生成)
 - [音频大语言模型](#音频大语言模型)


## 简介
&emsp;&emsp;随着技术发展，多模态大模型工作越来越多，但大多是基于视觉的，然而最近将音频和大语言模型结合的工作越来越多，最近发布的[mini-omni](https://github.com/gpt-omni/mini-omni)、[GLM-4-Voice](https://github.com/THUDM/GLM-4-Voice)、[CosyVoice](https://github.com/FunAudioLLM/CosyVoice)、[InspireMusic](https://github.com/FunAudioLLM/InspireMusic)等模型已在开发工作中有所应用，目前本人更看好音频大语言模型，因此本项目旨在探索LLM与音频结合的开源项目，尝试从头复现一些有意思的工作，做一些实验性质的尝试。

&emsp;&emsp;GLM-4-Voice基本基于CosyVoice的架构，最近刚发布的Cosyvoice2也是进一步将之前定制化的LM模块简化为Qwen架构，GLM-4-Voice中的LM模块是自家的GLM-4-9B，然后使用Flow matching训练。结构基本一致，一个用于TTS任务，一个用于语音聊天，其实本质上是训练数据性质的不同，就和mini-omni一样，同时支持ASR、TTS、语音聊天能力。本项目将针对此架构进行深入探究。

&emsp;&emsp;GLM-4-Voice和CosyVoice的主体架构基本一致，使用Flow matching训练是借鉴[Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS)。GLM-4-Voice和CosyVoice本质都是通过训练将文本和语音在隐空间中对齐，但CosyVoice是将文本映射到音频的隐空间，LLM模块只会预测speech token，GLM-4-Voice则更接近mini-omni，隐空间是文本和音频共存的，LLM模块会同时预测text token和speech token。从实现难易度来看，CosyVoice更容易复现，因为CosyVoice项目中提供了训练代码，目前公布了CosyVoice的训练代码，CosyVoice2还未公布，GLM-4-Voice应该是不会公布训练代码。目前的项目是先尝试在小批量数据上复现类CosyVoice2架构模型训练，再尝试复现类GLM-4-Voice模型。

## 语音生成
&emsp;&emsp;基于CosyVoice项目公布的信息对CosyVoice进行SFT或全量训练应该问题不大，但CosyVoice2中将LM模块简化为Qwen架构，并且优化了speech tokenizer，基于其进行复现的价值应该更大，故目前是想将Qwen架构的LLM模块用其他的LLM模型替换，如Internlm或Llama等，通过训练复现类CosyVoice2架构的语音生成模型，既证明CosyVoice工作的有效性，也提高个人对多模态LLM，音频生成等方面的理解。

&emsp;&emsp;CosyVoice系列模型中推理过程主要包括三个模块，分别是llm模块、flow模块和hift模块，llm模块可简单理解为将文本和语音对齐，然后预测speech tokens，flow模块是基于预测的speech tokens转换而来的特征向量生成音频mel谱图，hift则是将mel谱图转换为音频。此外，还有一个用于从音频中提取speech tokens的speech tokenizer模块。目前的想法是从头对llm模块和flow模块进行训练，而speech tokenizer和hift模块可以直接复用CosyVoice项目中提供的预训练模型。

待办事项
 - [ ] 先基于CosyVoice项目公布的信息，使用libritts数据集对原始CosyVoice模型中的llm模块和flow模块从头开始训练，了解熟悉训练数据构建、模型构建、模型训练全过程
 - [ ] 收集更多数据，构建更丰富的训练数据集和sft数据集
 - [ ] 基于构建的数据集，使用新的LLM模型替换CosyVoice项目中的llm模块，从头训练llm模块和flow模块
 - [ ] 将各个模块组合，构建可推理的完成模型

## 音频大语言模型
To be continued...
