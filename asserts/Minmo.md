# Minmo原理解析
Minmo是阿里FunAudioLLM团队开发的音频大语言模型，能接收文本、音频输入，并以端到端的方式输出高质量音频。  

 - [介绍](#介绍)
 - [架构](#架构)
 - [任务及训练数据](#任务及训练数据)
 - [训练](#训练)

## 介绍
&emsp;&emsp;Minmo是端到端的对齐多模态架构，具有音频理解、音频生成和端到端全双工语音交互能力，通过探索使用文本模型的隐藏层via哦是作为语音解码器的输入来对齐音频输出模态，能生成指定的情绪、方言、语速以及模仿特定音色。

## 架构
&emsp;&emsp;Minmo架构如图1所示，除了用于合成音频的Token2wav Synthesizer不参与训练外，参与训练的模块主要有以下六块，总参数量约为8B。在 L20 GPU 上进行测试时，从接收用户音频输入到提供音频响应的端到端延迟约为 600 毫秒。

 1. Voice Encoder：由预训练的SenseVoice-Large音频理解模型的编码器初始化而来，参数量约为636M；具有强将的声音理解能力，支持多语言识别、情绪识别和音频事件检测。
 2. Input Projector：由两个Transfoemr层和一个CNN层组成，用于进行维度转换和2倍下采样，随机初始化，参数量约为170M
 3. Large Language Model：由Qwen2.5-7B-instruct模型初始化而来，参数量约为7B；基于输入的音频特征或预测出的文本tokens预测新的文本tokens
 4. Output Projector：用于维度转换的线形层，参数量约为6M；对3输出的隐向量进行维度转换
 5. Voice Token LM：由预训练的CosyVoice2中的LLM初始化而来，参数量约为370M；基于来自LLM的隐向量特征或新预测出的音频tokens预测新的音频tokens
 6. Full Duplex Predictor：由一个Transfoemr层和一个linear-softmax输出层组成，随机初始化，参数量约为18M；该模块实时预测是否响应用户指令，或临时暂停当前系统生成以便处理用户后续的音频输入。当全双工预测器判定需进行系统响应时，MinMo将同步执行文本输出生成与逐token的音频tokens生成。

![enter image description here](images/Minmo.png?raw=true)
*图1: Minmo模型整体架构*

&emsp;&emsp;为了实现自然的语音响应，Minmo构建了一种流式声音解码机制，将LLM的文本输出转换为音频；在图1中所示，voice decoder由output projector、voice token LM和token2wav Synthesizer组成。

&emsp;&emsp;LLM模块输出的文本隐藏向量包含丰富的上下文信息，但在语义上是模糊的；但采样出来的文本tokens与生成的文本一致且语义更加清晰；同时，来自当前轮用户输入中的隐藏向量包含显示指令信息。在每个对话轮次中，用户输入的嵌入与LLM最后一层输出的隐藏状态将沿特征维度拼接，形成查询嵌入。随后，该查询嵌入将与五个采样出的文本tokens嵌入及其对应的LLM最后一层隐藏状态沿**序列维度**拼接，再输入到output projector，output projector的输出被称为**语义向量**。

&emsp;&emsp;voice token LM以自回归的方式，对交错的文本和语音tokens序列进行操作，生成speech tokens。具体来说，以 5:15 的固定比率混合语义向量和语音标记，即每五个语义向量后面跟着 15 个语音tokens。使用teacher forcing策略进行训练，并引入特征标记来标识何时该拼接下一次的语义向量。一旦LLM完成文本tokens预测，即语义向量耗尽，会插入“turn of speech”特殊token，表示voice token LM后续只预测语音tokens。当生成“end of speech”标记时，语音合成过程结束。

&emsp;&emsp;直接使用CosyVoice2中token2wave Synthesizer，由一个chunk-aware流匹配模型和声码器组成，能基于每个包含15个speech tokens的chunk生成音频。

&emsp;&emsp;voice decoder的理论延迟由以下公式表示：
$$Latency=5d_{llm}+15d_{lm}+15d_{syn} \tag1$$
其中$d_{llm}$LLM预测出单个text token的耗时，$d_{lm}$是voice token LM预测出单个speech token的耗时，$d_{syn}$是token2wav Synthesizer基于单个speech token生成音频的耗时。

## 任务及训练数据
&emsp;&emsp;Minmo的训练任务包括四种，分别是Speech-to-Text、Text-to-Speech、Speech-to-Speech和Speech-to-ControlToken，具体详情如下表所示。

![enter image description here](images/Minmo_data.png?raw=true)
*表1: Minmo训练任务及数据详情*

&emsp;&emsp;Speech-to-Text类型下各种任务的训练数据通过ChatML格式构建，如下所示；其中task_instrcution对应不同任务的自然语言描述，例如Speech Transcription用于语音识别任务，Translate {SRC LANG} into {TGT LANG}用于语音翻译任务；wav_path表示输入音频文件路径，task_output表示每个任务的输出内容

![enter image description here](images/Minmo_ChatML.png?raw=true)
*图2: ChatML格式案例*

&emsp;&emsp;Text-to-Speech tasks类型任务主要由基于语义合成数据构成，与训练CosyVoice2的数据相同。除了常规数据外，还包含1000小时的受指令控制生成的音频数据。

&emsp;&emsp;Speech-to-Speech任务数据主要来自模拟，包括大约 10,000 小时多轮对话语音和 100 小时风格可控的多轮对话语音。

&emsp;&emsp;Speech-to-ControlToken任务数据包含两个部分，一部分从现有的真实语音交互数据中抽取，另一部分是通过文本对话数据模拟的。构建双工训练数据时，使用启发式规则对样本进行双工标签注释，如下所示
 - 在assistant回答时，以user轮的endpoint作为起点
 - 在user talking时，在assistant回答完后的时间间隔T后作为起点，$T \in N(0.6,0.4^2)$
 - 对于用户的反馈信号（back-channel），从语音交互数据中筛选出用户（将对话中的一方视为用户）无法打断另一方说话的实例，并将其作为用户反馈信号的训练样本

## 训练