# CosyVoice原理解析
&emsp;&emsp;在实际训练之前，需要更进一步对CosyVoice的架构、训练过程等分析，加强理解，便于后续的复现。

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
&emsp;&emsp;