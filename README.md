# InternLM-Tutoral-JICHUDAO    
August 2024     

# 第一课 书生·浦语大模型全链路开源开放体系
[书生·浦语大模型全链路开源开放体系](https://www.bilibili.com/video/BV18142187g5/)   

## 要点：
书生·浦语 全链条开源 与社区生态无缝衔接， 包括数据、预训练、微调、部署、评测和应用。
- 书生·万卷：首个开源多模态语料库
- InternEvo：轻量级大模型预训练框架
- Xtuner：轻量化大模型微调工具
- LMDeploy：LLM 轻量化高效部署工具
- InternLM：预训练大语言模型
- OpenCompass：客观评估大模型性能的开源工具
- Lagent：大语言模型智能体框架

![](./jcd1.png)

## 亮点
- 100万Token 上下文实现大海捞针（对比GPT-4O 12.8万）
- 开源数据处理工具Lable U，支持AI标注，加速数据标注效率
- 开源的智能体框架可以将大语言模型转变为多种类型的智能体，适合多种商业场景应用。

 ![](./jcd2.png)

## 总结    

 书生·浦语大模型全链路开源体系提供了一整套完整的开源体系，从数据、预训练、微调、部署、评测、应用等一系列工具与框架，帮助用户更好地参与到大模型的研究、开发与应用中。这些工具与框架的开源，也为大模型的发展提供了更多的机会和可能性，必将主力国内大模型应用的快速发展。


# 第二课 8G 显存玩转书生大模型 Demo       
2024.8.19    

[第2课文档](https://github.com/InternLM/Tutorial/blob/camp3/docs/L1/Demo/easy_readme.md)
[第2课 视频](https://www.bilibili.com/video/BV18x4y147SU/)    

## 关卡任务    

本关任务主要包括：
- InternLM2-Chat-1.8B 模型的部署（基础任务）
- InternLM-XComposer2-VL-1.8B 模型的部署（进阶任务）
- InternVL2-2B 模型的部署（进阶任务）

## 创建开发机     
选择 10% 的开发机，镜像选择为 Cuda-12.2。在输入开发机名称后，点击创建开发机。
在创建完成后，我们便可以进入开发机了！

## 环境配置    
已经在 `/root/share/pre_envs` 中配置好了预置环境 icamp3_demo

可以通过如下指令进行激活：   
``
conda activate /root/share/pre_envs/icamp3_demo
```    

## Cli Demo 部署 InternLM2-Chat-1.8B 模型（基础任务）    

创建一个目录，用于存放我们的代码。并创建一个 `cli_demo.py`。
```
mkdir -p /root/demo
touch /root/demo/cli_demo.py    
```

然后，我们将下面的代码复制到 `cli_demo.py` 中
```
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


model_name_or_path = "/root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, device_map='cuda:0')
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='cuda:0')
model = model.eval()

system_prompt = """You are an AI assistant whose name is InternLM (书生·浦语).
- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
"""

messages = [(system_prompt, '')]

print("=============Welcome to InternLM chatbot, type 'exit' to exit.=============")

while True:
    input_text = input("\nUser  >>> ")
    input_text = input_text.replace(' ', '')
    if input_text == "exit":
        break

    length = 0
    for response, _ in model.stream_chat(tokenizer, input_text, messages):
        if response is not None:
            print(response[length:], flush=True, end="")
            length = len(response)
```

通过 `python /root/demo/cli_demo.py` 来启动我们的 Demo。效果如下图所示：

 ![](./jcd4.png)

## Streamlit Web Demo 部署 InternLM2-Chat-1.8B 模型

执行如下代码来把本教程仓库 clone 到本地，以执行后续的代码。
```
cd /root/demo
git clone https://github.com/InternLM/Tutorial.git
```

然后，我们执行如下代码来启动一个 Streamlit 服务。
```
cd /root/demo
streamlit run /root/demo/Tutorial/tools/streamlit_demo.py --server.address 127.0.0.1 --server.port 6006
```

接下来，我们在本地的 PowerShell 中输入以下命令，将端口映射到本地。
```
ssh -CNg -L 6006:127.0.0.1:6006 root@ssh.intern-ai.org.cn -p 45394
```

在完成端口映射后，我们便可以通过浏览器访问 `http://localhost:6006` 来启动我们的 Demo。
效果如下图所示：
![](./jcd5.png)



