# CogVLM

<p align="center">
⚒️ <a href="https://github.com/THUDM/SwissArmyTransformer" target="_blank">SwissArmyTransformer (sat)</a> • 🐦 <a href="https://twitter.com/thukeg" target="_blank">Twitter</a> • 👋 Join us on <a href="assets/WECHAT.md" target="_blank">WeChat</a>
</p>

## Introduction

We introduce CogVLM, a powerful open-source visual language foundation model. Different from the popular shallow-align method which maps image features into the input space of language model, CogVLM bridges the gap between the frozen pretrained language model and image encoder by a trainable visual expert module in the attention and FFN layers. As a result, CogVLM enables deep fusion of visual language features without sacrificing any performance on NLP tasks. CogVLM-17B achieves state-of-the-art performance on 9 classic cross-modal benchmarks, including NoCaps, Flicker30k captioning, RefCOCO, RefCOCO+, RefCOCOg, Visual7W, GQA, ScienceQA, VizWiz VQA and TDIUC, and rank the 2nd on VQAv2, OKVQA, TextVQA, COCO captioning, etc., surpassing or matching PaLI-X 55B. Codes and checkpoints are available at Github.

## Examples

CogVLM is powerful for answering various types of visual questions, including **Detailed Description & Visual Question Answering**,  **Complex Counting**, **Visual Math Problem Solving**, **OCR-Free Reasonging**, **OCR-Free Visual Question Answering**, **World Knowledge**, **Referring Expression Comprehension**, **Programming with Visual Input**, **Grounding with Caption**, **Grouning Visual Question Answering**, etc.

![chat-example](assets/chat.png)

## Usage
### Inference

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python cli_demo.py --from_pretrained cogvlm-base-224 --version base --english --bf16 --no_prompt
python cli_demo.py --from_pretrained cogvlm-base-490 --version base --english --bf16 --no_prompt
python cli_demo.py --from_pretrained cogvlm-chat --version chat --english --fp16
python cli_demo.py --from_pretrained cogvlm-grounding-base --version base --english --bf16
python cli_demo.py --from_pretrained cogvlm-grounding-generalist --version base --english --bf16
# We also support model parallel inference, which splits model to multiple (2/4/8) GPUs.
torchrun --standalone --nnodes=1 --nproc-per-node=2 cli_demo.py --from_pretrained cogvlm-chat --version chat --english --fp16
```

If you have trouble in accessing huggingface.co, you can add `--local_tokenizer /path/to/vicuna-7b-v1.5` to load the tokenizer.

### Fine-tuning

Start by downloading the [Captcha Images dataset](https://www.kaggle.com/datasets/aadhavvignesh/captcha-images). Once downloaded, extract the contents of the ZIP file.

To create a train/validation/test split in the ratio of 80/5/15, execute the following:

```bash
python scripts/split_dataset.py
```

Kickstart the fine-tuning process with this command:

```bash
bash scripts/finetune_lora.sh
```

To evaluate the performance of your model, use:

```bash
bash scripts/evaluate.sh checkpoints/your_model_path
```

The anticipated results are approximately **94.53%** accuracy. When disregarding letter cases, the accuracy is around **95.13%**.

### Command Line Demo

```shell
python cli_demo.py 
```
The program will automatically download the sat model and interact in the command line. You can generate replies by entering instructions and pressing enter. Enter 'clear' to clear the conversation history and 'stop' to stop the program.

The program provides the following hyperparameters to control the generation process and quantization accuracy:
```
usage: cli_demo.py [-h] [--max_length MAX_LENGTH] [--top_p TOP_P] [--top_k TOP_K] [--temperature TEMPERATURE] [--english] [--quant {8,4}]

optional arguments:
  -h, --help            show this help message and exit
  --max_length MAX_LENGTH
                        max length of the total sequence
  --top_p TOP_P         top p for nucleus sampling
  --top_k TOP_K         top k for top k sampling
  --temperature TEMPERATURE
                        temperature for sampling
  --english             only output English
  --quant {8,4}         quantization bits
```

### Web Demo
We provide a [web demo](http://36.103.203.44:7861/) based on [Gradio](https://gradio.app).

![web_demo](assets/web_demo.png)

## Model Quantization
In the sat implementation, you need to change the loading location to 'cpu' first, and then perform quantization. Here's how, see cli_demo.py for details:
```python
from sat.quantization.kernels import quantize
model = quantize(model.transformer, args.quant).cuda()
# Specify model.transformer to only quantize ChatGLM, as the error is larger when quantizing ViT
```

## License

The code in this repository is open source under the Apache-2.0 license, while the use of the CogVLM model weights must comply with the Model License.

## Citation & Acknowledgements

If you find our work helpful, please consider citing the following papers
```

```
In the instruction fine-tuning phase of the CogVLM, there are some English image-text data from the [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4), [LLAVA](https://github.com/haotian-liu/LLaVA), [LRV-Instruction](https://github.com/FuxiaoLiu/LRV-Instruction), and [LLaVAR](https://github.com/SALT-NLP/LLaVAR) projects, as well as many classic cross-modal work datasets. We sincerely thank them for their contributions.