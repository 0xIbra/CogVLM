from sat.model.mixins import CachedAutoregressiveMixin
from sat.mpu import get_model_parallel_world_size

from utils.parser import parse_response
from utils.chat import chat
from models.cogvlm_model import CogVLMModel
from utils.language import llama2_tokenizer, llama2_text_processor_inference
from utils.vision import get_image_processor

from PIL import Image

import base64
import json
import hashlib
import argparse
import torch
import time
import re
import os


world_size = int(os.environ.get('WORLD_SIZE', 1))


def process_image_without_resize(image_prompt):
    image = Image.open(image_prompt)
    # print(f"height:{image.height}, width:{image.width}")
    timestamp = int(time.time())
    file_ext = os.path.splitext(image_prompt)[1]
    filename_grounding = f"examples/{timestamp}_grounding{file_ext}"
    return image, filename_grounding


class ModelWrapper:

    def __init__(self, id: str = "cogvlm-chat", max_length: int = 2048, fp16: bool = True, bf16: bool = True, local_tokenizer: str = "lmsys/vicuna-7b-v1.5"):
        self.model_id = id
        self.max_length = max_length
        self.fp16 = fp16
        self.bf16 = bf16
        self.local_tokenizer = local_tokenizer

        self.load_model()

    def load_model(self):
        model, model_args = CogVLMModel.from_pretrained(
            self.model_id,
            args=argparse.Namespace(
                deepspeed=None,
                local_rank=0,
                rank=0,
                world_size=world_size,
                model_parallel_size=world_size,
                mode='inference',
                fp16=self.fp16,
                bf16=self.bf16,
                skip_init=True,
                use_gpu_initialization=True,
                device='cuda:0'
            ),
            overwrite_args={'model_parallel_size': world_size} if world_size != 1 else {}
        )
        model = model.eval()
        assert world_size == get_model_parallel_world_size(), "world size must equal to model parallel size for cli_demo!"

        self.model = model
        tokenizer = llama2_tokenizer(self.local_tokenizer, signal_type="chat")
        self.image_processor = get_image_processor(model_args.eva_args["image_size"][0])

        self.model.add_mixin('auto-regressive', CachedAutoregressiveMixin())

        self.text_processor_infer = llama2_text_processor_inference(tokenizer, self.max_length, self.model.image_length)

    def predict(self, input_text: str, image: Image.Image, temperature: float = 0.8, top_p: float = 0.4, top_k: int = 5):
        with torch.no_grad():
            response, _, cache_image = chat(
                image_path="",
                model=self.model,
                text_processor=self.text_processor_infer,
                img_processor=self.image_processor,
                query=input_text,
                history="",
                image=image,
                max_length=self.max_length,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                invalid_slices=self.text_processor_infer.invalid_slices if hasattr(self.text_processor_infer, "invalid_slices") else [],
                no_prompt=False
            )

        return response
