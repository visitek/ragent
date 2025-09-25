#  Copyright (c) 2025 Martin Lonsky (martin@lonsky.net)
#  All rights reserved.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from typing import List, Dict


class ModelInterface:
    def generate(self, prompt: List[Dict[str, str]]) -> str:
        raise NotImplementedError("Subclasses must implement generate()")


class HuggingFaceModel(ModelInterface):
    def __init__(
            self,
            model_name: str = '',
            device: str = None,
            max_new_tokens: int = 2048,
            temperature: float = 0.3
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"  # For Apple Silicon
            else:
                device = "cpu"

        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # @TODO tune for CUDA device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.bfloat16,
        )

    def generate(self, prompt: List[Dict[str, str]]) -> str:
        input_ids = self.tokenizer.apply_chat_template(
            prompt,
            return_tensors="pt",
            thinking=True,
            return_dict=True,
            add_generation_prompt=True,
        ).to(self.device)

        set_seed(42)
        generation_kwargs = {
            **input_ids,
            "max_new_tokens": self.max_new_tokens,
        }

        # Only apply temperature when it's greater than 0
        if self.temperature > 0:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = self.temperature

        output = self.model.generate(**generation_kwargs)

        return self.tokenizer.decode(output[0, input_ids["input_ids"].shape[1]:], skip_special_tokens=True)


class ModelFactory:
    @staticmethod
    def create(model_type: str = "huggingface", **kwargs) -> ModelInterface:
        if model_type.lower() == "huggingface":
            return HuggingFaceModel(**kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
