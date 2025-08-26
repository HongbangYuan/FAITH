from typing import Union, Literal
from langchain.chat_models import ChatOpenAI
# from langchain import OpenAI
from langchain.llms import OpenAI
from langchain.schema import (
    HumanMessage
)
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from utils.config_utils import get_model_name_mapping


class AnyOpenAILLM:
    def __init__(self, *args, **kwargs):
        # Determine model type from the kwargs
        model_name = kwargs.get('model_name', 'gpt-3.5-turbo')
        if model_name.split('-')[0] == 'text':
            self.model = OpenAI(*args, **kwargs)
            self.model_type = 'completion'
        else:
            self.model = ChatOpenAI(*args, **kwargs)
            self.model_type = 'chat'

    def __call__(self, prompt: str):
        if self.model_type == 'completion':
            return self.model(prompt)
        else:
            return self.model(
                [
                    HumanMessage(
                        content=prompt,
                    )
                ]
            ).content


def load_model_and_tokenizer(model_name, use_docker):
    model_name_mapping = get_model_name_mapping(use_docker)
    if model_name not in model_name_mapping.keys():
        raise ValueError(f"{model_name} not in pre-defined models!")
    model_name_or_path = model_name_mapping[model_name]
    model, tokenizer = load_llama_model_and_tokenizer(model_name_or_path)
    return model, tokenizer


def load_llama_model_and_tokenizer(model_name_or_path):
    # load models
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        device_map='balanced',
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_fast=False
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = 'left'

    return model, tokenizer


def load_llama_tokenizer(model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_fast=False
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = 'left'

    return tokenizer
