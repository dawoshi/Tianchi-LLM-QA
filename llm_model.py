import os
import torch
import time

from config import *
from vllm import LLM, SamplingParams

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig
from qwen_generation_utils import make_context, decode_tokens, get_stop_words_ids


os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEVICE = LLM_DEVICE
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE

IMEND = "<|im_end|>"
ENDOFTEXT = "<|endoftext|>"

def get_stop_words_ids(chat_format, tokenizer):
    if chat_format == "raw":
        stop_words_ids = [tokenizer.encode("Human:"), [tokenizer.eod_id]]
    elif chat_format == "chatml":
        stop_words_ids = [[tokenizer.im_end_id], [tokenizer.im_start_id]]
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")
    return stop_words_ids

def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

class ChatLLM(object):

    def __init__(self, model_path):
       self.tokenizer = AutoTokenizer.from_pretrained(
                        model_path,
                        pad_token='<|extra_0|>',
                        eos_token='<|endoftext|>',
                        padding_side='left',
                        trust_remote_code=True
                    )
       self.generation_config = GenerationConfig.from_pretrained(model_path, pad_token_id=self.tokenizer.pad_token_id)
       self.tokenizer.eos_token_id = self.generation_config.eos_token_id
       self.stop_words_ids = []
       self.model = LLM(model=model_path,
                            tokenizer=model_path,
                            tensor_parallel_size=1,
                            trust_remote_code=True,
                            gpu_memory_utilization=0.90,
                            dtype="bfloat16")
       for stop_id in get_stop_words_ids(self.generation_config.chat_format, self.tokenizer):
            self.stop_words_ids.extend(stop_id)
       self.stop_words_ids.extend([self.generation_config.eos_token_id])
       sampling_kwargs = {
            "stop_token_ids": self.stop_words_ids,
            "early_stopping": False,
            "top_p": 1.0,
            "top_k": -1 if self.generation_config.top_k == 0 else self.generation_config.top_k,
            "temperature": 0.0,
            "max_tokens": 2000,
            "repetition_penalty": self.generation_config.repetition_penalty,
            "n":1,
            "best_of":2,
            "use_beam_search":True
       }
       self.sampling_params = SamplingParams(**sampling_kwargs)

    def infer(self, prompts):
       batch_text = []
       for q in prompts:
            raw_text, _ = make_context(
              self.tokenizer,
              q,
              system="You are a helpful assistant.",
              max_window_size=self.generation_config.max_window_size,
              chat_format=self.generation_config.chat_format,
            )
            batch_text.append(raw_text)
       outputs = self.model.generate(batch_text,
                                sampling_params = self.sampling_params
                               )
       batch_response = []
       for output in outputs:
           output_str = output.outputs[0].text
           if IMEND in output_str:
               output_str = output_str[:-len(IMEND)]
           if ENDOFTEXT in output_str:
               output_str = output_str[:-len(ENDOFTEXT)]
           batch_response.append(output_str)
       torch_gc()
       return batch_response

if __name__ == "__main__":
    qwen7 = "/root/autodl-tmp/codes/pre_train_model/Qwen-7B-Chat"
    start = time.time()
    llm = ChatLLM(qwen7)
    test = ["吉利汽车座椅按摩","吉利汽车语音组手唤醒","自动驾驶功能介绍"]
    generated_text = llm.infer(test)
    print(generated_text)
    end = time.time()
    print("cost time: " + str((end-start)/60))
