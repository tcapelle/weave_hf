# pip install transformers weave simple_parsing setencepiece

import weave
from dataclasses import dataclass, field
import simple_parsing
from transformers import pipeline


@dataclass
class Args:
    model_id: str = "mistralai/Mistral-7B-Instruct-v0.3"
    prompt: str = "Tell me what is the best cheese, and why it is Comte"
    project: str = "hf_weave"
    device: str = "cuda:0"

args = simple_parsing.parse(Args)

# we define the pipeline outside of the op, so that we don't reload the model each time
pipe = pipeline("text-generation", model=args.model_id, device_map=args.device, torch_dtype="auto")


# simple decoration of the pipeline.__call__ method, ideally we would want a callback to trace the prompt, tokenizer and decode
@weave.op()
def simple_generate(pipe, prompt:str, temperature: float=0.7, max_new_tokens: int=64, do_sample: bool=True, return_full_text: bool=False) -> str:
    messages = [
        {"role": "user", "content": prompt},
    ]
    out = pipe(messages, temperature=temperature, max_new_tokens=max_new_tokens, do_sample=do_sample, return_full_text=return_full_text)
    return out[0]["generated_text"]

weave.init(args.project)
print(simple_generate(pipe, args.prompt))
