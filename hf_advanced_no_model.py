import weave
from dataclasses import dataclass, field
from pydantic import model_validator
from transformers import AutoModelForCausalLM
from transformers import PreTrainedModel, PreTrainedTokenizerBase, AutoTokenizer
import simple_parsing


class LLMModel:
    """A model class for a PreTrained Causal LLM"""

    def __init__(self, model_id: str, device: str):
        self.model_id = model_id
        self.device = device
        # load the model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device, torch_dtype="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    @weave.op()
    def format_prompt(self, messages: list[dict[str, str]]) -> str:
        "A simple function to apply the chat template to the prompt"
        return  self.tokenizer.apply_chat_template(messages, tokenize=False)

    @weave.op()
    def generate(self, messages: list[dict[str, str]], temperature: float = 0.5, max_new_tokens: int = 128) -> str:
        formatted_prompt = self.format_prompt(messages)
        tokenized_prompt = self.tokenizer.encode(formatted_prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            tokenized_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True, 
        )
        generated_text = self.tokenizer.decode(outputs[0][len(tokenized_prompt[0]):], skip_special_tokens=True)
        return {"generated_text": generated_text}




if __name__ == "__main__":
    
    @dataclass
    class Args:
        model_id: str = field(default="mistralai/Mistral-7B-Instruct-v0.3")
        prompt: str = field(default="What is the capital of France?")
        project: str = "hf_weave"
        device: str = "cuda:0"

    args = simple_parsing.parse(Args)

    weave.init(args.project)
    model = LLMModel(model_id=args.model_id, device=args.device)
    print(model.predict([{"role": "user", "content": args.prompt}], temperature=0.5, max_new_tokens=128))

