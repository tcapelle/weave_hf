import weave
from dataclasses import dataclass, field
from pydantic import model_validator
from transformers import AutoModelForCausalLM
from transformers import PreTrainedModel, PreTrainedTokenizerBase, AutoTokenizer
import simple_parsing


class MistralWeaveModel(weave.Model):
    """A model class for MetaAI-LLa ma models"""
    model_id: str
    device: str
    temperature: float = 0.5
    max_new_tokens: int = 128
    model: PreTrainedModel  # we may want to support LoRA fine-tunes
    tokenizer: PreTrainedTokenizerBase

    @model_validator(mode='before')
    def load_model_and_tokenizer(cls, v):
        "Pydantic validator to load the model and the tokenizer"
        if not v.get("model_id"):
            raise ValueError("model_id is required")
        v["model"] = AutoModelForCausalLM.from_pretrained(v["model_id"], device_map=v["device"], torch_dtype="auto")
        v["tokenizer"] = AutoTokenizer.from_pretrained(v["model_id"])
        return v

    @weave.op()
    def format_prompt(self, messages: list[dict[str, str]]) -> str:
        "A simple function to apply the chat template to the prompt"
        return  self.tokenizer.apply_chat_template(messages, tokenize=False)

    @weave.op()
    def predict(self, messages: list[dict[str, str]]) -> str:
        formatted_prompt = self.format_prompt(messages)
        tokenized_prompt = self.tokenizer.encode(formatted_prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            tokenized_prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
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
    model = MistralWeaveModel(model_id=args.model_id, device=args.device)
    print(model.predict([{"role": "user", "content": args.prompt}]))

