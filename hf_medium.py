import weave
from dataclasses import dataclass
from transformers import AutoModelForCausalLM
from transformers import PreTrainedModel, PreTrainedTokenizerBase, AutoTokenizer
import simple_parsing



class FineTunedModelHF(weave.Model): 
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase

    @weave.op()
    def format_prompt(self, messages: list[dict[str, str]]) -> str:
        "A simple function to apply the chat template to the prompt"
        return  self.tokenizer.apply_chat_template(messages, tokenize=False)

    @weave.op()
    def predict(self, messages: list[dict[str, str]]) -> str:
        formatted_prompt = self.format_prompt(messages)
        tokenized_prompt = self.tokenizer.encode(formatted_prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            tokenized_prompt,
            max_new_tokens=32,
            do_sample=True,
            temperature=0.7,
        )
        generated_text = self.tokenizer.decode(outputs[0][len(tokenized_prompt[0]):], skip_special_tokens=True)
        return {"generated_text": generated_text}


if __name__ == "__main__":
    
    @dataclass
    class Args:
        model_id: str = "mistralai/Mistral-7B-Instruct-v0.3"
        prompt: str = "What's the weather like today in Paris?"
        project: str = "hf_weave"
        device: str = "cuda:0"

    args = simple_parsing.parse(Args)

    weave.init(args.project)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype="auto", device_map=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = FineTunedModelHF(model=model, tokenizer=tokenizer)
    print(model.predict([{"role": "user", "content": args.prompt}]))
