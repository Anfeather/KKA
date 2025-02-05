from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from peft import PeftModel

if "__main__" == __name__:
    # Set the base model path
    base_model_path = '/home/zhengqing/glm-4-9b-chat'  # Foundation model path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the tokenizer and model for the base model
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
    model = model.to(device)

    # Enter commands to generate flower descriptions
    prompt = "Question: Generate descriptions of different flowers. The sentences should follow the exact structure and style of the example below: the flower has sparse pink petals that are thin and small.\nAnswer: "

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    generated_ids = inputs.input_ids
    with torch.no_grad():
        for _ in range(2000):  # Control the length of the build
            outputs = model(input_ids=generated_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
            if next_token_id == tokenizer.eos_token_id:
                break

    description = tokenizer.decode(generated_ids[0].cpu(), skip_special_tokens=True)  
    print(f"Generated Description: {description}")
