# app/llm/mistral_loader.py

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Model name (from Hugging Face)
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"

# Load tokenizer and model (only once, at startup)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.backends.mps.is_available() else torch.float32,
    device_map="auto"
)

# Create a text-generation pipeline using Mistral
llm = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    max_new_tokens=512,
    temperature=0.7,
    repetition_penalty=1.1,
)

def chat_with_mistral(prompt: str, system_prompt: str = None) -> str:
    """Send a prompt to the Mistral model in <s>[INST] ... [/INST]> format."""
    if system_prompt:
        full_prompt = f"<s>[INST] {system_prompt.strip()} [/INST] {prompt.strip()} </s>"
    else:
        full_prompt = f"<s>[INST] {prompt.strip()} [/INST]"

    output = llm(full_prompt)[0]["generated_text"]
    return output.strip()