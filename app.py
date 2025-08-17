import gradio as gr
import torch
from model import GPTModel
from tokenizer_utils import text_to_token_ids, token_ids_to_text, tokenizer
from huggingface_hub import hf_hub_download  # i have used huggingface_face because hugging face have limit of 1 GB so i used Hub


# Load Weights and Biases
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

CHOOSE_MODEL = "gpt2-medium (355M)"
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])


device = "cuda" if torch.cuda.is_available() else "cpu"
model = GPTModel(BASE_CONFIG).to(device)

# Download weights from Hugging Face Model repo
checkpoint_path = hf_hub_download(
    repo_id="Shiva8164/MyGPT2Weights",   # your model repo
    filename="gpt2-medium355M-sft.pth"   # the file you uploaded
)

# Load the checkpoint
state_dict = torch.load(checkpoint_path, map_location=device)

# Fixing missing bias if not in checkpoint
if "out_head.bias" not in state_dict:
    print("⚠️ Adding missing out_head.bias as zeros")
    state_dict["out_head.bias"] = torch.zeros(model.out_head.bias.shape)

# Load weights
model.load_state_dict(state_dict, strict=True)
model.eval()



# Generation function

import torch

def generate_response(instruction, input_text, max_new_tokens=100, temperature=0.8, top_k=50):
    # Format prompt
    prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    
    # Encode prompt
    prompt_ids = tokenizer.encode(prompt)
    generated = list(prompt_ids)

    for _ in range(max_new_tokens):
        # Feed only the last context window
        x = torch.tensor([generated[-BASE_CONFIG["context_length"]:]], device=device)

        with torch.inference_mode():
            logits = model(x)  # shape [1, vocab]

        # Apply temperature
        logits = logits[0] / temperature

        # top-k sampling 
        if top_k is not None:
            top_vals, top_idx = torch.topk(logits, top_k)
            probs = torch.softmax(top_vals, dim=-1)
            next_token = top_idx[torch.multinomial(probs, 1)].item()
        else:
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

        generated.append(next_token)

        # Stop on EOS
        if next_token == 50256:
            break

    # Decode only what was generated after the prompt
    new_tokens = [t for t in generated[len(prompt_ids):] if t != 50256]
    output_text = tokenizer.decode(new_tokens).strip()

    # Fallback: extract from the whole text
    if not output_text:
        full = tokenizer.decode([t for t in generated if t != 50256])
        if "### Response:" in full:
            output_text = full.split("### Response:")[-1].strip()

    return output_text or "(no output)"


# Gradio Interface
demo = gr.Interface(
    fn=generate_response,
    inputs=[
        gr.Textbox(label="Instruction", placeholder="Enter task instruction here..."),
        gr.Textbox(label="Input Text", placeholder="Enter input/context here...")
    ],
    outputs=gr.Textbox(label="Response"),
    title="Custom GPT-2 Instruction Model",
    description="Enter an instruction and input text, the model will generate a response."
)

if __name__ == "__main__":
    demo.launch()
