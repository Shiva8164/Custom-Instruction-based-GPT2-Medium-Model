# 📚 Custom GPT-2 Instruction Model (PyTorch + Gradio)

A from-scratch GPT-2–style decoder-only transformer fine-tuned for instruction-following.  
Ships with a Gradio app for instant, interactive generation and auto-downloads weights from Hugging Face Hub.

---

## 🚀 Project Highlights
- 🧠 Pure PyTorch stack: custom Multi-Head Attention, FeedForward (GELU), LayerNorm, residuals, and causal mask  
- 📝 Instruction format: `### Instruction ... ### Input ... ### Response ...` for structured outputs  
- 🌐 Gradio demo: type an instruction + input and get a response in the browser  
- ⚡ GPU/CPU friendly: runs on CUDA if available, otherwise CPU  

---

## 🏗️ Model Overview
- **Architecture:** GPT-2 style decoder-only transformer  
- **Variants Supported:** gpt2-small (124M), gpt2-medium (355M), gpt2-large (774M), gpt2-xl (1558M)  
- **Default:** gpt2-medium (355M)  
- **Config (medium):** `emb_dim=1024, n_layers=24, n_heads=16, vocab_size=50257, context_length=1024, dropout=0.0, qkv_bias=True`  
- **Tokenizer:** GPT-2 (tiktoken)  
- **Objective:** next-token language modeling on instruction-formatted data  
- **Sampling:** temperature, top-k with EOS stop at token `50256`  

---
