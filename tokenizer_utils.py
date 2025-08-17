import torch
import tiktoken

# Load GPT-2 tokenizer 
tokenizer = tiktoken.get_encoding("gpt2")

CONTEXT_LENGTH = 1024
EOT = 50256  # GPT-2 end-of-text

def text_to_token_ids(text, tokenizer=tokenizer, context_length=CONTEXT_LENGTH):
    # Do NOT pad for generation. Only clip to the last context_length tokens.
    ids = tokenizer.encode(text)
    if len(ids) > context_length:
        ids = ids[-context_length:]
    return torch.tensor([ids], dtype=torch.long)

def token_ids_to_text(token_ids, tokenizer=tokenizer):
    # Make shapes friendly
    if token_ids.ndim == 3:
        token_ids = token_ids.squeeze(0)
    if token_ids.ndim == 2:
        token_ids = token_ids[0]
    # Drop any EOS that may appear
    ids = [int(t) for t in token_ids.tolist() if t != EOT]
    return tokenizer.decode(ids)

