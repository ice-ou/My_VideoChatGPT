# Need to call this before importing transformers.
from video_chatgpt.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

# 注意，这里 
# "Head dim > 64 backward requires A100 or H100". 
# The forward for head dim <= 128, and backward for head dim <= 64 works on other GPUs.

# replace_llama_attn_with_flash_attn()

from video_chatgpt.train.train import train

if __name__ == "__main__":
    train()
