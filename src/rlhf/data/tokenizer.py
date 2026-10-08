"""The GPT-2 tokenizer (via tiktoken) and the special token ids every phase shares."""

import tiktoken

EOT_ID = 50256  # <|endoftext|>: separates documents in pretraining, ends every summary
PAD_ID = 50257  # our own pad token: an unused slot in the padded 50304 vocab, never trained on
N_REAL_TOKENS = 50257  # ids 0..50256 are real GPT-2 tokens; 50257..50303 are never real text


def get_tokenizer() -> tiktoken.Encoding:
    return tiktoken.get_encoding("gpt2")
