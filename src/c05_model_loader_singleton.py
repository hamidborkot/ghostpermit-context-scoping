"""MODEL_OBJ/TOK_OBJ singleton loader for Qwen3-8B (never load twice in one session).

Source: ghost-p-p (5).ipynb, code cell 6 of 71.
Output file(s): see the cell body and the README in data/processed/ or data/tests/.
"""

MODEL_OBJ, TOK_OBJ = None, None
def get_model():
    global MODEL_OBJ, TOK_OBJ
    if MODEL_OBJ is not None:               # never load twice in one session
        return MODEL_OBJ, TOK_OBJ
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tk = AutoTokenizer.from_pretrained(CONFIG["MODEL_ID"])
    if tk.pad_token is None: tk.pad_token = tk.eos_token
    tk.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(
        CONFIG["MODEL_ID"], dtype=torch.float16, device_map="auto",
        attn_implementation="sdpa", trust_remote_code=True)
    m.eval()
    MODEL_OBJ, TOK_OBJ = m, tk
    print("loaded once:", CONFIG["MODEL_ID"])
    return m, tk

model, tk = get_model()
