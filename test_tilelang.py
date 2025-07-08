from transformers import AutoTokenizer, AutoConfig
from seer_attn.decode_sparse.qwen3.modeling_qwen3_seerattn_tilelang import SeerDecodingQwen3ForCausalLM
import torch

## SeerAttention-R: sparse decoding 
model_name = "SeerAttention/SeerAttention-Decode-Qwen3-4B-AttnGates"
config = AutoConfig.from_pretrained(model_name)
# print(config.base_model)

tokenizer = AutoTokenizer.from_pretrained(
    config.base_model, 
    padding_side="right",
)
## Token budget based sparsity selection. You can also use threshold method
model = SeerDecodingQwen3ForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    seerattn_sparsity_method='token_budget',
    seerattn_token_budget=4096,
).cuda()

prompt1 = "The twelve letters $A$,$B$,$C$,$D$,$E$,$F$,$G$,$H$,$I$,$J$,$K$, and $L$ are randomly grouped into six pairs of letters. The two letters in each pair are placed next to each other in alphabetical order to form six two-letter words, and then those six words are listed alphabetically. For example, a possible result is $AB$, $CJ$, $DG$, $EK$, $FL$, $HI$. The probability that the last word listed contains $G$ is $\\frac mn$, where $m$ and $n$ are relatively prime positive integers. Find $m+n$."
prompt2 = "Let $k$ be a real number such that the system \\begin{align*} &|25 + 20i - z| = 5 \\ &|z - 4 - k| = |z - 3i - k| \\end{align*} has exactly one complex solution $z$. The sum of all possible values of $k$ can be written as $\\frac{m}{n}$, where $m$ and $n$ are relatively prime positive integers. Find $m + n$. Here $i = \\sqrt{-1}$.$"

messages = [
    [{"role": "user", "content": prompt1}],
    [{"role": "user", "content": prompt1}],
    [{"role": "user", "content": prompt1}],
]

prompts = []

for message in messages:
    prompts.append(tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True))

inputs = tokenizer(prompts, return_tensors="pt", add_special_tokens=True, padding="longest")
inputs = {key: value.to(model.device) for key, value in inputs.items()}
print("inputs.keys():", inputs.keys())


generated_ids = model.static_cache_generate(
    **inputs,
    max_length=200,
    do_sample=False,
)
generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
for i, text in enumerate(generated_text):
    print("output", i + 1, ":", text)
