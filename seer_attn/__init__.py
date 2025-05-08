
from seer_attn.llama.modeling_llama_seerattn import SeerAttnLlamaForCausalLM
from seer_attn.qwen.modeling_qwen2_seerattn import SeerAttnQwen2ForCausalLM
from seer_attn.qwen_decode_sparse.modeling_qwen2 import SeerDecodingQwen2ForCausalLM
from seer_attn.phi_decode_sparse.modeling_phi3 import SeerDecodingPhi3ForCausalLM
__all__ = [
    "SeerAttnLlamaForCausalLM",
    "SeerAttnQwen2ForCausalLM",
    "SeerDecodingQwen2ForCausalLM",
    "SeerDecodingPhi3ForCausalLM",
]