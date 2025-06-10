

from typing import Optional, List
import torch
from transformers import DynamicCache
from transformers.generation.logits_process import TopPLogitsWarper
import time
def batch_exist_generate(
    model,
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.LongTensor] = None,
    max_length: int = 100,
    do_sample: bool = False,
    **model_kwargs,
):
    """
    Modified generation loop that dynamically adjusts batch size by filtering finished sequences
    and reorganizing the KV cache.
    """
    # Initialize variables
    generation_config, model_kwargs = model._prepare_generation_config(None)
    generated = input_ids

    if isinstance(generation_config.eos_token_id, list):
        eos_token_id = generation_config.eos_token_id[0]
        eos_token_ids = torch.tensor(generation_config.eos_token_id, device=input_ids.device)
    else:
        eos_token_id = generation_config.eos_token_id
    initial_batch_size = input_ids.shape[0]

    device = input_ids.device
    finished = torch.zeros(initial_batch_size, dtype=torch.bool, device=device)
    current_cache = DynamicCache()
    
    cur_input = generated
    cur_to_orig = torch.arange(initial_batch_size, device=device)

    if do_sample:
        top_p_warper = TopPLogitsWarper(top_p=generation_config.top_p, min_tokens_to_keep=1)

    for step in range(max_length - generated.shape[1]):
        # Forward pass: get next token logits and updated past_key_values
        with torch.no_grad():
            outputs = model(
                cur_input, 
                attention_mask=attention_mask,
                past_key_values=current_cache, 
                use_cache=True,
                logits_to_keep=1,
        )
            

        logits = outputs.logits[:, -1, :].clone().float()
        logits = logits.to(input_ids.device)

        if do_sample:
            logits /= generation_config.temperature
            processed_logits = top_p_warper(cur_input, logits)
            probs = torch.softmax(processed_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
        else:
            next_tokens = torch.argmax(logits, dim=-1, keepdim=True)

        # Update the kv cache with the new keys and values.
        current_cache = outputs.past_key_values

        new_tokens_full = torch.full((initial_batch_size, 1), eos_token_id,
                                     dtype=next_tokens.dtype, device=device)
        new_tokens_full[cur_to_orig] = next_tokens

        # Append the token to each sequence.
        generated = torch.cat([generated, new_tokens_full], dim=1)
        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.size(0), 1), device=device)], dim=1)


        # Update finished flags for the active sequences.
        if isinstance(generation_config.eos_token_id, list):
            finished[cur_to_orig] |= torch.isin(next_tokens.squeeze(1), eos_token_ids)
        else:
            finished[cur_to_orig] |= (next_tokens.squeeze(1) == eos_token_id)

        # If all sequences are finished, break.
        if finished.all():
            break

        # Determine which sequences in the current batch (cache) are still active.
        current_finished = finished[cur_to_orig]
        active_local = ~current_finished
        if active_local.sum().item() < cur_to_orig.shape[0]:
            active_indices_local = torch.nonzero(active_local, as_tuple=False).squeeze(-1)
            # Update the kv cache using indices relative to the current cache.
            print("active batch index", active_indices_local, "kvlen:", attention_mask.shape[1])
            current_cache.batch_select_indices(active_indices_local)
        
            if attention_mask is not None:
                attention_mask = attention_mask[active_indices_local]

            cur_to_orig = cur_to_orig[active_indices_local]

        # Prepare the next input tokens using the updated mapping.
        cur_input = generated[cur_to_orig, -1:].clone()

    # Pad finished sequences back to original batch size if needed
    return generated