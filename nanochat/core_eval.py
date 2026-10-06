"""
Functions for evaluating the CORE metric, as described in the DCLM paper.
https://arxiv.org/abs/2406.11794
"""
import random
import re
import string

from jinja2 import Template
import torch
import torch.distributed as dist

from cobpe.tokenization.encoding import EncodedSequence, stack_sequences

# -----------------------------------------------------------------------------
# Prompt rendering utilities

def render_prompts_mc(item, continuation_delimiter, fewshot_examples=None):
    """Render complete prompts for a multiple choice question"""
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.query }}{{ continuation_delimiter }}{{ example.choices[example.gold] }}

{% endfor -%}
{{ item.query }}{{ continuation_delimiter }}{{ choice }}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    prompts = [template.render(choice=choice, **context) for choice in item['choices']]
    return prompts


def render_prompts_schema(item, continuation_delimiter, fewshot_examples=None):
    """Render complete prompts for a schema question"""
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context_options[example.gold] }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ context }}{{ continuation_delimiter }}{{ item.continuation }}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    prompts = [template.render(context=context_option, **context)
               for context_option in item['context_options']]
    return prompts


def render_prompts_lm(item, continuation_delimiter, fewshot_examples=None):
    """
    Render complete prompt for a language modeling task.
    Notice that we manually trim the context in the template,
    which in some datasets seems to have trailing whitespace (which we don't want).
    """
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context | trim }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ item.context | trim }}{{ continuation_delimiter }}{% if include_continuation %}{{ item.continuation }}{% endif %}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    # Return two prompts: without and with the continuation
    prompt_without = template.render(include_continuation=False, **context)
    prompt_with = template.render(include_continuation=True, **context)
    # Due to the way the data seems to be stored, I think I need to strip in the case of LM here.
    # Otherwise we may get trailing whitespaces in prompt_without (which get absorbed into the next
    # token in prompt_with), meaning we don't get a nice and clean prefix in the token space
    # to detect the final continuation. Tokenizers...
    prompt_without = prompt_without.strip()
    return [prompt_without, prompt_with]


def find_common_length(token_sequences, direction='left'):
    """
    Find the length of the common prefix or suffix across token sequences
    - direction: 'left' for prefix, 'right' for suffix
    """
    min_len = min(len(seq) for seq in token_sequences)
    indices = {
        'left': range(min_len),
        'right': range(-1, -min_len-1, -1)
    }[direction]
    # Find the first position where the token sequences differ
    for i, idx in enumerate(indices):
        token = token_sequences[0][idx]
        if not all(seq[idx] == token for seq in token_sequences):
            return i
    return min_len


def _tokenizer_has_modifiers(tokenizer):
    return bool(
        hasattr(tokenizer, "has_compositional_mode")
        and tokenizer.has_compositional_mode()
    )


def _modifier_predictions_match_with_suffix_boundary_rule(
    predicted_modifiers, actual_modifiers, tokenizer
):
    """Compare LM modifier predictions, ignoring an ungenerated default suffix."""
    matches = predicted_modifiers == actual_modifiers
    spec = getattr(tokenizer, "spec", None)
    group_to_idx = getattr(spec, "group_to_idx", {})
    if (
        predicted_modifiers.numel() > 0
        and "suffix_punctuation" in group_to_idx
        and hasattr(tokenizer, "get_default_modifier")
    ):
        suffix_group_idx = int(group_to_idx["suffix_punctuation"])
        default_suffix = int(tokenizer.get_default_modifier()[suffix_group_idx])
        target_suffix = int(actual_modifiers[-1, suffix_group_idx].item())
        if target_suffix == default_suffix:
            matches[-1, suffix_group_idx] = True
    return bool(torch.all(matches).item())


def _encode_prompts_as_sequences(tokenizer, prompts):
    bos = resolve_bos_token_id(tokenizer)
    if _tokenizer_has_modifiers(tokenizer):
        if hasattr(tokenizer, "encode_sequences"):
            return tokenizer.encode_sequences(prompts, prepend=bos)
        return [
            EncodedSequence(token_ids, modifier_rows)
            for token_ids, modifier_rows in tokenizer.encode_with_modifiers(prompts, prepend=bos)
        ]
    return [EncodedSequence(token_ids) for token_ids in encode_prompts_with_bos(tokenizer, prompts)]


def _sequence_units(sequence: EncodedSequence):
    return sequence.units()


def resolve_bos_token_id(tokenizer):
    """Best-effort BOS resolution across nanochat and HF tokenizer APIs."""
    for obj in (tokenizer, getattr(tokenizer, "tokenizer", None)):
        if obj is None:
            continue
        get_bos = getattr(obj, "get_bos_token_id", None)
        if callable(get_bos):
            bos = get_bos()
            if bos is not None:
                return int(bos)
        bos = getattr(obj, "bos_token_id", None)
        if bos is not None:
            return int(bos)
        eos = getattr(obj, "eos_token_id", None)
        if eos is not None:
            return int(eos)
    return None


def _normalize_token_batch(token_batch):
    # HF BatchEncoding / mapping-like objects
    if hasattr(token_batch, "input_ids"):
        token_batch = token_batch.input_ids
    elif isinstance(token_batch, dict):
        token_batch = token_batch.get("input_ids")
    # tokenizers.Encoding (single)
    elif hasattr(token_batch, "ids"):
        token_batch = token_batch.ids

    # tokenizers.Encoding list
    if isinstance(token_batch, (list, tuple)) and len(token_batch) > 0 and hasattr(token_batch[0], "ids"):
        token_batch = [enc.ids for enc in token_batch]

    if torch.is_tensor(token_batch):
        token_batch = token_batch.tolist()
    if isinstance(token_batch, tuple):
        token_batch = list(token_batch)
    if not isinstance(token_batch, list):
        raise TypeError(f"Unsupported token batch type: {type(token_batch)}")
    if not token_batch:
        return []
    if isinstance(token_batch[0], int):
        return [list(map(int, token_batch))]
    return [list(map(int, row)) for row in token_batch]


def encode_prompts_with_bos(tokenizer, prompts):
    """Encode list[str] prompts and prefix BOS when available."""
    bos = resolve_bos_token_id(tokenizer)
    # nanochat tokenizer wrapper path
    if hasattr(tokenizer, "__call__") and hasattr(tokenizer, "get_bos_token_id"):
        try:
            return _normalize_token_batch(tokenizer(prompts, prepend=bos))
        except TypeError:
            pass
    # HF tokenizer path
    try:
        encoded = tokenizer(prompts, add_special_tokens=False)
    except TypeError:
        encoded = tokenizer(prompts)
    tokens = _normalize_token_batch(encoded)
    if bos is not None:
        tokens = [[bos, *row] for row in tokens]
    return tokens


def batch_sequences_mc(tokenizer, prompts):
    # In multiple choice, contexts are the same but the continuation is different (common prefix)
    sequences = _encode_prompts_as_sequences(tokenizer, prompts)
    # figure out the start and end of each continuation
    answer_start_idx = find_common_length(
        [_sequence_units(sequence) for sequence in sequences],
        direction='left',
    )
    start_indices = [answer_start_idx] * len(prompts)
    end_indices = [len(sequence) for sequence in sequences]
    return sequences, start_indices, end_indices


def batch_sequences_schema(tokenizer, prompts):
    # In schema tasks, contexts vary but continuation is the same (common suffix)
    sequences = _encode_prompts_as_sequences(tokenizer, prompts)
    # figure out the start and end of each context
    suffix_length = find_common_length(
        [_sequence_units(sequence) for sequence in sequences],
        direction='right',
    )
    end_indices = [len(sequence) for sequence in sequences]
    start_indices = [ei - suffix_length for ei in end_indices]
    return sequences, start_indices, end_indices


def batch_sequences_lm(tokenizer, prompts):
    # In LM tasks, we have two prompts: without and with continuation
    sequences = _encode_prompts_as_sequences(tokenizer, prompts)
    seq_without, seq_with = sequences
    units_without = _sequence_units(seq_without)
    units_with = _sequence_units(seq_with)
    # Exact LM span: first/last position where token ids differ.
    min_len = min(len(seq_without), len(seq_with))
    prefix_len = 0
    while prefix_len < min_len and units_without[prefix_len] == units_with[prefix_len]:
        prefix_len += 1

    max_suffix = min_len - prefix_len
    suffix_len = 0
    while suffix_len < max_suffix:
        if units_without[len(seq_without) - 1 - suffix_len] != units_with[len(seq_with) - 1 - suffix_len]:
            break
        suffix_len += 1

    start_idx = prefix_len
    end_idx = len(seq_with) - suffix_len
    if start_idx >= end_idx:
        def _fmt_prompt(p: str) -> str:
            flat = p.replace("\n", "\\n")
            head = flat[:180]
            tail = flat[-180:] if len(flat) > 180 else flat
            return f"chars={len(p)} head='{head}' tail='{tail}'"
        raise ValueError(
            "LM prompt tokenization produced no changed span between without/with prompts. "
            f"len(tokens_without)={len(seq_without)} len(tokens_with)={len(seq_with)} "
            f"prefix_len={prefix_len} suffix_len={suffix_len}. "
            f"prompt_without[{_fmt_prompt(prompts[0])}] prompt_with[{_fmt_prompt(prompts[1])}]"
        )
    # we only need the with continuation prompt in the LM task, i.e. batch size of 1
    return [seq_with], [start_idx], [end_idx]


def normalize_qa_answer_squad_style(text: str) -> str:
    """Official-style normalization used by SQuAD/CoQA evaluation scripts."""
    if text is None:
        return ""
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())
    return text


def get_task_text_normalization_policy(task_label: str) -> str:
    """
    Return normalization policy name for text exact-match benchmarks.
    Policies:
      - "none": raw exact string match
      - "squad_qa": lowercase + remove punctuation/articles + whitespace fix
    """
    label = str(task_label or "").strip().lower()
    if label in {"squad", "coqa"}:
        return "squad_qa"
    return "none"


def normalize_text_for_policy(text: str, policy: str) -> str:
    if policy == "squad_qa":
        return normalize_qa_answer_squad_style(text)
    return str(text if text is not None else "")


@torch.no_grad()
def forward_model(model, input_ids, modifier_ids=None):
    """
    Take BxT tensor of token ids, return BxT tensor of losses and argmax predictions.
    The last column of losses is set to nan because we don't have autoregressive targets there.
    """
    batch_size, seq_len = input_ids.size()
    if modifier_ids is None:
        outputs = model(input_ids)
        prediction_modifier_ids = None
    else:
        outputs, hidden = model(input_ids, modifier_ids=modifier_ids, return_hidden=True)
        prediction_base_ids = outputs.argmax(dim=-1)
        pred_modifier_logits = model.get_modifier_logits(hidden, prediction_base_ids)
        prediction_modifier_ids = torch.stack(
            [group_logits.argmax(dim=-1) for group_logits in pred_modifier_logits],
            dim=-1,
        )
    # Roll the tensor to the left by one position to get the (autoregressive) target ids
    target_ids = torch.roll(input_ids, shifts=-1, dims=1)
    # Calculate cross entropy at all positions
    losses = torch.nn.functional.cross_entropy(
        outputs.view(batch_size * seq_len, -1),
        target_ids.view(batch_size * seq_len),
        reduction='none'
    ).view(batch_size, seq_len)
    if modifier_ids is not None:
        # Score the probability of the actual surface token. Modifier groups are
        # independent factors conditioned on the target base token, so their
        # teacher-forced NLLs add to the base-token NLL.
        target_modifier_ids = torch.roll(modifier_ids, shifts=-1, dims=1)
        target_modifier_logits = model.get_modifier_logits(hidden, target_ids)
        for group_idx, group_logits in enumerate(target_modifier_logits):
            losses = losses + torch.nn.functional.cross_entropy(
                group_logits.reshape(batch_size * seq_len, -1),
                target_modifier_ids[..., group_idx].reshape(batch_size * seq_len),
                reduction="none",
            ).view(batch_size, seq_len)
    # Set the last column to be nan because there is no autoregressive loss there
    losses[:, -1] = float('nan')
    # Get the argmax predictions at each position
    predictions = outputs.argmax(dim=-1)
    return losses, predictions, prediction_modifier_ids


@torch.no_grad()
def evaluate_example(idx, model, tokenizer, data, device, task_meta, return_normalized: bool = False):
    """Evaluate a single example, return True if correct, False otherwise"""
    item = data[idx]
    task_type = task_meta['task_type']
    num_fewshot = task_meta['num_fewshot']
    continuation_delimiter = task_meta['continuation_delimiter']

    # Sample few-shot examples (excluding current item)
    fewshot_examples = []
    if num_fewshot > 0:
        rng = random.Random(1234 + idx)
        available_indices = [i for i in range(len(data)) if i != idx]
        fewshot_indices = rng.sample(available_indices, num_fewshot)
        fewshot_examples = [data[i] for i in fewshot_indices]

    # Render prompts and batch sequences based on task type
    if task_type == 'multiple_choice':
        prompts = render_prompts_mc(item, continuation_delimiter, fewshot_examples)
        sequences, start_idxs, end_idxs = batch_sequences_mc(tokenizer, prompts)
    elif task_type == 'schema':
        prompts = render_prompts_schema(item, continuation_delimiter, fewshot_examples)
        sequences, start_idxs, end_idxs = batch_sequences_schema(tokenizer, prompts)
    elif task_type == 'language_modeling':
        prompts = render_prompts_lm(item, continuation_delimiter, fewshot_examples)
        sequences, start_idxs, end_idxs = batch_sequences_lm(tokenizer, prompts)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    # Some models can't forward sequences beyond a certain length (e.g. GPT-2)
    # In these cases, we have to truncate sequences to max length and adjust the indices
    if hasattr(model, 'max_seq_len') and model.max_seq_len is not None:
        max_tokens = model.max_seq_len
        new_sequences, new_start_idxs, new_end_idxs = [], [], []
        for sequence, s, e in zip(sequences, start_idxs, end_idxs):
            if len(sequence) > max_tokens:
                num_to_crop = len(sequence) - max_tokens
                new_sequences.append(sequence.slice(-max_tokens)) # take the last max_tokens tokens
                new_start_idxs.append(s - num_to_crop) # shift the indices down
                new_end_idxs.append(e - num_to_crop)
                assert s - num_to_crop >= 0, "this should never happen right?"
                assert e - num_to_crop >= 0, "this should never happen right?"
            else:
                new_sequences.append(sequence) # keep unchanged
                new_start_idxs.append(s)
                new_end_idxs.append(e)
        sequences, start_idxs, end_idxs = new_sequences, new_start_idxs, new_end_idxs

    # Stack up all the sequences into a batch
    pad_token_id = resolve_bos_token_id(tokenizer)
    if pad_token_id is None:
        pad_token_id = 0
    has_modifiers = any(sequence.modifiers is not None for sequence in sequences)
    default_modifier = (
        tokenizer.get_default_modifier()
        if has_modifiers and hasattr(tokenizer, "get_default_modifier")
        else None
    )
    stacked = stack_sequences(sequences, pad_token_id, default_modifier)
    input_ids = stacked.ids.to(device)
    modifier_ids = None if stacked.modifiers is None else stacked.modifiers.to(device)

    # Forward the model, get the autoregressive loss and argmax prediction at each token
    losses, predictions, prediction_modifier_ids = forward_model(model, input_ids, modifier_ids)

    # See if the losses/predictions come out correctly
    normalized_correct = None
    if task_type == 'language_modeling':
        # language modeling task is currently always batch size 1
        si = start_idxs[0]
        ei = end_idxs[0]
        # predictions[i] predict input_ids[i+1] autoregressively
        predicted_tokens = predictions[0, si-1:ei-1]
        actual_tokens = input_ids[0, si:ei]
        is_correct = torch.all(predicted_tokens == actual_tokens).item()
        if modifier_ids is not None:
            predicted_mods = prediction_modifier_ids[0, si-1:ei-1]
            actual_mods = modifier_ids[0, si:ei]
            is_correct = bool(
                is_correct
                and _modifier_predictions_match_with_suffix_boundary_rule(
                    predicted_mods, actual_mods, tokenizer
                )
            )
        if return_normalized:
            policy = get_task_text_normalization_policy(task_meta.get("label", ""))
            if policy == "none":
                normalized_correct = is_correct
            else:
                pred_ids = predicted_tokens.detach().cpu().tolist()
                gold_ids = actual_tokens.detach().cpu().tolist()
                if modifier_ids is not None:
                    pred_mods = prediction_modifier_ids[0, si-1:ei-1].detach().cpu().tolist()
                    gold_mods = modifier_ids[0, si:ei].detach().cpu().tolist()
                    pred_text = tokenizer.decode_with_modifiers(pred_ids, pred_mods)
                    gold_text = tokenizer.decode_with_modifiers(gold_ids, gold_mods)
                else:
                    pred_text = tokenizer.decode(pred_ids)
                    gold_text = tokenizer.decode(gold_ids)
                normalized_correct = int(
                    normalize_text_for_policy(pred_text, policy)
                    == normalize_text_for_policy(gold_text, policy)
                )
    elif task_type in ['multiple_choice', 'schema']:
        # For MC/schema: find the option with lowest average loss
        mean_losses = [losses[i, si-1:ei-1].mean().item()
                        for i, (si, ei) in enumerate(zip(start_idxs, end_idxs))]
        pred_idx = mean_losses.index(min(mean_losses))
        is_correct = pred_idx == item['gold']
        if return_normalized:
            normalized_correct = is_correct
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    if return_normalized:
        if normalized_correct is None:
            normalized_correct = is_correct
        return is_correct, bool(normalized_correct)
    return is_correct


def evaluate_task(model, tokenizer, data, device, task_meta, return_normalized: bool = False):
    """
    This function is responsible for evaluating one task across many examples.
    It also handles dispatch to all processes if the script is run with torchrun.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    correct = torch.zeros(len(data), dtype=torch.float32, device=device)
    normalized_correct = torch.zeros(len(data), dtype=torch.float32, device=device) if return_normalized else None
    # stride the examples to each rank
    for idx in range(rank, len(data), world_size):
        if return_normalized:
            is_correct, is_correct_norm = evaluate_example(
                idx, model, tokenizer, data, device, task_meta, return_normalized=True
            )
            correct[idx] = float(is_correct)
            normalized_correct[idx] = float(is_correct_norm)
        else:
            is_correct = evaluate_example(idx, model, tokenizer, data, device, task_meta)
            correct[idx] = float(is_correct)
    # sync results across all the processes if running distributed
    if world_size > 1:
        dist.barrier()
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
        if normalized_correct is not None:
            dist.all_reduce(normalized_correct, op=dist.ReduceOp.SUM)
    # compute the mean
    mean_correct = correct.mean().item()
    if return_normalized:
        mean_correct_norm = normalized_correct.mean().item()
        return mean_correct, mean_correct_norm
    return mean_correct
