"""
BPE Tokenizer in the style of GPT-4.

Two implementations are available:
1) HuggingFace Tokenizer that can do both training and inference but is really confusing
2) Our own RustBPE Tokenizer for training and tiktoken for efficient inference
"""

import os
import json
from functools import lru_cache

from nanochat.compositional import CompositionalSpec, CompositionalTokenizer
from cobpe.tokenization.encoding import EncodedSequenceMixin
from cobpe.tokenization.fingerprints import fingerprint_named_files

SPECIAL_TOKENS = [
    # every document begins with the Beginning of Sequence (BOS) token that delimits documents
    "<|bos|>",
    # tokens below are only used during finetuning to render Conversations into token ids
    "<|user_start|>", # user messages
    "<|user_end|>",
    "<|assistant_start|>", # assistant messages
    "<|assistant_end|>",
    "<|python_start|>", # assistant invokes python REPL tool
    "<|python_end|>",
    "<|output_start|>", # python REPL outputs back to assistant
    "<|output_end|>",
]

# Apostrophe-like characters we keep inside word chunks for contractions/clitics.
_APOSTROPHE_CLASS = r"['`\u2018\u2019\u02BC\u05F3]"

# NOTE: this split pattern deviates from GPT-4 in that we use \p{N}{1,2} instead of \p{N}{1,3}
# I did this because I didn't want to "waste" too many tokens on numbers for smaller vocab sizes.
# I verified that 2 is the sweet spot for vocab size of 32K. 1 is a bit worse, 3 was worse still.
#
# Additional policy for baseline:
# - keep contractions in one chunk (e.g., don't / don’t / don`t)
# - allow optional leading "-" for suffix-like forms (e.g., -less)
# - do not allow other boundary punctuation to fuse with letter chunks
SPLIT_PATTERN = (
    rf""" ?-?\p{{L}}+(?:{_APOSTROPHE_CLASS}\p{{L}}+)*|\p{{N}}{{1,2}}| ?[^\s\p{{L}}\p{{N}}]++[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
)

# GPT-4-like variant for normalized tokenizer experiments:
# - isolate whitespace into dedicated whitespace-only chunks (\s+)
# - keep punctuation detached and merged only with punctuation chars
# - keep in-word apostrophes/geresh together with surrounding letters
#   (e.g., don't / don’t / don`t / אב׳גד stay one pretokenization chunk)
# - keep "-less" style punctuation detached ("-", "less") so punctuation can be modifierized.
WHITESPACE_ISOLATING_SPLIT_PATTERN = (
    rf"""\p{{L}}+(?:{_APOSTROPHE_CLASS}\p{{L}}+)*|\p{{N}}{{1,2}}|[^\s\p{{L}}\p{{N}}]++|\s+"""
)

# -----------------------------------------------------------------------------
# Generic GPT-4-style tokenizer based on HuggingFace Tokenizer
from tokenizers import Tokenizer as HFTokenizer
from tokenizers import pre_tokenizers, decoders, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

class HuggingFaceTokenizer(EncodedSequenceMixin):
    """Light wrapper around HuggingFace Tokenizer for some utilities"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, hf_path):
        # init from a HuggingFace pretrained tokenizer (e.g. "gpt2")
        tokenizer = HFTokenizer.from_pretrained(hf_path)
        return cls(tokenizer)

    @classmethod
    def from_directory(cls, tokenizer_dir):
        # init from a local directory on disk (e.g. "out/tokenizer")
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        tokenizer = HFTokenizer.from_file(tokenizer_path)
        return cls(tokenizer)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        # train from an iterator of text
        # Configure the HuggingFace Tokenizer
        tokenizer = HFTokenizer(BPE(
            byte_fallback=True, # needed!
            unk_token=None,
            fuse_unk=False,
        ))
        # Normalizer: None
        tokenizer.normalizer = None
        # Pre-tokenizer: GPT-4 style
        # the regex pattern used by GPT-4 to split text into groups before BPE
        # NOTE: The pattern was changed from \p{N}{1,3} to \p{N}{1,2} because I suspect it is harmful to
        # very small models and smaller vocab sizes, because it is a little bit wasteful in the token space.
        # (but I haven't validated this! TODO)
        gpt4_split_regex = Regex(SPLIT_PATTERN) # huggingface demands that you wrap it in Regex!!
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(pattern=gpt4_split_regex, behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
        ])
        # Decoder: ByteLevel (it pairs together with the ByteLevel pre-tokenizer)
        tokenizer.decoder = decoders.ByteLevel()
        # Post-processor: None
        tokenizer.post_processor = None
        # Trainer: BPE
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            show_progress=True,
            min_frequency=0, # no minimum frequency
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=SPECIAL_TOKENS,
        )
        # Kick off the training
        tokenizer.train_from_iterator(text_iterator, trainer)
        return cls(tokenizer)

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def get_special_tokens(self):
        special_tokens_map = self.tokenizer.get_added_tokens_decoder()
        special_tokens = [w.content for w in special_tokens_map.values()]
        return special_tokens

    def id_to_token(self, id):
        return self.tokenizer.id_to_token(id)

    def _encode_one(self, text, prepend=None, append=None, num_threads=None):
        # encode a single string
        # prepend/append can be either a string of a special token or a token id directly.
        # num_threads is ignored (only used by the nanochat Tokenizer for parallel encoding)
        assert isinstance(text, str)
        ids = []
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
            ids.append(prepend_id)
        ids.extend(self.tokenizer.encode(text, add_special_tokens=False).ids)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)
            ids.append(append_id)
        return ids

    def encode_special(self, text):
        # encode a single special token via exact match
        return self.tokenizer.token_to_id(text)

    def get_bos_token_id(self):
        # Different HuggingFace models use different BOS tokens and there is little consistency
        # 1) attempt to find a <|bos|> token
        bos = self.encode_special("<|bos|>")
        # 2) if that fails, attempt to find a <|endoftext|> token (e.g. GPT-2 models)
        if bos is None:
            bos = self.encode_special("<|endoftext|>")
        # 3) if these fail, it's better to crash than to silently return None
        assert bos is not None, "Failed to find BOS token in tokenizer"
        return bos

    def encode(self, text, *args, **kwargs):
        if isinstance(text, str):
            return self._encode_one(text, *args, **kwargs)
        elif isinstance(text, list):
            return [self._encode_one(t, *args, **kwargs) for t in text]
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def save(self, tokenizer_dir):
        # save the tokenizer to disk
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        self.tokenizer.save(tokenizer_path)
        print(f"Saved tokenizer to {tokenizer_path}")

# -----------------------------------------------------------------------------
# Tokenizer based on rustbpe + tiktoken combo
import pickle
import rustbpe
import tiktoken

class RustBPETokenizer(EncodedSequenceMixin):
    """Light wrapper around tiktoken (for efficient inference) but train with rustbpe"""

    def __init__(self, enc, bos_token):
        self.enc = enc
        self.bos_token_id = self.encode_special(bos_token)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size, pattern=None):
        # 1) train using rustbpe
        tokenizer = rustbpe.Tokenizer()
        # the special tokens are inserted later in __init__, we don't train them here
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        assert vocab_size_no_special >= 256, f"vocab_size_no_special must be at least 256, got {vocab_size_no_special}"
        if pattern is None:
            pattern = SPLIT_PATTERN
        tokenizer.train_from_iterator(text_iterator, vocab_size_no_special, pattern=pattern)
        # 2) construct the associated tiktoken encoding for inference
        pattern = tokenizer.get_pattern()
        mergeable_ranks_list = tokenizer.get_mergeable_ranks()
        mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
        tokens_offset = len(mergeable_ranks)
        special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
        enc = tiktoken.Encoding(
            name="rustbpe",
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks, # dict[bytes, int] (token bytes -> merge priority rank)
            special_tokens=special_tokens, # dict[str, int] (special token name -> token id)
        )
        return cls(enc, "<|bos|>")

    @classmethod
    def from_directory(cls, tokenizer_dir):
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "rb") as f:
            enc = pickle.load(f)
        return cls(enc, "<|bos|>")

    @classmethod
    def from_pretrained(cls, tiktoken_name):
        # https://github.com/openai/tiktoken/blob/eedc8563/tiktoken_ext/openai_public.py
        enc = tiktoken.get_encoding(tiktoken_name)
        # tiktoken calls the special document delimiter token "<|endoftext|>"
        # yes this is confusing because this token is almost always PREPENDED to the beginning of the document
        # it most often is used to signal the start of a new sequence to the LLM during inference etc.
        # so in nanoChat we always use "<|bos|>" short for "beginning of sequence", but historically it is often called "<|endoftext|>".
        return cls(enc, "<|endoftext|>")

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_special_tokens(self):
        return self.enc.special_tokens_set

    def id_to_token(self, id):
        return self.enc.decode([id])

    @lru_cache(maxsize=32)
    def encode_special(self, text):
        return self.enc.encode_single_token(text)

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, append=None, num_threads=8):
        # text can be either a string or a list of strings

        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)

        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id) # TODO: slightly inefficient here? :( hmm
            if append is not None:
                ids.append(append_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for ids_row in ids:
                    ids_row.insert(0, prepend_id) # TODO: same
            if append is not None:
                for ids_row in ids:
                    ids_row.append(append_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

        return ids

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        return self.enc.decode(ids)

    def save(self, tokenizer_dir):
        # save the encoding object to disk
        os.makedirs(tokenizer_dir, exist_ok=True)
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(self.enc, f)
        print(f"Saved tokenizer encoding to {pickle_path}")

# -----------------------------------------------------------------------------
# nanochat-specific convenience functions

def _resolve_tokenizer_dir():
    """
    Resolve tokenizer directory with explicit overrides before defaulting to get_base_dir().

    Priority:
    1) NANOCHAT_TOKENIZER_DIR (explicit path to tokenizer dir)
    2) DATA_BASE_DIR/tokenizer (used by slurm scripts that separate data/output roots)
    3) NANOCHAT_DATA_DIR/tokenizer (legacy compatibility)
    4) get_base_dir()/tokenizer
    """
    explicit_dir = os.environ.get("NANOCHAT_TOKENIZER_DIR", "").strip()
    if explicit_dir:
        return os.path.abspath(os.path.expanduser(explicit_dir))

    for env_name in ("DATA_BASE_DIR", "NANOCHAT_DATA_DIR"):
        root = os.environ.get(env_name, "").strip()
        if not root:
            continue
        candidate = os.path.join(os.path.abspath(os.path.expanduser(root)), "tokenizer")
        if os.path.isdir(candidate):
            return candidate

    from nanochat.common import get_base_dir
    base_dir = get_base_dir()
    return os.path.join(base_dir, "tokenizer")


def get_tokenizer_fingerprint(tokenizer_dir=None):
    """Return a stable content fingerprint for model/tokenizer compatibility checks."""

    tokenizer_dir = os.path.abspath(
        os.path.expanduser(tokenizer_dir or _resolve_tokenizer_dir())
    )
    return fingerprint_named_files(
        tokenizer_dir,
        ("tokenizer.pkl", "compositional.json"),
        required=("tokenizer.pkl",),
    )


def get_tokenizer():
    tokenizer_dir = _resolve_tokenizer_dir()
    tokenizer = RustBPETokenizer.from_directory(tokenizer_dir)
    metadata_path = os.path.join(tokenizer_dir, "compositional.json")
    if os.path.exists(metadata_path):
        buffer_config_path = os.path.join(tokenizer_dir, "tokenizer_buffer.json")
        if not os.path.exists(buffer_config_path):
            buffer_config_path = os.path.join(tokenizer_dir, "vd_buffer_config.json")
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        if os.path.exists(buffer_config_path):
            with open(buffer_config_path, "r", encoding="utf-8") as f:
                buffer_config = json.load(f)
            target_vocab_size = int(buffer_config.get("vocab_size_target", -1))
            train_vocab_size = int(buffer_config.get("vocab_size_train", -1))
            if train_vocab_size > target_vocab_size > 0:
                finalized = bool(buffer_config.get("finalized_at_metadata_export", False))
                if not finalized or tokenizer.get_vocab_size() != target_vocab_size:
                    raise RuntimeError(
                        "Refusing to load an unfinalized buffered CoBPE tokenizer: "
                        f"current={tokenizer.get_vocab_size()} target={target_vocab_size} "
                        f"train={train_vocab_size} finalized={finalized}. "
                        "Re-run scripts.export_compositional_metadata to finalize it."
                    )
        runtime_vocab_size = int(metadata.get("runtime_vocab_size", tokenizer.get_vocab_size()))
        if runtime_vocab_size != tokenizer.get_vocab_size():
            raise RuntimeError(
                "CoBPE metadata/runtime vocabulary mismatch: "
                f"metadata={runtime_vocab_size} tokenizer={tokenizer.get_vocab_size()}"
            )
        spec = CompositionalSpec.from_path(metadata_path)
        referenced_ids = [
            int(token_id)
            for entry in spec.sequence_entries
            for token_id in (*entry.token_ids, *entry.base_ids)
        ]
        referenced_ids.extend(
            int(entry["base_id"])
            for entry in metadata.get("inverse_entries", [])
            if "base_id" in entry
        )
        if referenced_ids and (min(referenced_ids) < 0 or max(referenced_ids) >= runtime_vocab_size):
            raise RuntimeError(
                "CoBPE metadata references IDs outside the runtime vocabulary: "
                f"min={min(referenced_ids)} max={max(referenced_ids)} vocab={runtime_vocab_size}"
            )
        return CompositionalTokenizer(tokenizer, spec, tokenizer_dir=tokenizer_dir)
    # return HuggingFaceTokenizer.from_directory(tokenizer_dir)
    return tokenizer

def get_token_bytes(device="cpu"):
    import torch
    tokenizer_dir = _resolve_tokenizer_dir()
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    assert os.path.exists(token_bytes_path), f"Token bytes not found at {token_bytes_path}? It gets written by tok_train.py"
    with open(token_bytes_path, "rb") as f:
        token_bytes = torch.load(f, map_location=device)
    tokenizer = RustBPETokenizer.from_directory(tokenizer_dir)
    if int(token_bytes.numel()) != int(tokenizer.get_vocab_size()):
        raise RuntimeError(
            "token_bytes/tokenizer vocabulary mismatch: "
            f"token_bytes={token_bytes.numel()} tokenizer={tokenizer.get_vocab_size()}"
        )
    return token_bytes
