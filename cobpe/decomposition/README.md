# CoBPE metadata construction

This package contains the Python reference builder used by
`scripts.export_compositional_metadata` to create `compositional.json` for the
CoBPE tokenizer runtime.

The published path decomposes tokenizer vocabulary items with surface
modifiers: punctuation, articles and determiners, prepositions, capitalization,
and whitespace prefixes. Tokenizer training lives in `scripts.tok_train.py`;
the runtime encoder lives in `nanochat.compositional`.

`bundle.py` coordinates metadata construction.
`surface.py` and its supporting modules define modifier labels and
vocabulary transformations. `dual_stream.py` contains the Python
reference tokenizer used while building metadata.
