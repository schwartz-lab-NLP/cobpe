# Third-party notices

CoBPE contributions are released under the root Apache-2.0 license.
Vendored and derived code retains the applicable upstream notices below.

## nanochat

The `nanochat/` training backend contains code derived from Andrej Karpathy's
`nanochat` project: <https://github.com/karpathy/nanochat>. The checked-in
upstream source identifies the baseline dataloader as commit `3c3a3d7`; CoBPE
modifies that code for compositional tokenization and the experiments in this
repository.

The applicable MIT license and its copyright notice are preserved in
[MIT license](nanochat/LICENSE):

> Copyright (c) 2025 Andrej Karpathy

## Cut Cross Entropy

The `cut_cross_entropy/` implementation is derived from Apple's Cut Cross
Entropy project. Files retain the Apple 2024 copyright headers. The associated
Apple license and acknowledgements are included in that directory. They were
retrieved from the official repository at commit
`3de376c106a1916bc5e1b619f9c77c87a461ee1c`:

- `cut_cross_entropy/LICENSE` SHA-256:
  `4862c77f9bf843ff8fca191ae1d124a94849105dba5891bc33aa934440732600`
- `cut_cross_entropy/ACKNOWLEDGEMENTS.md` SHA-256:
  `2fececc3e8819466cc24ec67e6b31d78e9be708671b5bb38ac18006f0cee929c`
- Source: <https://github.com/apple/ml-cross-entropy/tree/3de376c106a1916bc5e1b619f9c77c87a461ee1c>

The vendored CCE files have local changes; their copyright headers are
preserved. Other third-party dependencies retain their own notices in their
package metadata and distributions.

## CoBPE decomposition helpers

The decomposition helper code retains an MIT notice credited to Keller Jordan
(2024). Its [LICENSE](cobpe/decomposition/LICENSE) file is preserved with the
helper package.
