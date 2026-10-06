# Base-Conditioned Modifier Head: Concat-Gated-Refine

## Overview

CoBPE represents each surface token as a base token together with a tuple of
categorical modifiers. If the modifier groups are indexed by
\(g \in \{1,\ldots,G\}\), a token is represented as

\[
(b, m_1, \ldots, m_G),
\]

where \(b\) is the base-token identity and each \(m_g\) selects one value from
modifier group \(g\). Examples of modifier groups include whitespace behavior,
capitalization, determiners, prepositions, and punctuation.

The model factorizes the next-token distribution as

\[
p(b,m_1,\ldots,m_G \mid c)
= p(b \mid c)\prod_{g=1}^{G}p(m_g \mid b,c),
\]

where \(c\) denotes the preceding context. The standard language-model head
predicts the base token. A separate base-conditioned head predicts the
modifier tuple. Conditioning the modifiers on the base is important because
the plausible realization of a token depends strongly on its lexical identity;
for example, punctuation, capitalization, and function-word attachment are not
independent of the selected base.

## Inputs to the modifier head

At position \(t\), let

- \(h_t \in \mathbb{R}^{d}\) be the final transformer representation used by
  the modifier head;
- \(b_t\) be the selected base token; and
- \(u_{b_t} \in \mathbb{R}^{d}\) be the corresponding row of the base-token
  unembedding matrix.

Thus, the conditioning representation of the base is taken from the same
output-space geometry used to score base tokens. This ties modifier prediction
to the model's learned representation of base-token identity without requiring
a separate lexical encoder.

During training, the modifier head is conditioned on the target base token.
During autoregressive generation, it is conditioned on the base token selected
by the model.

## Concat-gated prediction

Let \(K_g\) be the number of values in modifier group \(g\), and define the
total modifier-logit dimension

\[
M = \sum_{g=1}^{G} K_g.
\]

The hidden-state and base-token branches are normalized and independently
projected into this shared \(M\)-dimensional logit space:

\[
z_h = W_h\,\operatorname{RMSNorm}(h_t),
\]

\[
z_b = W_b\,\operatorname{RMSNorm}(u_{b_t}).
\]

A hidden-state-conditioned gate controls how strongly the selected base token
affects modifier prediction. In the default per-group mode, the gate has one
value for each modifier group:

\[
\alpha_{t,g} = \sigma(w_{\alpha,g}^\top h_t),
\qquad g \in \{1,\ldots,G\}.
\]

The base contribution is gated independently inside each group block:

\[
z_t^{(g)} = z_h^{(g)} + \alpha_{t,g} z_b^{(g)}.
\]

The group blocks are concatenated before refinement:

\[
z_t = [z_t^{(1)};\ldots;z_t^{(G)}].
\]

The scalar-gate option shares one gate across groups. The hidden branch captures
modifier evidence available from context, while the base branch supplies
lexical evidence. This is equivalent to a gated projection of the concatenated
pair \([h_t;u_{b_t}]\), with the gate deciding how much the base representation
contributes to each modifier group.

Within a group, its gate is shared across that group's logits. Each gate scales
the lexical evidence for one group, while the base projection \(W_b\) specifies
which modifier values the base supports or suppresses.

## Residual logit refinement

The concat-gated logits are passed through a residual refinement over the joint
modifier-logit space:

\[
\tilde z_t = z_t + W_r\,\operatorname{SiLU}(z_t),
\]

where \(W_r \in \mathbb{R}^{M \times M}\). The residual form preserves the
direct context-plus-base prediction while allowing a learned nonlinear
correction.

Importantly, refinement occurs before the vector is divided into modifier
groups. The refinement matrix can therefore communicate across groups: evidence
for capitalization can alter punctuation logits, for example, even though the
final group distributions are normalized separately. This supplies a limited
form of dependency modeling between modifier groups without replacing the
factorized output distribution with an exponentially large classifier over all
modifier combinations.

## Groupwise output distributions

The refined vector is partitioned into contiguous group-specific blocks,

\[
\tilde z_t =
[\tilde z_t^{(1)};\ldots;\tilde z_t^{(G)}],
\qquad
\tilde z_t^{(g)} \in \mathbb{R}^{K_g},
\]

and each block defines its own categorical distribution:

\[
p(m_g \mid b_t,c_t)
= \operatorname{softmax}(\tilde z_t^{(g)})_{m_g}.
\]

The modifier training objective is the sum of the groupwise negative
log-likelihoods:

\[
\mathcal{L}_{\mathrm{mod}}
= -\sum_{g=1}^{G}\log p(m_g^* \mid b_t^*,c_t).
\]

Together with the base-token objective, the per-position loss is

\[
\mathcal{L}_t
= -\log p(b_t^* \mid c_t)
  -\lambda_{\mathrm{mod}}
   \sum_{g=1}^{G}\log p(m_{t,g}^* \mid b_t^*,c_t),
\]

with modifier-loss weight \(\lambda_{\mathrm{mod}}\) (set to one in the main
configuration).

## Initialization

The refinement branch is initialized as a neutral residual, so the head begins
as the simpler concat-gated predictor. The gate begins at the midpoint of its
sigmoid range, allowing both the contextual and lexical branches to receive
learning signal from the start. This makes the additional refinement capacity
available gradually rather than perturbing the initial modifier distribution.

## Architectural interpretation

The supported head modes are:

- `lexical-bias` (checkpoint mode `base_bias`): add a learned base-token vector to a
  context-only modifier prediction.
- `gated-concat` (checkpoint mode `concat_gated`): gate a projection of the base-token
  unembedding into context-based modifier logits.
- `gated-refinement` (checkpoint mode `concat_gated_refine`, default): apply
  the joint residual refinement after the per-group gated projections.

Concat-gated-refine combines three complementary inductive biases:

1. **Contextual prediction.** The transformer state predicts modifiers from the
   preceding sequence.
2. **Lexical conditioning.** The selected base token directly influences which
   surface-form transformations are plausible.
3. **Cross-group refinement.** A residual nonlinear map captures interactions
   among modifier groups before groupwise normalization.

The resulting head remains substantially smaller and more structured than a
flat classifier over every valid base-modifier combination. At the same time,
it is more expressive than independent modifier classifiers that see only the
transformer state or incorporate the base token as a fixed additive bias.

## Method summary

> We predict CoBPE modifiers with a base-conditioned concat-gated-refine head.
> Given the final transformer state \(h_t\) and the unembedding vector
> \(u_{b_t}\) of the selected base token, the head projects both representations
> to modifier logits. For each modifier group \(g\), a hidden-state-conditioned
> gate \(\alpha_{t,g}=\sigma(w_{\alpha,g}^\top h_t)\) scales the base-token
> projection before it is added to the context projection. The resulting joint
> logit vector is passed through a residual refinement
> \(\tilde z_t=z_t+W_r\operatorname{SiLU}(z_t)\) over the joint modifier-logit
> space. Finally, \(\tilde z_t\) is partitioned by modifier group and normalized
> with a separate softmax for each group. This design conditions surface-form
> prediction on both context and lexical identity, while the joint refinement
> permits interactions among modifier groups without enumerating their full
> Cartesian product.
