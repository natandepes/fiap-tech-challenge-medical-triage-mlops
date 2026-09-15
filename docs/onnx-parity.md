# ONNX parity notes

Converting a TF-IDF pipeline to ONNX is not lossless, and on real clinical text the gap is big
enough to matter. Two problems showed up only once the synthetic corpus was replaced with real
abstracts, and both shaped the vectorizer's final configuration.

## 1. `skl2onnx` drops bigrams that span a discarded token

scikit-learn's default token pattern `(?u)\b\w\w+\b` throws away one-character tokens *before*
forming n-grams, so `"crosslinked D-dimer"` yields the bigram `crosslinked dimer` and
`"p less than 0.001"` yields `than 001`. ONNX Runtime's RE2-based tokenizer does not reproduce
those, so the exported model silently saw a different feature vector. Also, RE2 rejects the `(?u)`
inline flag outright.

Passing the explicit pattern `\w\w+` cut label disagreement from 6/600 to 1/600 with an identical
vocabulary and identical accuracy.

## 2. Bigrams made ONNX *slower* than scikit-learn

With `ngram_range=(1, 2)` the vocabulary is 233k features, the ONNX artifact is 11.2 MB and
inference measured **0.6x** — a slowdown. Unigrams cut the artifact to 954 KB, run 2.50x faster,
*and* score better (0.635 vs. 0.612 macro-F1), since bigrams over long abstracts mostly add sparse
noise.

## What remains

Float32-vs-float64 rounding: the ONNX and scikit-learn labels still disagree on ~0.6% of inputs,
and in every observed case the top two classes were within 0.018 of each other — genuine coin-flips
rather than substantive disagreement.

`tests/test_onnx.py` encodes exactly that invariant instead of a blanket equality assertion:
probabilities must track within `5e-2`, and a label may differ *only* when the top-two gap is under
`0.05`. It runs on 60 real sample abstracts.

## Takeaway

The optimization is real, but it constrained the model. The vectorizer's configuration is driven by
what converts faithfully, not only by what scores best — and here, fortunately, the two agreed.
