# Dataset

[**Medical Abstracts TC Corpus**](https://github.com/sebischair/Medical-Abstracts-TC-Corpus)
(Schopf, Braun & Matthes, NLPIR '22): 14,438 real medical abstracts published by the
[sebis chair at TU München](https://wwwmatthes.in.tum.de) under CC BY-SA 3.0, downloadable as plain
CSV with no account required. `triage_api.dataset` downloads both official splits into `data/raw/`
(cached, so a second run is offline), maps them, and writes `data/triage.csv`.

## Reproducible ingestion

The download URL is **pinned to commit `70a2d91`**, not `main`. A `raw.githubusercontent.com` URL at
a commit SHA is content-addressed, so the pin makes ingestion reproducible and tamper-evident in one
move: the recorded accuracy and latency numbers cannot silently drift because an upstream push
changed the data. Override with `TRIAGE_CORPUS_BASE_URL` to test against a newer revision.

Both official splits are concatenated and re-split 80/20 (stratified, seed 42) rather than using the
corpus's own train/test division: the published split is calibrated for the 5-class condition task,
and ours is a 3-class urgency task, so published baselines would not be comparable either way.

## Urgency mapping

The corpus labels **condition category**, not urgency, so the triage target is derived from it with
an explicit, documented mapping:

| Corpus class | Urgency | Rationale |
| --- | --- | --- |
| Cardiovascular diseases | `urgent` | Acute coronary syndromes, infarction, aortic events; treatment windows measured in minutes |
| Neoplasms | `attention` | Require prompt oncologic workup and staging, but not same-hour intervention |
| Nervous system diseases | `attention` | Prompt specialist referral; the corpus mixes acute and chronic presentations |
| Digestive system diseases | `normal` | Predominantly chronic or electively managed in this corpus |
| General pathological conditions | `normal` | Catch-all class with no organ-specific acute pathway |

The mapping lives in `URGENCY_BY_CONDITION` in `src/triage_api/dataset.py`. It is a clinical
simplification and the honest limitation of this project: the *text* is real, the *urgency label*
is a proxy we defined. A production system would need clinician-adjudicated urgency labels. The
resulting distribution is `normal` 6,299 / `attention` 5,088 / `urgent` 3,051, and the imbalance is
handled with `class_weight="balanced"`.

## What is committed

Raw corpus files and the built CSV stay out of git; `data/sample_triage.csv` holds a committed,
stratified 150-row sample used by the tests, the latency scripts and the load generator, so a clean
checkout can run everything except training without network access.

## Citation

Schopf, Braun & Matthes, *Evaluating Unsupervised Text Classification: Zero-shot and
Similarity-based Approaches*, NLPIR '22 ([doi](https://doi.org/10.1145/3582768.3582795)).
