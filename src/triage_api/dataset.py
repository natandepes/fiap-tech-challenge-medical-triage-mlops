import random
from pathlib import Path

import pandas as pd

from triage_api import URGENCY_LABELS
from triage_api.config import DATASET_PATH, DATASET_SIZE, RANDOM_SEED

_BODY_SITES = [
    "chest", "abdomen", "left lung", "right lung", "cranium", "lumbar spine",
    "pelvis", "left kidney", "liver", "left knee", "right shoulder", "sinuses",
]

_MODALITIES = [
    "radiograph", "CT scan", "ultrasound", "MRI", "laboratory panel",
    "physical examination", "ECG", "clinical note",
]

_FINDINGS = {
    "normal": [
        "no acute abnormality identified",
        "findings within normal limits",
        "unremarkable study",
        "mild degenerative changes consistent with age",
        "stable appearance compared with the prior exam",
        "no evidence of fracture or dislocation",
        "clear lung fields with no consolidation",
        "no focal lesion detected",
        "results within the reference range",
        "trace physiologic fluid, not clinically significant",
    ],
    "attention": [
        "moderate findings that warrant outpatient follow-up",
        "a small indeterminate nodule, recommend surveillance imaging",
        "mildly elevated inflammatory markers",
        "non-obstructing calculus without hydronephrosis",
        "chronic changes that should be correlated clinically",
        "borderline cardiomegaly, suggest echocardiography",
        "a stable but abnormal lesion, non-urgent referral advised",
        "low-grade findings requiring a scheduled review",
        "mild pleural thickening, follow-up in a few weeks",
        "abnormal but not immediately threatening result",
    ],
    "urgent": [
        "acute intracranial hemorrhage",
        "large pulmonary embolism with right heart strain",
        "findings compatible with acute myocardial infarction",
        "free intraperitoneal air suggesting perforation",
        "tension pneumothorax with mediastinal shift",
        "critical potassium value requiring immediate action",
        "signs of septic shock with rising lactate",
        "acute ischemic stroke in the middle cerebral artery territory",
        "ruptured aortic aneurysm with active extravasation",
        "impending respiratory failure with severe hypoxemia",
    ],
}

_TEMPLATES = [
    "{modality} of the {site}: {finding}.",
    "The {modality} demonstrates {finding}.",
    "Impression: {finding} on {modality} of the {site}.",
    "Reported {finding}. Correlate with {modality} history.",
    "{site} {modality} performed; {finding}.",
    "Clinical summary: {finding}; obtained via {modality}.",
]

_QUALIFIERS = {
    "normal": ["Patient is comfortable.", "No new complaints.", "Routine review.", ""],
    "attention": ["Symptoms are stable.", "Patient reports mild discomfort.", "Non-emergent.", ""],
    "urgent": [
        "Patient is hemodynamically unstable.",
        "Immediate clinical attention required.",
        "Rapid deterioration noted.",
        "",
    ],
}

_SHARED_FILLER = [
    "History and physical documented separately.",
    "Comparison made with available priors.",
    "Technique and contrast administration as per protocol.",
    "Findings discussed with the referring team.",
    "Report dictated and electronically signed.",
    "",
]

_NEIGHBOURS = {
    "normal": "attention",
    "attention": "normal",
    "urgent": "attention",
}

_AMBIGUITY_RATE = 0.18


def _make_row(rng: random.Random, label: str) -> str:
    template = rng.choice(_TEMPLATES)
    sentence = template.format(
        modality=rng.choice(_MODALITIES),
        site=rng.choice(_BODY_SITES),
        finding=rng.choice(_FINDINGS[label]),
    )

    parts = [sentence]
    if rng.random() < _AMBIGUITY_RATE:
        parts.append(f"Also noted: {rng.choice(_FINDINGS[_NEIGHBOURS[label]])}.")
        parts.append(rng.choice(_SHARED_FILLER))
    else:
        parts.append(rng.choice(_QUALIFIERS[label]))
        parts.append(rng.choice(_SHARED_FILLER))

    return " ".join(part for part in parts if part).strip()


def generate_dataframe(n_samples: int = DATASET_SIZE, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = random.Random(seed)
    per_label = n_samples // len(URGENCY_LABELS)
    rows = [
        {"text": _make_row(rng, label), "urgency": label}
        for label in URGENCY_LABELS
        for _ in range(per_label)
    ]
    rng.shuffle(rows)
    return pd.DataFrame(rows, columns=["text", "urgency"])


def build_dataset(
    path: Path = DATASET_PATH,
    n_samples: int = DATASET_SIZE,
    seed: int = RANDOM_SEED,
) -> Path:
    frame = generate_dataframe(n_samples, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    written = build_dataset()
    print(f"wrote {written}")
