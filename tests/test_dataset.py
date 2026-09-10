from triage_api import URGENCY_LABELS
from triage_api.dataset import generate_dataframe


def test_generator_is_deterministic():
    first = generate_dataframe(n_samples=300, seed=7)
    second = generate_dataframe(n_samples=300, seed=7)
    assert first.equals(second)


def test_generator_covers_every_label_and_size():
    frame = generate_dataframe(n_samples=900, seed=1)
    assert set(frame["urgency"]) == set(URGENCY_LABELS)
    assert len(frame) == 900
    assert frame["text"].str.len().min() > 0
