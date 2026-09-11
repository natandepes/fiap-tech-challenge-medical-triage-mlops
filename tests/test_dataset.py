from triage_api.dataset import generate_dataframe
from triage_api.enums import Urgency


def test_generator_is_deterministic():
    first = generate_dataframe(n_samples=300, seed=7)
    second = generate_dataframe(n_samples=300, seed=7)
    assert first.equals(second)


def test_generator_covers_every_label_and_size():
    frame = generate_dataframe(n_samples=900, seed=1)
    assert set(frame["urgency"]) == set(Urgency)
    assert len(frame) == 900
    assert frame["text"].str.len().min() > 0
