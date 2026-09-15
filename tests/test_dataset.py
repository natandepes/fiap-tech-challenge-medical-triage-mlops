import pandas as pd
import pytest

from triage_api.dataset import (
    LABEL_COLUMN,
    TEXT_COLUMN,
    URGENCY_BY_CONDITION,
    download_corpus,
    to_triage_frame,
)
from triage_api.enums import Condition, Urgency


@pytest.fixture
def corpus():
    return pd.DataFrame(
        {
            LABEL_COLUMN: [
                Condition.CARDIOVASCULAR,
                Condition.NEOPLASMS,
                Condition.DIGESTIVE_SYSTEM,
                Condition.NERVOUS_SYSTEM,
                Condition.GENERAL_PATHOLOGICAL,
            ],
            TEXT_COLUMN: [
                "  Acute myocardial infarction in a cohort of 40 patients.  ",
                "Adenocarcinoma of the colon staged after resection.",
                "Chronic gastritis reviewed over a twelve month period.",
                "Peripheral neuropathy following prolonged exposure.",
                "Inflammatory response measured across the study group.",
            ],
        }
    )


def test_every_condition_label_maps_to_an_urgency():
    assert set(URGENCY_BY_CONDITION) == set(Condition)
    assert set(URGENCY_BY_CONDITION.values()) == set(Urgency)


def test_mapping_produces_the_documented_urgency(corpus):
    frame = to_triage_frame(corpus)
    urgency_by_text = dict(zip(frame["text"], frame["urgency"], strict=True))

    assert urgency_by_text["Acute myocardial infarction in a cohort of 40 patients."] == (
        Urgency.URGENT
    )
    assert urgency_by_text["Adenocarcinoma of the colon staged after resection."] == (
        Urgency.ATTENTION
    )
    assert urgency_by_text["Chronic gastritis reviewed over a twelve month period."] == (
        Urgency.NORMAL
    )


def test_mapping_strips_whitespace_and_keeps_every_row(corpus):
    frame = to_triage_frame(corpus)
    assert len(frame) == len(corpus)
    assert not frame["text"].str.startswith(" ").any()
    assert list(frame.columns) == ["text", "urgency"]


def test_mapping_is_deterministic_for_a_given_seed(corpus):
    assert to_triage_frame(corpus, seed=7).equals(to_triage_frame(corpus, seed=7))


def test_download_is_skipped_when_the_raw_corpus_is_cached(tmp_path, monkeypatch):
    from triage_api import dataset

    for filename in dataset.CORPUS_FILES:
        (tmp_path / filename).write_text(f"{LABEL_COLUMN},{TEXT_COLUMN}\n1,abstract\n")

    def fail(*_args, **_kwargs):
        raise AssertionError("cached corpus should not be downloaded again")

    monkeypatch.setattr(dataset.urllib.request, "urlretrieve", fail)
    assert len(download_corpus(tmp_path)) == len(dataset.CORPUS_FILES)
