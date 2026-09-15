import json

import pytest

from triage_api.config import METRICS_PATH, MODEL_PATH, ONNX_MODEL_PATH
from triage_api.enums import Urgency
from triage_api.model import TriageModel
from triage_api.onnx_model import OnnxTriageModel
from triage_api.onnx_validate import validate_onnx
from triage_api.samples import sample_reports

REPORTS = sample_reports(limit=60)

PROBABILITY_TOLERANCE = 5e-2
NEAR_TIE_GAP = 0.05


def _top_two_gap(probabilities: dict[Urgency, float]) -> float:
    ranked = sorted(probabilities.values(), reverse=True)
    return ranked[0] - ranked[1]


def test_onnx_probabilities_track_sklearn():
    sklearn_model = TriageModel.load(MODEL_PATH)
    onnx_model = OnnxTriageModel.load(ONNX_MODEL_PATH)

    for text in REPORTS:
        expected = sklearn_model.predict(text)
        actual = onnx_model.predict(text)
        for urgency, probability in expected.probabilities.items():
            assert actual.probabilities[urgency] == pytest.approx(
                probability, abs=PROBABILITY_TOLERANCE
            )


def test_onnx_labels_only_differ_on_near_ties():
    sklearn_model = TriageModel.load(MODEL_PATH)
    onnx_model = OnnxTriageModel.load(ONNX_MODEL_PATH)

    for text in REPORTS:
        expected = sklearn_model.predict(text)
        actual = onnx_model.predict(text)
        if actual.urgency != expected.urgency:
            assert _top_two_gap(expected.probabilities) < NEAR_TIE_GAP


def test_onnx_probabilities_are_well_formed():
    onnx_model = OnnxTriageModel.load(ONNX_MODEL_PATH)
    result = onnx_model.predict(REPORTS[0])

    assert set(result.probabilities) == set(Urgency)
    assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=1e-3)
    assert result.urgency == max(result.probabilities, key=result.probabilities.get)


def test_validate_onnx_passes_for_the_freshly_trained_model():
    sklearn_metrics = json.loads(METRICS_PATH.read_text())

    onnx_metrics = validate_onnx(sklearn_metrics)

    assert onnx_metrics["macro_f1"] > 0


def test_validate_onnx_rejects_a_large_macro_f1_drift():
    with pytest.raises(ValueError, match="diverges"):
        validate_onnx({"macro_f1": 0.99})
