import csv
import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("torch")

from compare_experiments import clean_history, config_diff


def test_rewind_removes_abandoned_tail_and_keeps_latest_duplicate(tmp_path):
    path = tmp_path / "metrics.csv"
    rows = [(10, "a"), (20, "old"), (30, "abandoned"),
            (20, "resumed"), (25, "new"), (25, "latest"), (40, "final")]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step", "value"])
        writer.writerows(rows)
    before = path.read_bytes()
    retained, audit = clean_history(path)
    assert [(int(r["step"]), r["value"]) for r in retained] == [
        (10, "a"), (20, "resumed"), (25, "latest"), (40, "final")]
    assert audit["superseded_rows"] == 3
    assert len(audit["rewinds"]) == 2
    assert path.read_bytes() == before


def test_config_comparison_includes_implicit_and_nested_differences():
    differences = config_diff(
        {"seed": 1, "training": {"gamma": .99, "total_steps": 100}},
        {"seed": 2, "training": {"gamma": .95, "total_steps": 100}})
    assert differences == [
        {"field": "seed", "left": 1, "right": 2},
        {"field": "training.gamma", "left": .99, "right": .95}]
