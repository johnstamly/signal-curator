import json

from Curation_Paper.eswa_full.lib.protocol import load_config
from Curation_Paper.eswa_full.lib.runio import load_result, save_result


def test_save_result_records_provenance(tmp_path):
    cfg = load_config() | {"results_dir": str(tmp_path)}
    p = save_result("probe", {"x": 1}, cfg)
    meta = json.loads(p.read_text())["meta"]
    assert set(meta) >= {"git_sha", "input_hashes", "seeds", "created"}
    assert len(meta["git_sha"]) == 40


def test_load_result_roundtrip(tmp_path):
    cfg = load_config() | {"results_dir": str(tmp_path)}
    save_result("probe", {"x": 1}, cfg)
    assert load_result("probe", cfg)["x"] == 1
