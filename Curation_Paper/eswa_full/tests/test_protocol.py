import pytest

from Curation_Paper.eswa_full.lib.protocol import (
    InputMismatch, load_config, load_pools, reference_labels, verify_inputs)


def test_verify_inputs_passes_on_frozen_files():
    out = verify_inputs(load_config())
    assert out["n_al"] == 1011 and out["n_rand"] == 1011
    assert out["n_shared"] == 746 and out["n_union"] == 1276
    assert out["n_shared_disagree"] == 13 and out["n_missing_in_h5"] == 0


def test_verify_inputs_refuses_hash_mismatch(tmp_path):
    cfg = load_config()
    cfg["pools"]["al"]["sha256"] = "0" * 64
    with pytest.raises(InputMismatch):
        verify_inputs(cfg)


def _copy_cfg_dbs(tmp_path):
    import shutil
    from Curation_Paper.eswa_full.lib.protocol import repo_path
    cfg = load_config()
    for name, spec in cfg["pools"].items():
        dst = tmp_path / f"{name}.db"
        shutil.copyfile(repo_path(spec["path"]), dst)
        spec["path"] = str(dst)
    return cfg


def test_verify_inputs_refuses_nonempty_wal(tmp_path):
    cfg = _copy_cfg_dbs(tmp_path)
    verify_inputs(cfg)  # the copies pass with no WAL
    (tmp_path / "al.db-wal").write_bytes(b"x" * 32)
    with pytest.raises(InputMismatch, match="wal"):
        verify_inputs(cfg)


def test_verify_inputs_allows_empty_wal(tmp_path):
    cfg = _copy_cfg_dbs(tmp_path)
    (tmp_path / "rand.db-wal").write_bytes(b"")
    verify_inputs(cfg)


def test_verify_inputs_refuses_too_short_window():
    cfg = load_config()
    cfg["window"] = 5000
    with pytest.raises(InputMismatch, match="window"):
        verify_inputs(cfg)


def test_reference_labels_use_al_label_on_disagreement():
    al, rand = load_pools(load_config())
    ref = reference_labels(al, rand)
    assert len(ref) == 746
    assert all(ref[s] == al[s]["label_int"] for s in ref)
