"""E0: verify frozen inputs and record counts. Run from the repo root:
    python -m Curation_Paper.eswa_full.scripts.e00_verify_inputs
"""
from Curation_Paper.eswa_full.lib.protocol import load_config, verify_inputs
from Curation_Paper.eswa_full.lib.runio import save_result


def main() -> None:
    cfg = load_config()
    out = verify_inputs(cfg)
    path = save_result("e00_inputs", out, cfg)
    print({k: v for k, v in out.items() if k.startswith("n_")}, "->", path)


if __name__ == "__main__":
    main()
