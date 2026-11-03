"""E11: downstream value of curation (rulings R24, R25).  CPU only, no training.

Curated = universe minus signals labelled BAD (reference = L_AL label where both pools hold the
signal) minus unlabelled signals with E10 mean p < 0.5.  Uncurated = the whole universe.
Per panel: monotonicity, panel-DI Spearman and Coble-Hines trendability, both conditions and the
paired difference.  Reported whichever way it comes out.  From the repo root:
    conda run --no-capture-output -n AI python Curation_Paper/eswa_full/scripts/e11_downstream.py
"""
import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.chdir(REPO)

import numpy as np  # noqa: E402

from Curation_Paper.eswa_full.lib import downstream as D  # noqa: E402
from Curation_Paper.eswa_full.lib.protocol import (  # noqa: E402
    REPO_ROOT, load_config, load_pools, repo_path, sha256_file, verify_inputs)
from Curation_Paper.eswa_full.lib.runio import git_sha, load_result, save_result  # noqa: E402
from Curation_Paper.eswa_full.lib.signals import load_universe, parse_id  # noqa: E402
from Curation_Paper.eswa_full.lib.versioning import dirty_files  # noqa: E402

CODE_FILES = ["Curation_Paper/eswa_full/lib/downstream.py",
              "Curation_Paper/eswa_full/scripts/e11_downstream.py",
              "Curation_Paper/eswa_full/tests/test_downstream.py"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="E11 downstream value of curation")
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args(argv)
    cfg = load_config()
    if args.results_dir:
        cfg["results_dir"] = args.results_dir
    verify_inputs(cfg)
    al, rand = load_pools(cfg)
    ids, X = load_universe(cfg)
    if len(ids) != 12504 and not args.results_dir:
        print(f"REFUSING: universe has {len(ids)} signals, expected 12504", file=sys.stderr)
        return 6
    e10 = load_result("e10_scores", cfg if args.results_dir is None else load_config())
    if list(e10["ids"]) != list(ids):
        print("REFUSING: e10_scores ids differ from the universe order", file=sys.stderr)
        return 7
    mean_p = np.asarray(e10["mean_p"], dtype=float)

    keep_cur = D.curated_mask(ids, mean_p, al, rand)
    keep_all = np.ones(len(ids), dtype=bool)
    di_cur, cnt_cur = D.damage_indices(ids, X, keep_cur)
    di_unc, cnt_unc = D.damage_indices(ids, X, keep_all)
    panels = D.compare_conditions(di_cur, di_unc)
    # R27 controls: common-cycle (score uncurated only on the curated condition's path-cycles)
    # and labels-only curation (remove reference-BAD only; no model removal)
    keep_lab = D.labels_only_mask(ids, al, rand)
    di_lab, cnt_lab = D.damage_indices(ids, X, keep_lab)
    unc_on_cur = D.restrict_to_common(di_unc, di_cur)
    unc_on_lab = D.restrict_to_common(di_unc, di_lab)
    common = D.compare_conditions(di_cur, unc_on_cur)
    lab_full = D.compare_conditions(di_lab, di_unc)
    lab_common = D.compare_conditions(di_lab, unc_on_lab)
    n_pc = {"uncurated": D.count_path_cycles(di_unc), "curated": D.count_path_cycles(di_cur),
            "labels_only": D.count_path_cycles(di_lab),
            "uncurated_restricted_to_curated": D.count_path_cycles(unc_on_cur),
            "uncurated_restricted_to_labels_only": D.count_path_cycles(unc_on_lab)}
    panel_of = [parse_id(sid)["panel"] for sid in ids]
    lab_sig = {}
    for panel in sorted(set(panel_of)):
        idx = [i for i, pn in enumerate(panel_of) if pn == panel]
        lab_sig[panel] = {"n_signals": len(idx), "n_kept": int(keep_lab[idx].sum()),
                          "kept_fraction": float(keep_lab[idx].mean())}
    for res in (lab_full, lab_common):
        for panel, d in res.items():
            d["coverage_signals"] = lab_sig[panel]
            d["coverage_path_cycles"] = {
                "labels_only": D.count_path_cycles(
                    {k: v for k, v in di_lab.items() if k[0] == panel}),
                "uncurated": D.count_path_cycles(
                    {k: v for k, v in di_unc.items() if k[0] == panel})}
    for panel, d in common.items():
        d["coverage_path_cycles"] = {
            "curated": D.count_path_cycles({k: v for k, v in di_cur.items() if k[0] == panel}),
            "uncurated_restricted": D.count_path_cycles(
                {k: v for k, v in unc_on_cur.items() if k[0] == panel}),
            "uncurated_full": D.count_path_cycles(
                {k: v for k, v in di_unc.items() if k[0] == panel})}

    meta = [parse_id(s) for s in ids]
    lab = [D.reference_label(s, al, rand) for s in ids]
    comp = {"n_universe": len(ids),
            "n_labelled_in_universe": sum(l is not None for l in lab),
            "n_removed_labelled_bad": sum(l == 0 for l in lab),
            "n_kept_labelled_good": sum(l == 1 for l in lab),
            "n_unlabelled": sum(l is None for l in lab),
            "n_removed_unlabelled_low_p": sum(1 for l, k in zip(lab, keep_cur) if l is None and not k),
            "n_kept_unlabelled": sum(1 for l, k in zip(lab, keep_cur) if l is None and k),
            "n_curated": int(keep_cur.sum())}
    sig_cov = {}
    for panel in sorted({m["panel"] for m in meta}):
        idx = [i for i, m in enumerate(meta) if m["panel"] == panel]
        sig_cov[panel] = {"n_signals": len(idx), "n_kept": int(keep_cur[idx].sum()),
                          "kept_fraction": float(keep_cur[idx].mean())}
    cyc_lt = sorted({m["cycle"] for m in meta if m["cycle"] < D.MIN_CYCLE})
    below = {"min_cycle": D.MIN_CYCLE, "cycles_below_min_in_universe": cyc_lt,
             "n_signals_below_min": sum(m["cycle"] < D.MIN_CYCLE for m in meta),
             "handling": "signals at cycles < 200 are excluded from every DI trajectory and from "
                         "the Spearman/trendability computations (cycle 0 has different boundary "
                         "conditions); they remain in the 'coverage' signal counts and in "
                         "curated-set membership"}

    # path-cycle coverage per panel and condition
    def pc_cov(di):
        out = {}
        for panel, paths in D.split_by_panel(di).items():
            out[panel] = {"n_paths": len(paths),
                          "n_path_cycles": sum(len(t) for t in paths.values())}
        return out
    pc_cur, pc_unc = pc_cov(di_cur), pc_cov(di_unc)
    for panel, d in panels.items():
        d["coverage"] = {
            "signals": sig_cov[panel],
            "curated": pc_cur.get(panel, {"n_paths": 0, "n_path_cycles": 0}),
            "uncurated": pc_unc.get(panel, {"n_paths": 0, "n_path_cycles": 0}),
            "path_cycle_fraction_retained":
                pc_cur.get(panel, {"n_path_cycles": 0})["n_path_cycles"]
                / pc_unc[panel]["n_path_cycles"]}

    payload = {
        "definition": {
            "curated": "universe minus signals labelled BAD in either pool (reference label = L_AL "
                       "where both pools hold the signal) minus unlabelled signals with E10 mean "
                       "p < 0.5; labelled GOOD always kept",
            "uncurated": "the whole 12,504-signal universe",
            "di": "per path (panel, actuator, receiver) and cycle: mean over the condition's "
                  "repetitions of RMS(x - b); x = raw first-500-sample window, no z-scoring; "
                  "b = mean of that condition's kept repetitions at the path's baseline cycle, "
                  "which is the path's first cycle >= 200 in the universe, chosen "
                  "independently of the condition",
            "monotonicity": "mean over paths of |Spearman(DI, cycle)| (nan paths excluded, counted)",
            "panel_di": "median over paths per cycle; |Spearman| vs cycle",
            "trendability": "Coble-Hines form: min and mean pairwise Pearson of the paths' DI "
                            "trajectories over common cycles (units = paths; undefined pairs "
                            "skipped, counted)",
            "scores_source": "results/e10_scores.json (R24): E10 MicroConv1D on all of L_AL, "
                             "seeds 0-4, scores averaged; no model retrained",
            "threshold_p_accept": D.P_ACCEPT},
        "n_signals": len(ids), "window": cfg["window"],
        "composition_of_curated_set": comp,
        "cycles_below_200": below,
        "counts": {"curated": cnt_cur, "uncurated": cnt_unc},
        "panels": panels,
        "common_cycle": {"description": "both conditions scored only on the path-cycles present "
                         "in the curated condition (uncurated path-cycles missing in curated "
                         "dropped); curated vs uncurated", "panels": common},
        "labels_only": {"description": "removes only signals whose reference label is BAD "
                        "(L_AL wins); no model removal.  'full' = vs whole uncurated; "
                        "'common_cycle' = vs uncurated restricted to the labels-only path-cycles",
                        "n_signals_kept": int(keep_lab.sum()), "counts": cnt_lab,
                        "full": lab_full, "common_cycle": lab_common},
        "path_cycle_counts_used": n_pc,
        "e10_scores_seeds": e10.get("seeds"),
        "extra_input_hashes": {
            "e10_scores.json": sha256_file(REPO / "Curation_Paper/eswa_full/results/e10_scores.json"),
            "morpho_50khz_full.h5": sha256_file(repo_path(cfg["signals_h5"]))},
        "code": {"git_sha": git_sha(), "files": CODE_FILES,
                 "dirty_files": dirty_files(CODE_FILES, REPO_ROOT)},
    }
    out = save_result("e11_downstream", payload, cfg)
    print("->", out)
    print(f"curated {comp['n_curated']}/{comp['n_universe']}; "
          f"removed BAD {comp['n_removed_labelled_bad']}, low-p unlabelled "
          f"{comp['n_removed_unlabelled_low_p']}")
    print(f"{'panel':6s} {'cond':9s} {'paths':>5s} {'mono':>6s} {'pDI':>6s} {'tmin':>7s} {'tmean':>7s}")
    for panel, d in panels.items():
        for cond in ("uncurated", "curated"):
            m = d[cond]
            print(f"{panel:6s} {cond:9s} {m['n_paths']:5d} {m['monotonicity']:6.3f} "
                  f"{m['panel_di_abs_spearman']:6.3f} {m['trend_min']:7.3f} {m['trend_mean']:7.3f}")
        print(f"{panel:6s} diff      {'':5s} {d['difference']['monotonicity']:+6.3f} "
              f"{d['difference']['panel_di_abs_spearman']:+6.3f} {d['difference']['trend_min']:+7.3f} "
              f"{d['difference']['trend_mean']:+7.3f}")
    for title, res in (("common-cycle (curated vs uncurated on curated path-cycles)", common),
                       ("labels-only, full", lab_full), ("labels-only, common-cycle", lab_common)):
        print(title, n_pc)
        for panel, d in res.items():
            df = d["difference"]
            print(f"  {panel} diff mono {df['monotonicity']:+.3f} pDI {df['panel_di_abs_spearman']:+.3f} "
                  f"tmin {df['trend_min']:+.3f} tmean {df['trend_mean']:+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
