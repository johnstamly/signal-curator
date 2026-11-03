"""Registry of macro specs per study (Task 14, ruling R28).

A study contributes a ``builder(data) -> [Macro]`` that expands its specs against the loaded
JSON (so budgets, strategies, etas and the E5 architecture selection come from the data, never
from a hand-typed list).  A ``Macro`` names a JSON key path, a format and where its seed count
lives.  Naming: ``\\<StudyPrefix><Metric><Variant><Budget>``, CamelCase, letters only
(``\\EoneAccAlBsix`` = E1, accuracy, AL pipeline, budget 600).

Macro kinds: value (``part="value"``), CI (``part="ci"``: "a--b" from two key paths) and
paired difference (``minus`` set: mean / bootstrap CI of the per-seed differences of two
``values`` arrays, each side with its own seed count; the two must agree).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .numbers_core import tag

METRICS = {"acc": "Acc", "bal_acc": "BalAcc", "bad_recall": "BadRecall",
           "good_retention": "GoodRet", "auroc": "Auroc", "ece": "Ece"}
MODEL_TAG = {"microconv1d": "Micro", "cnn_wide": "CnnWide", "resnet1d": "Resnet",
             "bigru": "BiGru", "transformer": "Transformer", "hgb": "Hgb", "logreg": "Logreg",
             "minirocket": "MiniRocket", "microconv1d_baseline": "MicroBase"}
CONFIG_TAG = {"unc_oracle_al": "Al", "random_rand": "Rand", "random_al": "RandAl"}
STRAT_TAG = {"uncertainty": "Unc", "unc_oracle": "Oracle", "unc_selftrain": "Self"}
VARIANT_TAG = {"default": "Default", "relu": "Relu", "class_weight_none": "CwNone",
               "class_weight_full": "CwFull", "no_augment": "NoAug", "window_250": "WinTwoFifty",
               "window_1000": "WinThousand", "no_zscore": "NoZ"}


@dataclass(frozen=True)
class Macro:
    name: str
    study: str
    path: tuple                      # key path into the study's data (dict tokens / (key, value))
    kind: str                        # pct | pp | ppnum | int | f1 | f2 | f3 | sec | bool | text
    seeds: object = None             # None (deterministic) | int | key path (tuple) -> list/int
    part: str = "value"              # value | ci
    ci_paths: tuple | None = None    # (lo_path, hi_path) when part == "ci"
    not_defined: bool = False        # NaN allowed -> "n/a"
    minus: tuple | None = None       # path of the subtrahend stat dict (paired difference)
    minus_study: str | None = None
    minus_seeds: object = None
    note: str = ""


@dataclass
class Study:
    key: str
    file: str | None                 # JSON file name under results/ (None = computed)
    prefix: str
    mode: str                        # cells | static | none  (which provenance fields it carries)
    kind: str = "loop"               # cell kind of a single cells_code_key
    builder: Callable = field(default=lambda data: [], repr=False)
    loader: Callable | None = None   # computed studies (fold-0 slice)
    stage2: bool = False             # E1-E4: results/<file stem>_s20.json (20 seeds) is preferred when present


# ------------------------------------------------------------------ helpers
def _stat(name, study, base, kind, seeds, *, nd=False, extra=()):
    """Value + CI macros of a stat dict {mean, ci_lo, ci_hi}."""
    return [Macro(name, study, base + ("mean",), kind, seeds, "value", None, nd),
            Macro(name + "Ci", study, base, kind, seeds, "ci",
                  (base + ("ci_lo",), base + ("ci_hi",)), nd)]


def _diff(name, study, a, b, kind, seeds_a, seeds_b, *, minus_study=None):
    """Mean and CI of (a.values - b.values); a, b are stat-dict paths."""
    return [Macro(name, study, a, kind, seeds_a, "value", None, False, b, minus_study, seeds_b),
            Macro(name + "Ci", study, a, kind, seeds_a, "ci", None, False, b, minus_study,
                  seeds_b)]


def bt(budget) -> str:
    """Budget tag: hundreds spelled, lower case after B (600 -> Bsix, as in the brief)."""
    b = int(budget)
    return "B" + (tag(b // 100).lower() if b % 100 == 0 else tag(b))


def _row(rows_path, budget):
    return rows_path + (("budget", budget),)


def curve_macros(study, prefix, variant_tag, rows_path, budgets, metrics, *, seeds_top=None):
    out = []
    for b in budgets:
        r = _row(rows_path, b)
        for m in metrics:
            out += _stat(f"{prefix}{METRICS[m]}{variant_tag}{bt(b)}", study,
                         r + ("metrics", m), "pct", r + ("n_seeds",))
    return out


def _eta_tag(eta):
    return "Eta" + tag(round(float(eta) * 100)) + "Pct"


# ------------------------------------------------------------------ e00 (inputs)
def b_e00(d):
    out = []
    for k, n in (("n_al", "NAl"), ("n_rand", "NRand"), ("n_shared", "NShared"),
                 ("n_union", "NUnion"), ("n_shared_disagree", "NDisagree"),
                 ("shared_good_al", "SharedGood"), ("shared_bad_al", "SharedBad")):
        out.append(Macro("Ezero" + n, "e00", (k,), "int", None))
    return out


# ------------------------------------------------------------------ e01 selection
E1_METRICS = ("acc", "bal_acc", "bad_recall", "auroc")


def b_e01(d):
    out, P, S = [], "Eone", "e01"
    budgets = d["budgets"]
    for c in d["configs"]:
        out += curve_macros(S, P, CONFIG_TAG[c], ("configs", c, "rows"), budgets, E1_METRICS)
        for b in budgets:
            r = _row(("configs", c, "rows"), b)
            out.append(Macro(f"{P}ClaimOk{CONFIG_TAG[c]}{bt(b)}", S, r + ("selection_claim_ok",),
                             "bool", r + ("n_seeds",)))
    top = budgets[-1]
    for c in d["configs"]:
        out.append(Macro(f"{P}MinPool{CONFIG_TAG[c]}", S, _row(("configs", c, "rows"), top)
                         + ("min_train_pool",), "int", ("n_seeds",)))
        out += curve_macros(S, P + "Sens", CONFIG_TAG[c],
                            ("sensitivity_excluding_revised", c, "rows"), [top],
                            ("acc", "bal_acc", "bad_recall"))
    if {"unc_oracle_al", "random_al", "random_rand"} <= set(d["configs"]):
        for b in budgets:
            ra = _row(("configs", "unc_oracle_al", "rows"), b)
            rb = _row(("configs", "random_al", "rows"), b)
            rc = _row(("configs", "random_rand", "rows"), b)
            for m in ("bal_acc", "acc"):
                out += _diff(f"{P}Sel{METRICS[m]}Diff{bt(b)}", S, ra + ("metrics", m),
                             rb + ("metrics", m), "pp", ra + ("n_seeds",), rb + ("n_seeds",))
                out += _diff(f"{P}Pool{METRICS[m]}Diff{bt(b)}", S, rb + ("metrics", m),
                             rc + ("metrics", m), "pp", rb + ("n_seeds",), rc + ("n_seeds",))
            # unit-free twin for prose that spells out "percentage points" (abstract)
            out.append(_diff(f"{P}SelBalAccDiffNum{bt(b)}", S, ra + ("metrics", "bal_acc"),
                             rb + ("metrics", "bal_acc"), "ppnum", ra + ("n_seeds",),
                             rb + ("n_seeds",))[0])
    out.append(Macro(f"{P}NScored", S, _row(("configs", "unc_oracle_al", "rows"), top)
                     + ("n_scored",), "int", ("n_seeds",)))
    # seed count of E1's own file: 5 on Stage 1, 20 once e01_selection_s20.json exists (Task 16).
    # The Stage-1 studies (E5-E7, E8, E10) keep their own count in EsevenNSeeds.
    out.append(Macro(f"{P}NSeeds", S, ("n_seeds",), "int", ("n_seeds",)))
    return out


# ------------------------------------------------------------------ e02 / e03
def _strat_eta_macros(d, S, P, budgets, metrics, counters_at):
    out = []
    for key, c in d["cells"].items():
        v = STRAT_TAG[c["strategy"]] + _eta_tag(c["eta"])
        base = ("cells", key, "rows")
        out += curve_macros(S, P, v, base, budgets, metrics)
        r = _row(base, counters_at)
        for ctr in ("recovery_rate", "false_flag_rate"):
            nd = ctr == "recovery_rate" and float(c["eta"]) == 0.0
            out += _stat(f"{P}{'Recovery' if ctr == 'recovery_rate' else 'FalseFlag'}{v}{bt(counters_at)}",
                         S, r + ("counters", ctr), "pct", r + ("n_seeds",), nd=nd)
    return out


def b_e02(d):
    return _strat_eta_macros(d, "e02", "Etwo", [300, 600], ("acc", "bal_acc", "bad_recall"), 600)


def b_e03(d):
    out = _strat_eta_macros(d, "e03", "Ethree", [300, 600], ("acc", "bal_acc", "bad_recall"), 600)
    for key, c in d["cells"].items():
        if c["strategy"] != "unc_oracle":
            continue
        other = f"uncertainty|eta={c['eta']:g}"
        if other in d["cells"]:
            ra = _row(("cells", key, "rows"), 600)
            rb = _row(("cells", other, "rows"), 600)
            out += _diff(f"EthreeOracleGainBalAcc{_eta_tag(c['eta'])}{bt(600)}", "e03",
                         ra + ("metrics", "bal_acc"), rb + ("metrics", "bal_acc"), "pp",
                         ra + ("n_seeds",), rb + ("n_seeds",))
            out.append(_diff(f"EthreeOracleGainBalAccNum{_eta_tag(c['eta'])}{bt(600)}", "e03",
                             ra + ("metrics", "bal_acc"), rb + ("metrics", "bal_acc"), "ppnum",
                             ra + ("n_seeds",), rb + ("n_seeds",))[0])
    # E9's learned-model row (e09_rules.json has microconv1d: null by design)
    r = _row(("cells", "unc_oracle|eta=0", "rows"), 600)
    out += _stat("EnineLearnedBadRecall", "e03", r + ("metrics", "bad_recall"), "pct", r + ("n_seeds",))
    out += _stat("EnineLearnedGoodKept", "e03", r + ("metrics", "good_retention"), "pct", r + ("n_seeds",))
    return out


# ------------------------------------------------------------------ e04
def b_e04_frozen(d):
    out, P, S = [], "EfourFrozen", "e04_frozen"
    seeds = ("tune_seeds",)
    out += [Macro("EfourTuneNTrain", S, ("tune_n_train",), "int", seeds),
            Macro("EfourTuneFold", S, ("tune_fold",), "int", seeds)]
    for m, mv in d["models"].items():
        t = MODEL_TAG[m]
        f = ("models", m, "frozen")
        out += [Macro(f"{P}Config{t}", S, f + ("name",), "text", seeds),
                Macro(f"{P}EsLoss{t}", S, f + ("es_loss_mean",), "f3", seeds),
                Macro(f"{P}EsAuroc{t}", S, f + ("es_auroc_mean",), "f3", seeds),
                Macro(f"{P}Size{t}", S, f + ("size",), "int", seeds),
                Macro(f"{P}IsDefault{t}", S, f + ("equals_default",), "bool", seeds)]
    return out


def b_e04_compare(d):
    out, P, S = [], "Efour", "e04_compare"
    for m, mv in d["models"].items():
        t = MODEL_TAG[m]
        base = ("models", m, "rows")
        for b in (200, 600):
            r = _row(base, b)
            for k in ("acc", "bal_acc", "auroc", "bad_recall") + (() if m == "minirocket" else ("ece",)):
                out += _stat(f"{P}{METRICS[k]}{t}{bt(b)}", S, r + ("metrics", k), "pct",
                             r + ("n_seeds",))
            out += _stat(f"{P}FitSec{t}{bt(b)}", S, r + ("train_seconds_per_fit",), "sec",
                         r + ("n_seeds",))
        out += [Macro(f"{P}Size{t}", S, ("models", m, "size"), "int", ("n_seeds",)),
                Macro(f"{P}Calibrated{t}", S, ("models", m, "calibrated"), "bool", ("n_seeds",)),
                Macro(f"{P}FitSecMean{t}", S, ("models", m, "train_seconds_per_fit_mean"), "sec",
                      ("n_seeds",))]
    for ma, mb in (("microconv1d", "minirocket"), ("cnn_wide", "resnet1d"), ("cnn_wide", "bigru"),
                   ("resnet1d", "bigru")):        # paired model differences at 600 labels (final review M1, M2)
        if ma in d["models"] and mb in d["models"]:
            ra, rb = _row(("models", ma, "rows"), 600), _row(("models", mb, "rows"), 600)
            out += _diff(f"{P}{MODEL_TAG[ma]}Minus{MODEL_TAG[mb]}BalAccBsix", S, ra + ("metrics", "bal_acc"),
                         rb + ("metrics", "bal_acc"), "pp", ra + ("n_seeds",), rb + ("n_seeds",))
    if "microconv1d" in d["models"]:        # tuned (E4) vs deployed default (E1 random/AL), Task 17 carry
        ra = _row(("models", "microconv1d", "rows"), 600)
        rb = _row(("configs", "random_al", "rows"), 600)
        out += _diff("EfourTunedMinusDefaultBalAccBsix", S, ra + ("metrics", "bal_acc"),
                     rb + ("metrics", "bal_acc"), "pp", ra + ("n_seeds",), rb + ("n_seeds",),
                     minus_study="e01")
    return out


# ------------------------------------------------------------------ e05
RANK = ["One", "Two", "Three", "Four", "Five"]


def b_e05(d):
    out, P, S = [], "Efive", "e05"
    for i, sel in enumerate(d["selected"]):
        v = "Rank" + RANK[i]
        out.append(Macro(f"{P}{v}Arch", S, ("selected", i, "arch"), "text", ("n_seeds",)))
        base = ("models", sel["arch"], "rows")
        out += curve_macros(S, P, v, base, [600], ("acc", "bal_acc", "auroc"))
        if "microconv1d_baseline" in d:
            ra, rb = _row(base, 600), _row(("microconv1d_baseline", "rows"), 600)
            out += _diff(f"{P}{v}MinusMicroBalAccBsix", S, ra + ("metrics", "bal_acc"),
                         rb + ("metrics", "bal_acc"), "pp", ra + ("n_seeds",), rb + ("n_seeds",))
    for ma, mb in (("resnet1d", "cnn_wide"), ("resnet1d", "bigru"), ("cnn_wide", "bigru")):
        if ma in d["models"] and mb in d["models"]:     # paired differences among the three others (M2)
            ra, rb = _row(("models", ma, "rows"), 600), _row(("models", mb, "rows"), 600)
            out += _diff(f"{P}{MODEL_TAG[ma]}Minus{MODEL_TAG[mb]}BalAccBsix", S, ra + ("metrics", "bal_acc"),
                         rb + ("metrics", "bal_acc"), "pp", ra + ("n_seeds",), rb + ("n_seeds",))
    if "microconv1d_baseline" in d:
        out += curve_macros(S, P, "Micro", ("microconv1d_baseline", "rows"), [600],
                            ("acc", "bal_acc", "auroc"))
    return out


# ------------------------------------------------------------------ e06
def b_e06(d):
    out, P, S = [], "Esix", "e06"
    a = d["ablate"]
    sd = ("ablate", "n_seeds")
    for v, ve in a["variants"].items():
        t = VARIANT_TAG[v]
        for k in ("bal_acc", "auroc"):
            out += _stat(f"{P}{METRICS[k]}{t}", S, ("ablate", "variants", v, "metrics", k), "pct", sd)
        out.append(Macro(f"{P}Params{t}", S, ("ablate", "variants", v, "n_params"), "int", sd))
        if ve.get("paired_vs_default"):
            for k in ("bal_acc", "auroc"):
                out += _stat(f"{P}{METRICS[k]}{t}VsDefault", S,
                             ("ablate", "variants", v, "paired_vs_default", k), "pp", sd)
    w = d["warm_start"]
    for b in (300, 500, 600):
        pr = ("warm_start", "paired", ("budget", b))
        for k in ("bal_acc", "auroc"):
            out += _stat(f"{P}Warm{METRICS[k]}MinusScratch{bt(b)}", S,
                         pr + ("warm_minus_scratch", k), "pp", pr + ("n_seeds",))
    out += curve_macros(S, P + "Scratch", "", ("warm_start", "scratch_rows"), [600], ("bal_acc",))
    out += curve_macros(S, P + "Warm", "", ("warm_start", "warm_rows"), [600], ("bal_acc",))
    return out


# ------------------------------------------------------------------ e07
def b_e07(d):
    out, P, S = [], "Eseven", "e07"
    top = d["max_budget"]
    # seed count of the studies that stay on Stage 1 (E1-E4 may be on Stage 2, where EoneNSeeds = 20)
    out.append(Macro(P + "NSeeds", S, ("n_seeds",), "int", ("n_seeds",)))
    for key, c in d["cells"].items():
        v = f"Seed{tag(c['seed_size'])}Batch{tag(c['batch'])}"
        out += curve_macros(S, P, v, ("cells", key, "rows"), [top], ("acc", "bal_acc"))
    return out


# ------------------------------------------------------------------ e08
def b_e08(d):
    out, P, S = [], "Eeight", "e08"
    sd = ("n_seeds",)
    out += [Macro(P + "NScored", S, ("n_scored",), "int", sd),
            Macro(P + "NBad", S, ("n_bad",), "int", sd),
            Macro(P + "NGood", S, ("n_good",), "int", sd),
            Macro(P + "Budget", S, ("budget",), "int", sd)]
    for row in d["thresholds"]:
        t = "P" + tag(round(row["p_accept"] * 100))
        base = ("thresholds", ("p_accept", row["p_accept"]))
        for k, nm in (("precision", "Precision"), ("contamination", "Contam"),
                      ("bad_recall", "BadRecall"), ("good_retention", "GoodRet"),
                      ("fraction_kept", "Kept")):
            out += _stat(f"{P}{nm}{t}", S, base + (k,), "pct", sd,
                         nd=k in ("precision", "contamination"))  # empty kept set: n/a
    return out


# ------------------------------------------------------------------ e09
def b_e09(d):
    out, P, S = [], "Enine", "e09"
    out += [Macro(P + "NShared", S, ("n_shared",), "int", None),
            Macro(P + "NSharedBad", S, ("n_shared_bad",), "int", None)]
    for f in d["filters"]:
        t = tag(f).capitalize() if isinstance(f, str) else f
        for k, nm, kind in (("bad_recall", "BadRecall", "pct"), ("good_kept", "GoodKept", "pct"),
                            ("n_bad_flagged", "BadFlagged", "int"), ("n_good_flagged", "GoodFlagged", "int")):
            out.append(Macro(f"{P}Filter{t}{nm}", S, ("filters", f, k), kind, None))
    for k, nm, kind in (("bad_recall", "BadRecall", "pct"), ("good_kept", "GoodKept", "pct"),
                        ("n_bad_flagged", "BadFlagged", "int"), ("n_good_flagged", "GoodFlagged", "int")):
        out.append(Macro(f"{P}Tree{nm}", S, ("tree", "pooled", k), kind, None))
    return out


# ------------------------------------------------------------------ e10
def b_e10(d):
    out, P, S = [], "Eten", "e10"
    sd = ("seeds",)
    out += [Macro(P + "NSignals", S, ("n_signals",), "int", sd),
            Macro(P + "NGood", S, ("n_good",), "int", sd)]
    for pn, pv in d["panels"].items():
        t = "Fod" + tag(int(pn[3:]))
        for k, nm, kind in (("n_signals", "NSignals", "int"), ("n_good", "NGood", "int"),
                            ("pooled_good_fraction", "PooledGood", "pct"),
                            ("mean_good_fraction", "MeanGood", "pct"),
                            ("n_cycles", "NCycles", "int"),
                            ("n_cycles_below_50", "NBelowHalf", "int")):
            out.append(Macro(f"{P}{nm}{t}", S, ("panels", pn, k), kind, sd))
    return out


def b_e10_famb(d):
    out, S = [], "e10_famb"
    for row in d["rows"]:
        b = row["budget"]
        out += _stat(f"EtenFamb{bt(b)}", S, ("rows", ("budget", b), "over_seeds"), "pct",
                     ("seeds",))
    return out


# ------------------------------------------------------------------ e11
def b_e11(d):
    out, P, S = [], "Eeleven", "e11"
    sd = ("e10_scores_seeds",)
    for k, nm in (("n_universe", "NUniverse"), ("n_labelled_in_universe", "NLabelled"),
                  ("n_removed_labelled_bad", "NRemovedLabelled"),
                  ("n_removed_unlabelled_low_p", "NRemovedModel"), ("n_curated", "NCurated")):
        out.append(Macro(P + nm, S, ("composition_of_curated_set", k), "int", sd))
    groups = [("Full", ("panels",)), ("Common", ("common_cycle", "panels")),
              ("LabelsFull", ("labels_only", "full")), ("LabelsCommon", ("labels_only", "common_cycle"))]
    for gname, root in groups:
        node = d
        for tkn in root:
            node = node[tkn]
        for pn, pv in node.items():
            if not pn.startswith("FOD"):
                continue
            t = P + gname + "Fod" + tag(int(pn[3:]))
            base = root + (pn,)
            for cond in ("curated", "uncurated"):
                if cond in pv:
                    cn = cond.capitalize()
                    for k, nm in (("monotonicity", "Mono"), ("trend_mean", "TrendMean"),
                                  ("trend_min", "TrendMin")):
                        out.append(Macro(f"{t}{nm}{cn}", S, base + (cond, k), "f3", sd))
            for k, nm in (("monotonicity", "MonoDiff"), ("trend_mean", "TrendMeanDiff"),
                          ("panel_di_abs_spearman", "PanelDiDiff")):
                out.append(Macro(f"{t}{nm}", S, base + ("difference", k), "f3", sd))
            pp = pv.get("path_paired_monotonicity")
            if pp:
                out.append(Macro(f"{t}PathMonoDiff", S, base + ("path_paired_monotonicity", "mean_diff"),
                                 "f3", sd))
                ci = base + ("path_paired_monotonicity", "ci95")
                out.append(Macro(f"{t}PathMonoDiffCi", S, ci, "f3", sd, "ci", (ci + (0,), ci + (1,))))
            if "coverage" in pv:
                out.append(Macro(f"{t}Retained", S, base + ("coverage", "path_cycle_fraction_retained"),
                                 "pct", sd))
    return out


# ------------------------------------------------------------------ e14
def b_e14(d):
    out, P, S = [], "Efourteen", "e14"
    for k, nm in (("n_L_AL", "NLal"), ("n_L_rand", "NLrand"), ("n_shared", "NShared")):
        out.append(Macro(P + nm, S, (k,), "int", None))
    for k, nm in (("n_seed", "NSeed"), ("n_added", "NAdded"), ("n_total", "NTotal"),
                  ("n_seed_relabelled_in_L_AL", "NSeedRelabelled")):
        out.append(Macro(P + nm, S, ("seed_and_added", k), "int", None))
    out.append(Macro(P + "FirstDay", S, ("seed_and_added", "first_day"), "text", None))
    for pool, nm in (("L_AL", "Al"), ("L_rand", "Rand")):
        for i, day in enumerate(d["labels_per_day"][pool]):
            out.append(Macro(f"{P}Day{RANK_N(i)}{nm}N", S, ("labels_per_day", pool, day), "int", None))
    r = d["revisions"]
    out += [Macro(P + "NRevised", S, ("revisions", "n_revised"), "int", None),
            Macro(P + "RevisionRate", S, ("revisions", "revision_rate"), "pct", None),
            Macro(P + "RevGoodToBad", S, ("revisions", "direction_in_time", "GOOD->BAD"), "int", None),
            Macro(P + "RevBadToGood", S, ("revisions", "direction_in_time", "BAD->GOOD"), "int", None),
            Macro(P + "RevPoolGoodToBad", S, ("revisions", "direction_rand_to_AL", "GOOD->BAD"), "int", None),
            Macro(P + "RevPoolBadToGood", S, ("revisions", "direction_rand_to_AL", "BAD->GOOD"), "int", None),
            Macro(P + "RevEarlierRand", S, ("revisions", "earlier_pool", "L_rand"), "int", None),
            Macro(P + "RevEarlierAl", S, ("revisions", "earlier_pool", "L_AL"), "int", None)]
    for i, _ in enumerate(d["cross_check_famb"]["increments"]):        # live batch sizes
        out.append(Macro(f"{P}Batch{RANK_N(i)}", S, ("cross_check_famb", "increments", i), "int", None))
    for it in d["famb_trajectory"]["iterations"]:
        i = it["iteration"]
        base = ("famb_trajectory", "iterations", ("iteration", i))
        out += [Macro(f"{P}FambIter{RANK_N(i - 1)}NLabels", S, base + ("n_labels",), "int", None),
                Macro(f"{P}FambIter{RANK_N(i - 1)}Pool", S, base + ("f_amb_pool_12504",), "pct", None)]
    return out


def RANK_N(i):
    return tag(i + 1)


# ------------------------------------------------------------------ e13 (labelling throughput; Task 17b)
def b_e13(d):
    out, P, S = [], "Ethirteen", "e13"
    for pool, nm in (("L_AL", "Al"), ("L_rand", "Rand")):
        base = ("pools", pool, "overall")
        for k, t, kind in (("n_labels", "NLabels", "int"), ("n_breaks", "NBreaks", "int"),
                           ("active_s", "ActiveSec", "int"), ("median_s", "MedianSec", "sec"),
                           ("q25_s", "QuarterLoSec", "sec"), ("q75_s", "QuarterHiSec", "sec")):
            out.append(Macro(f"{P}{t}{nm}", S, base + (k,), kind, None))
    out += [Macro(P + "NExcludedSeedRand", S, ("pools", "L_rand", "n_excluded_seed_rows"), "int", None),
            Macro(P + "NUsedRand", S, ("pools", "L_rand", "n_used"), "int", None)]
    return out


# ------------------------------------------------------------------ E1 standard shift (R36) and pools
SHIFT_TAG = {"seed_L_AL": "SeedAl", "seed_L_rand": "SeedRand",
             "added_L_AL": "AddedAl", "added_L_rand": "AddedRand"}


def b_e01_shift(d):
    out, P, S = [], "EoneShift", "e01_shift"
    for pool, t in (("L_AL", "Al"), ("L_rand", "Rand")):
        out += [Macro(f"EzeroNGood{t}", S, ("pools", pool, "n_good"), "int", None),
                Macro(f"EzeroNBad{t}", S, ("pools", pool, "n_bad"), "int", None)]
    out.append(Macro("EzeroNSharedOutsideSeed", S, ("n_shared_outside_seed",), "int", None))
    for g, gv in d["groups"].items():
        t = P + SHIFT_TAG[g]
        base = ("groups", g)
        for k, nm, kind in (("n", "N", "int"), ("n_bad", "Bad", "int"), ("n_good", "Good", "int"),
                            ("bad_fraction", "BadPct", "pct")):
            out.append(Macro(f"{t}{nm}", S, base + (k,), kind, None))
        out.append(Macro(f"{t}CycleZeroPct", S, base + ("cycle_zero", "share"), "pct", None))
        for r in gv["receiver"]:
            out.append(Macro(f"{t}Recv{tag(int(r))}Pct", S, base + ("receiver", r, "share"), "pct", None))
        for pn in gv["per_panel"]:
            q = f"{t}Fod{tag(int(pn[3:]))}"
            out += [Macro(f"{q}N", S, base + ("per_panel", pn, "n"), "int", None),
                    Macro(f"{q}Bad", S, base + ("per_panel", pn, "n_bad"), "int", None)]
        for k, nm in (("n_scored", "NScored"), ("n_bad_called_good", "BadCalledGood"),
                      ("n_good_called_bad", "GoodCalledBad"), ("n_called_bad", "CalledBad"),
                      ("n_in_L_AL_training", "InTrain")):
            out.append(Macro(f"{t}{nm}", S, base + ("model", k), "int", None))
    return out


def b_pool_sizes(d):
    out, S = [], "pool_sizes"
    for k, nm in (("len_train_pool", "TrainPool"), ("len_es", "EarlyStop"), ("n_test", "TestFold")):
        for e in ("min", "max"):
            out.append(Macro(f"Ezero{nm}{e.capitalize()}", S, (k, e), "int", ("seeds",)))
    return out


# ------------------------------------------------------------------ fold-0 clean slice
def b_fold0(d):
    """d = fold0.fold0_study(...): models/<m>/rows[budget] with metric stat dicts."""
    out = []
    for m in d["models"]:
        out += curve_macros("fold0_" + d["of"], d["prefix"], MODEL_TAG.get(m, m), ("models", m, "rows"),
                            [d["budgets"][-1]], ("acc", "bal_acc", "auroc"))
    return out


def _load_f0_e04(cfg, results_dir):
    from .fold0 import load_e04
    return load_e04(cfg, results_dir)


def _load_f0_e05(cfg, results_dir):
    from .fold0 import load_e05
    return load_e05(cfg, results_dir)


STUDIES = [
    Study("e00", "e00_inputs.json", "Ezero", "none", builder=b_e00),
    Study("e01", "e01_selection.json", "Eone", "cells", builder=b_e01, stage2=True),
    Study("e02", "e02_reinspection.json", "Etwo", "cells", builder=b_e02, stage2=True),
    Study("e03", "e03_noise.json", "Ethree", "cells", builder=b_e03, stage2=True),
    Study("e04_frozen", "e04_frozen_configs.json", "EfourFrozen", "cells", kind="tune",
          builder=b_e04_frozen),
    Study("e04_compare", "e04_compare.json", "Efour", "cells", builder=b_e04_compare, stage2=True),
    Study("e05", "e05_arch_loop.json", "Efive", "cells", builder=b_e05),
    Study("e06", "e06_ablations.json", "Esix", "cells", builder=b_e06),
    Study("e07", "e07_loop_settings.json", "Eseven", "cells", builder=b_e07),
    Study("e08", "e08_thresholds.json", "Eeight", "cells", builder=b_e08),
    Study("e09", "e09_rules.json", "Enine", "static", builder=b_e09),
    Study("e10", "e10_coverage.json", "Eten", "static", builder=b_e10),
    Study("e10_famb", "e10_famb.json", "Eten", "cells", builder=b_e10_famb),
    Study("e11", "e11_downstream.json", "Eeleven", "static", builder=b_e11),
    Study("e13", "e13_throughput.json", "Ethirteen", "static", builder=b_e13),
    Study("e14", "e14_live_record.json", "Efourteen", "static", builder=b_e14),
    Study("e01_shift", "e01_standard_shift.json", "Eone", "static", builder=b_e01_shift),
    Study("pool_sizes", "e00_pool_sizes.json", "Ezero", "cells", builder=b_pool_sizes),
    Study("fold0_e04", None, "EfourFoldZero", "cells", builder=b_fold0, loader=_load_f0_e04),
    Study("fold0_e05", None, "EfiveFoldZero", "cells", builder=b_fold0, loader=_load_f0_e05),
]
BY_KEY = {s.key: s for s in STUDIES}
