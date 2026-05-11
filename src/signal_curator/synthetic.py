"""Synthetic 1D-signal generator for the demo dataset.

Generates realistic-looking GOOD + 4 BAD classes mirroring the signal
gallery in Fig. 2 of the paper:

  GOOD       : clean wave-packet with the expected envelope
  BAD class A: low-energy (debonded sensor or trigger jitter)
  BAD class B: saturated / high-amplitude (drive overdrive)
  BAD class C: noisy / glitchy (high-frequency contamination)
  BAD class D: ramp-up with mean drift (start at low amplitude, grow,
               stabilise at non-zero mean, wavy tail)

Saves to the generic HDF5 schema:
    /<signal_id>/data           (L_full,) float32
    /<signal_id>.attrs.freq_khz int
"""
from __future__ import annotations
import numpy as np
import h5py


def _wave_packet(t: np.ndarray, f_hz: float, t0: float, sigma: float, amp: float) -> np.ndarray:
    """Gaussian-windowed sinusoid -- the building block of a 'clean' guided wave."""
    envelope = np.exp(-0.5 * ((t - t0) / sigma) ** 2)
    return amp * envelope * np.sin(2 * np.pi * f_hz * t)


def make_good(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Clean wave-packet at ~50 kHz centre, slight reflections."""
    sig = _wave_packet(t, f_hz=50_000, t0=2e-4, sigma=2.5e-5, amp=0.05)
    sig += _wave_packet(t, f_hz=50_000, t0=3.5e-4, sigma=3e-5, amp=0.02)
    sig += rng.normal(0, 0.0008, t.size).astype(np.float32)
    return sig.astype(np.float32)


def make_bad_low_energy(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Near-flat low-energy signal -- debonded sensor case."""
    sig = rng.normal(0, 0.0002, t.size).astype(np.float32)
    return sig


def make_bad_saturated(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """High-amplitude, clipped-looking signal -- amplifier saturation."""
    sig = _wave_packet(t, f_hz=50_000, t0=2e-4, sigma=2.5e-5, amp=1.5)
    sig = np.clip(sig, -1.0, 1.0)  # clipping
    sig += rng.normal(0, 0.05, t.size).astype(np.float32)
    return sig.astype(np.float32)


def make_bad_noisy(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Glitchy, high-frequency-contaminated signal."""
    base = _wave_packet(t, f_hz=50_000, t0=2e-4, sigma=2.5e-5, amp=0.03)
    hf = 0.04 * np.sin(2 * np.pi * 200_000 * t).astype(np.float32)
    spikes = np.zeros_like(t, dtype=np.float32)
    n_spikes = rng.integers(5, 15)
    for _ in range(n_spikes):
        i = rng.integers(0, len(t))
        spikes[i] = rng.choice([-1, 1]) * rng.uniform(0.05, 0.2)
    return (base + hf + spikes + rng.normal(0, 0.005, t.size)).astype(np.float32)


def make_bad_drift(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Ramp-up signal: starts low, rises linearly, stabilises at non-zero mean, wavy tail."""
    L = len(t)
    sig = np.zeros(L, dtype=np.float32)
    ramp_end = int(0.3 * L)
    plateau_end = int(0.7 * L)
    sig[:ramp_end] = np.linspace(0, 0.05, ramp_end)
    sig[ramp_end:plateau_end] = 0.05 + rng.normal(0, 0.002, plateau_end - ramp_end)
    sig[plateau_end:] = 0.05 + 0.01 * np.sin(2 * np.pi * 5_000 * t[plateau_end:])
    sig += rng.normal(0, 0.001, L).astype(np.float32)
    return sig.astype(np.float32)


CLASSES = {
    "good":       (make_good,            "GOOD"),
    "bad_low":    (make_bad_low_energy,  "BAD"),
    "bad_sat":    (make_bad_saturated,   "BAD"),
    "bad_noise":  (make_bad_noisy,       "BAD"),
    "bad_drift":  (make_bad_drift,       "BAD"),
}


def generate_demo_h5(out_path: str, n_per_class: int = 20, seed: int = 0) -> None:
    """Generate a synthetic demo HDF5 with the generic flat schema.

    Default: 20 signals per class x 5 classes = 100 signals total.
    """
    rng = np.random.default_rng(seed)
    fs = 1_000_000  # 1 MHz acquisition (matches paper)
    n_samples = 500
    t = np.arange(n_samples) / fs

    with h5py.File(out_path, "w") as f:
        i = 0
        for class_name, (make_fn, _gold_label) in CLASSES.items():
            for _ in range(n_per_class):
                sig = make_fn(t, rng)
                grp = f.create_group(f"sig_{i:04d}_{class_name}")
                grp.create_dataset("data", data=sig)
                grp.attrs["freq_khz"] = 50
                grp.attrs["panel"] = "demo"
                grp.attrs["class_truth"] = class_name  # for evaluation, not used at labelling time
                i += 1
    print(f"Wrote {i} synthetic demo signals to {out_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/demo/synthetic_demo.h5")
    p.add_argument("--n-per-class", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    generate_demo_h5(args.out, n_per_class=args.n_per_class, seed=args.seed)
