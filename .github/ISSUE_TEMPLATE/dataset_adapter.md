---
name: Dataset adapter request
about: You want to use signal-curator with a dataset format we don't support yet
title: '[adapter] '
labels: adapter, enhancement
---

**Dataset description**
- Domain: (guided waves / vibration / accel / ECG / audio / other)
- Approximate size: (signals, samples per signal, total MB)
- Public? Link if so.

**Current storage format**
What you have today -- file structure, format (HDF5/numpy/CSV/...), sample
sizes, frequency, any metadata.

**What's missing**
What stops you from using the `GenericAdapter` directly. E.g., "my signals
are stored as one .mat per cycle and there's no obvious way to give each
signal a stable signal_id."

**Conversion you've tried (if any)**
```python
# paste any conversion code you've experimented with
```
