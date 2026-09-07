# TemporalFocus.jl → NeuroPulse.jl

**This repository is no longer the canonical Julia package.**

[ADR 0002](docs/adr/0002-merge-temporalfocus-into-neuropulse.md) imports this
attention surface into [`rmems/NeuroPulse.jl`](https://github.com/rmems/NeuroPulse.jl).
Use that repository.

| | Survivor (`rmems/NeuroPulse.jl`) | This repository (provenance) |
|---|---|---|
| Package `name` | `TemporalFocus` | `TemporalFocus` (do not add alongside the survivor) |
| UUID | `b7e4c3f2-1d2e-4a5b-8c9d-0e1f2a3b4c5e` | `7f3c9f2a-6b2e-4d91-9c4f-1a2b3c4d5e6f` (**retired**) |
| Load | `Pkg.add(url="https://github.com/rmems/NeuroPulse.jl")` then `using TemporalFocus` | Do not depend on this UUID |

Import PR: [NeuroPulse.jl#45](https://github.com/rmems/NeuroPulse.jl/pull/45) (Closes [NeuroPulse.jl#43](https://github.com/rmems/NeuroPulse.jl/issues/43)).

**Do not** add this repo and NeuroPulse.jl to the same Julia environment — both
declare `name = "TemporalFocus"`. **Do not** archive this repo yet (owner action
later). **Do not** import NeuroPulse or SpikeStream here.

`experiments/` stays here as provenance. It was not imported into NeuroPulse.

---

# Historical README

[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://rmems.github.io/TemporalFocus.jl/dev/)

Pure spike-native temporal interaction primitives. The live package is
[NeuroPulse.jl](https://github.com/rmems/NeuroPulse.jl).

**Wiki:** GitHub wiki is enabled but not populated. When content exists, clone
`git@github.com:rmems/TemporalFocus.jl.wiki.git`.

## What this tree used to ship

- `SpikeEvent`, `SpikeTrain`, `TemporalBuffer`
- `spike_attention_discrete` / `spike_attention_temporal` / `spike_attention_continuous`
- `temporal_weight`, `prune!`, `normalize_l1!`, `normalize_max!`

Those names now live under `TemporalFocus.Attention` in NeuroPulse (re-exported
from the parent `TemporalFocus` module).

This tree does not own STDP, plasticity, routing (`ActivityRegion` /
`RegionRouter`), SpikeStream feature extraction, or finance/HFT semantics.

## Experiment Gallery (provenance)

Spike-native characterization experiments remain under
[`experiments/`](experiments/). They were **not** imported into NeuroPulse.

**→ [Experiment Gallery](https://rmems.github.io/TemporalFocus.jl/dev/experiments/)**

```bash
julia --project=experiments -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=experiments experiments/run_all.jl
```

See [`experiments/README.md`](experiments/README.md).

## Examples (this tree)

```bash
julia --project=. examples/discrete_attention.jl
julia --project=. examples/temporal_attention.jl
julia --project=. examples/continuous_buffer.jl
julia --project=. examples/normalize_readout.jl
```

## License

MIT OR Apache-2.0.
