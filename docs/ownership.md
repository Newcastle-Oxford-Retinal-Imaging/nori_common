# NORI contract and implementation ownership

The schema layer is split into contracts and implementation so product teams can
own their data contracts without making `nori_common` the semantic owner of
every pipeline stage.

## Layers

| Layer | Owns | Must not depend on |
|---|---|---|
| `nori_common.contracts` | protocols, Pydantic metadata models, value objects, contract errors | `nori_common.adapters`, `nori_common.fits`, Astropy |
| `nori_common.adapters` | FITS/container/in-memory implementations, legacy layout adapters | stage code |
| `nori_common.fits` | low-level FITS I/O and `norifmt` constants | contracts/adapters |
| `nori_common.manifest` | manifest extension and sidecar mechanics | stage code |
| `nori_common.validation` | array-aware consistency validation | `nori_common.adapters`, `nori_common.fits`, Astropy |
| `nori_refactor` | stage logic, orchestration, stage policy | `nori_common` directly (use `nori.utils.fs`) |

## Product ownership

| Contract | Producer | Primary owner | Consumers |
|---|---|---|---|
| `contracts/raw.py` | acquisition / process | acquisition/process maintainers | `process`, ERICA bridge |
| `contracts/trial.py` | process | process maintainers | motion estimation, make-trace, montage removal |
| `contracts/eye_motion.py` (future) | motion estimation | motion-estimation maintainers | make-trace, remove-motion |
| `contracts/metrics.py` (future) | motion estimation | motion-estimation maintainers | make-trace, montage removal |
| `contracts/trace.py` (future) | make-trace | make-trace maintainers | downstream analysis |
| `contracts/protocols.py` | shared | library maintainers | all consumers |
| `contracts/manifest.py` | shared | library maintainers | all producers/consumers |
| `adapters/`, `fits/`, `manifest/` | shared | `nori_common` maintainers | all consumers |

## Public import boundary

- `nori_refactor` imports only through `nori.utils.fs`.
- `nori_common.data` remains a compatibility facade during migration.
- New code should import contracts from `nori_common.contracts` and
  implementations from `nori_common.adapters`.
