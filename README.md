# nori-common

Shared FITS I/O and `norifmt` schema utilities for AOSLO retinal imaging.

The package is deliberately small and format-explicit: `nori_common.fits`
provides a lossless FITS read/write layer with an Astropy-free boundary, and
`nori_common.fits.schema` describes the on-disk layout shared between the
processing pipeline and downstream consumers.

## Install

```bash
uv sync --dev
```

## Test

```bash
uv run pytest
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
```

## Inspect schema artifacts

```bash
uv run nori-schema-inspect path/to/file.fits
uv run nori-schema-inspect output/ --summary
```

The inspector supports current schema-aware files, legacy processed trials, raw acquisition FITS + YAML and unknown FITS files. See [docs/inspection.md](docs/inspection.md).
