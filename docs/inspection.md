# Inspecting NORI schema artifacts

`nori-schema-inspect` reports the schema, array shape and axis identity of NORI FITS files. It is read-only and works on current schema-aware files as well as older formats.

## Usage

```bash
uv run nori-schema-inspect path/to/file.fits
uv run nori-schema-inspect output/ --summary
uv run nori-schema-inspect output/ --json
uv run nori-schema-inspect legacy.fits --strict
uv run nori-schema-inspect output/ --html
```

## What it detects

- current schema-aware FITS files with `MANIFEST` and `NORIFMT_*` markers,
- schema declared only by an adjacent YAML sidecar,
- older/partial manifests, shown with inferred shape where needed,
- legacy processed trials with named or positional `FIT_PARAMS` / `LINE_STARTS` HDUs,
- legacy derived arrays inferred from filename suffixes such as `_EYEMOTION_PER_PIXEL`, `_METRICS` and `_TRACE`,
- raw acquisition FITS + YAML, including `metadata_version` 1, 2 and absent-version metadata,
- unknown FITS files, shown structurally rather than rejected.

## Validation

For schema-aware files it checks:

- schema markers and manifest presence,
- manifest schema name/version,
- array shape against the FITS data,
- axis sizes against the shape,
- categorical axis labels,
- sidecar/manifest consistency,
- trace `cols` length against the field axis.

Legacy and raw files produce warnings rather than hard failures unless `--strict` is used.

## Exit codes

- `0`: valid, or warnings only without `--strict`
- `1`: validation errors, or warnings with `--strict`
- `2`: usage or I/O error
