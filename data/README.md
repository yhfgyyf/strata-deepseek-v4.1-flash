# Distributed metadata

- `models-1017.json`: approved original main/draft/vision files, SHA256 and verified mirror identities.
- `draft-input-vocab-114801.bin`, `draft-vocab-90000.bin`: sorted little-endian int32 tokenizer ID-selection maps. These are static metadata, not model weights or tokens from requests. `draft-vocab-provenance.json` records the tokenizer and mapping hashes.
- `performance-20261023.csv`, `performance-summary-20261023.json`: 12 formal UD-Q2_K_XL length-test measurements and source-summary provenance; no private prompts, outputs or traces.
- `performance-20261019.csv`, `performance-summary.json`: retained historical measurements for the previous model/configuration, not current Unsloth speed claims.
- `dspark-costs-rtxpro6000.txt`: historical draft/verify calibration. Its first scalar is the 5-token draft cost; the next eight scalars are costs for 1–8 verify rows. The starter uses at most six rows; costs are hardware/residency dependent.
- `source-manifest.json`: current published source hashes and packaging exceptions.
- `publication-checks.json`: current build/CPU/static checks. `publication-checks-20261020.json` preserves the earlier release receipt.

Weights, tokenizer text, calibrated expert rankings/heat, private configuration, request token streams, raw requests and traces are excluded. The launcher can generate an uncalibrated ranking and zero heat locally.
