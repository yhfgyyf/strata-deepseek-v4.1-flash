# Distributed metadata

- `performance-20261019.csv`: all 18 timed requests from the two named A/B/A studies, excluding warmup; derived counts/times only.
- `performance-summary.json`: three-candidate arithmetic means and hashes of the original private summaries.
- `dspark-costs-rtxpro6000.txt`: historical draft/verify calibration. The first scalar is the 5-token draft cost; the following eight scalars are costs for 1–8 verify rows. The five-token starter uses at most six rows. This table is hardware and residency specific.
- `source-manifest.json`: published source hashes, generated during packaging.
- `publication-checks.json`: build/CPU/static checks for the publication snapshot.

Weights, tokenizer files, vocabulary maps, expert rankings, heat snapshots, local configuration, raw requests and traces are not distributed here.
