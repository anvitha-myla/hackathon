Place IndPenSim files here (not committed).

Official 100-batch simulator dump (Goldrick et al., CC BY 4.0):
https://data.mendeley.com/datasets/pdnjz7zz5x/1

Typical names:
- 100_Batches_IndPenSim_V3.csv
- 100_Batches_IndPenSim_Statistics.csv

The dump is a **simulator reference**, not physical industrial telemetry.
Raman wavelength columns are optional; process-variable extracts are enough for ingestion.

Configure `raw_path` in `configs/data.yaml`. Unit tests use
`tests/fixtures/indpensim_synthetic_fixture.csv` (labeled synthetic rows,
published column names only).
