# embedding-arena

[![CI](https://github.com/Amineabdessalem/embedding-arena/actions/workflows/ci.yml/badge.svg)](https://github.com/Amineabdessalem/embedding-arena/actions/workflows/ci.yml)

How far can you compress embeddings before search quality breaks? This project
measures **recall@k against exact float32 search** for the usual ways of
shrinking vectors, next to the bytes each one stores per vector.

| method | bytes / vector (384-d) | compression | recall@10 |
|---|---:|---:|---:|
| float32 (baseline) | 1536 | 1× | 1.000 |
| **int8** (per-dimension scalar quantisation) | 384 | 4× | **0.988** |
| PCA to 96 dims, float32 | 384 | 4× | 0.740 |
| PCA to 48 dims, float32 | 192 | 8× | 0.731 |
| truncate to first 96 dims | 384 | 4× | 0.496 |
| truncate to first 48 dims | 192 | 8× | 0.336 |
| **binary** (one bit per dimension) | 48 | 32× | 0.456 |

10,000 clustered 384-d vectors, 200 queries, k = 10, seed 0:
`PYTHONPATH=src python -m arena.cli bench -n 10000 --dim 384 --queries 200`.

## What it says

- **int8 is almost free.** Per-dimension scaling keeps 98.8% of the true top-10
  at a quarter of the memory. For most systems it is the right default.
- **Binary is a shortlist, not a search.** It is 32× smaller but keeps less than
  half of the true neighbours, so use it to pre-select 10 to 20 times k
  candidates and re-rank with int8 or float32.
- **PCA beats prefix truncation at equal size.** Real embeddings spread their
  variance unevenly, so keeping the principal directions loses far less than
  keeping the first coordinates. Truncation only works when the model was
  trained for it (Matryoshka embeddings), which these synthetic vectors are not.
- **Recall, not the proxy, decides.** Mean squared error of the reconstruction
  looked fine for PCA in earlier drafts; the ranking, which is what search uses,
  did not.

## Caveats

The data is synthetic: clusters on a 24-dimensional manifold with noise, so
the variance is anisotropic like real embeddings, but it is not a real model's
output. Run the same bench on a real corpus before trusting the numbers for a
product decision. The "vs float32" column is storage only; queries are scored
in each method's own space, so the qps column is a rough guide.

## Layout

```
src/arena/compress.py  int8, binary, PCA and truncation, each with encode and score
src/arena/evaluate.py  recall@k against exact search, timing
src/arena/data.py      synthetic clustered embeddings
src/arena/cli.py       `arena bench`
```

## Development

```bash
pip install -e ".[dev]"
pytest -q
ruff check src tests
```

## Roadmap

See [ROADMAP.md](ROADMAP.md).

## License

MIT
