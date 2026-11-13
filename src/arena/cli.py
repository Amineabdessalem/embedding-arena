"""``arena bench`` compares compressors on synthetic embeddings."""

from __future__ import annotations

import argparse

from arena.compress import binary, float32, int8, pca, truncate
from arena.data import clustered
from arena.evaluate import evaluate


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="arena", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench", help="recall vs memory for each compression method")
    b.add_argument("-n", type=int, default=20000, help="corpus size")
    b.add_argument("--dim", type=int, default=384)
    b.add_argument("--queries", type=int, default=300)
    b.add_argument("-k", type=int, default=10)
    b.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    data = clustered(args.n + args.queries, args.dim, seed=args.seed)
    corpus, queries = data[: args.n], data[args.n:]
    d = args.dim
    methods = [int8(d), binary(d), pca(d, d // 4), pca(d, d // 8), truncate(d, d // 4), truncate(d, d // 8)]
    reports = evaluate(methods, corpus, queries, k=args.k)
    base_bytes = float32(d).bytes_per_vector
    print(f"{args.n} x {d} corpus, {args.queries} queries, recall@{args.k} vs exact float32\n")
    print(f"{'method':<14}{'bytes/vec':>10}{'vs float32':>12}{'recall':>9}{'qps':>10}")
    print(f"{'float32':<14}{base_bytes:>10}{'1.0x':>12}{1.0:>9.3f}{'':>10}")
    for r in reports:
        print(f"{r.name:<14}{r.bytes_per_vector:>10}{base_bytes / r.bytes_per_vector:>11.1f}x"
              f"{r.recall_at_k:>9.3f}{r.queries_per_second:>10.0f}")
    return 0




def main_entry() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    main_entry()
