#!/usr/bin/env python3
"""Generate a reproducible benchmark dataset: reference.fa, query.fa, meta.json.

Each query is derived from the reference with the same index (for --pairing zip),
mutated by substitutions and short indels. The same files also serve all_vs_all
runs, where every query is aligned to every reference. Existing directories are
never overwritten.
"""
import argparse
import json
import pathlib
import random

BASES = "ACGT"


def mutate(rng, ref, substitution, indel, max_indel):
    out = []
    i = 0
    while i < len(ref):
        r = rng.random()
        if r < indel / 2:  # deletion from the reference
            i += rng.randint(1, max_indel)
            continue
        if r < indel:  # insertion into the query
            out.extend(rng.choices(BASES, k=rng.randint(1, max_indel)))
        base = ref[i]
        if rng.random() < substitution:
            base = rng.choice([b for b in BASES if b != base])
        out.append(base)
        i += 1
    return "".join(out)


def write_fasta(path, prefix, sequences, width=80):
    with path.open("x") as f:
        for i, seq in enumerate(sequences):
            f.write(f">{prefix}{i}\n")
            for start in range(0, len(seq), width):
                f.write(seq[start:start + width] + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=pathlib.Path, required=True, help="new directory to create")
    parser.add_argument("--pairs", type=int, required=True, help="number of reference/query pairs")
    parser.add_argument("--length", type=int, required=True, help="reference length (or minimum)")
    parser.add_argument("--max-length", type=int, help="maximum reference length (uniform); default --length")
    parser.add_argument("--substitution", type=float, default=0.05, help="per-base substitution rate")
    parser.add_argument("--indel", type=float, default=0.01, help="per-base indel event rate")
    parser.add_argument("--max-indel", type=int, default=5, help="maximum indel length")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    max_length = args.max_length or args.length
    if args.pairs < 1 or args.length < 1 or max_length < args.length:
        parser.error("need --pairs >= 1 and 1 <= --length <= --max-length")
    if not (0 <= args.substitution <= 1 and 0 <= args.indel <= 1 and args.max_indel >= 1):
        parser.error("rates must be in [0, 1] and --max-indel >= 1")

    rng = random.Random(args.seed)
    refs, queries = [], []
    for _ in range(args.pairs):
        ref = "".join(rng.choices(BASES, k=rng.randint(args.length, max_length)))
        refs.append(ref)
        queries.append(mutate(rng, ref, args.substitution, args.indel, args.max_indel))

    args.out.mkdir(parents=True, exist_ok=False)
    write_fasta(args.out / "reference.fa", "ref", refs)
    write_fasta(args.out / "query.fa", "qry", queries)
    meta = {
        "generator": "gen_dataset.py",
        "seed": args.seed,
        "pairs": args.pairs,
        "length": [args.length, max_length],
        "substitution": args.substitution,
        "indel": args.indel,
        "max_indel": args.max_indel,
        "reference_bases": sum(map(len, refs)),
        "query_bases": sum(map(len, queries)),
        "zip_cells": sum(len(r) * len(q) for r, q in zip(refs, queries)),
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {args.out}: {args.pairs} pairs, {meta['zip_cells']:,} zip cells")


if __name__ == "__main__":
    main()
