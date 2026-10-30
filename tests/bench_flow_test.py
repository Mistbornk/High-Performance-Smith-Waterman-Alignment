#!/usr/bin/env python3
"""End-to-end check of gen_dataset.py -> sw-bench -> run_benchmarks.py on tiny inputs."""
import csv
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

BENCH = sys.argv.pop(1)
SCRIPTS = pathlib.Path(sys.argv.pop(1))


def run(*cmd, check=True):
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True, check=check)


class BenchFlow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="sw-bench-flow-")
        cls.root = pathlib.Path(cls.tmp.name)
        cls.data = cls.root / "tiny"
        run(sys.executable, SCRIPTS / "gen_dataset.py", "--out", cls.data, "--pairs", "40",
            "--length", "20", "--max-length", "80", "--seed", "7")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_generator_is_reproducible_and_protective(self):
        other = self.root / "again"
        run(sys.executable, SCRIPTS / "gen_dataset.py", "--out", other, "--pairs", "40",
            "--length", "20", "--max-length", "80", "--seed", "7")
        for name in ("reference.fa", "query.fa", "meta.json"):
            self.assertEqual((self.data / name).read_bytes(), (other / name).read_bytes())
        before = (self.data / "reference.fa").read_bytes()
        result = run(sys.executable, SCRIPTS / "gen_dataset.py", "--out", self.data, "--pairs", "1",
                     "--length", "5", check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((self.data / "reference.fa").read_bytes(), before)

    def test_sw_bench_row_and_results(self):
        meta = json.loads((self.data / "meta.json").read_text())
        rows, tsvs = [], []
        for threads in (1, 3):
            tsv = self.root / f"results_t{threads}.tsv"
            out = run(BENCH, "--reference", self.data / "reference.fa", "--query", self.data / "query.fa",
                      "--threads", threads, "--repetitions", "2", "--results", tsv).stdout
            rows.append(next(csv.DictReader(out.splitlines())))
            tsvs.append(tsv.read_text())
        for row in rows:
            self.assertEqual(row["backend"], "striped")
            self.assertEqual(int(row["tasks"]), 40)
            self.assertEqual(int(row["cells"]), meta["zip_cells"])
            self.assertLessEqual(float(row["min_s"]), float(row["median_s"]))
            self.assertGreater(float(row["gcups_median"]), 0)
        self.assertEqual(rows[0]["score_checksum"], rows[1]["score_checksum"])
        self.assertEqual(tsvs[0], tsvs[1])
        self.assertEqual(len(tsvs[0].splitlines()), 41)

        all_row = next(csv.DictReader(run(
            BENCH, "--reference", self.data / "reference.fa", "--query", self.data / "query.fa",
            "--pairing", "all_vs_all", "--repetitions", "1").stdout.splitlines()))
        self.assertEqual(int(all_row["tasks"]), 1600)
        self.assertEqual(int(all_row["cells"]), meta["reference_bases"] * meta["query_bases"])

    def test_sw_bench_errors(self):
        ref, qry = self.data / "reference.fa", self.data / "query.fa"
        self.assertEqual(run(BENCH, "--reference", ref, "--query", qry, "--backend", "cuda",
                             check=False).returncode, 77)
        for extra in (["--threads", "0"], ["--repetitions", "0"],
                      ["--backend", "gpu"], ["--scoring", "1,2,3"], ["--bogus", "1"]):
            with self.subTest(extra=extra):
                result = run(BENCH, "--reference", ref, "--query", qry, *extra, check=False)
                self.assertEqual(result.returncode, 1)
                self.assertIn("sw-bench:", result.stderr)

    def test_score_only_matches_traceback(self):
        sums = set()
        for backend in ("striped", "scalar"):
            for extra in ([], ["--score-only"]):
                tsv = self.root / f"{backend}{len(extra)}.tsv"
                out = run(BENCH, "--reference", self.data / "reference.fa", "--query", self.data / "query.fa",
                          "--backend", backend, "--threads", "2", "--repetitions", "1",
                          "--results", tsv, *extra).stdout
                row = next(csv.DictReader(out.splitlines()))
                self.assertEqual(row["traceback"], "0" if extra else "1")
                sums.add(row["score_checksum"])
                if extra:
                    self.assertTrue(all(line.endswith("\t*") for line in tsv.read_text().splitlines()[1:]))
        # cuda is score-only; it joins the comparison when built and a GPU is present.
        cuda = run(BENCH, "--reference", self.data / "reference.fa", "--query", self.data / "query.fa",
                   "--backend", "cuda", "--repetitions", "1", "--score-only", check=False)
        if cuda.returncode != 77:
            self.assertEqual(cuda.returncode, 0, cuda.stderr)
            sums.add(next(csv.DictReader(cuda.stdout.splitlines()))["score_checksum"])
        self.assertEqual(len(sums), 1)

    def test_driver(self):
        out = self.root / "sweep"
        result = run(sys.executable, SCRIPTS / "run_benchmarks.py", "--bench", BENCH,
                     "--dataset", self.data, "--out", out, "--threads", "1,2",
                     "--repetitions", "1", "--warmup", "0", check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = list(csv.DictReader((out / "results.csv").open()))
        self.assertEqual([(r["backend"], r["threads"]) for r in rows],
                         [("striped", "1"), ("striped", "2"), ("scalar", "1"), ("scalar", "2")])
        meta = json.loads((out / "metadata.json").read_text())
        self.assertEqual(meta["skipped"], ["tiny/cuda/t1"])
        self.assertEqual(meta["score_mismatches"], [])
        self.assertNotEqual(run(sys.executable, SCRIPTS / "run_benchmarks.py", "--bench", BENCH,
                                "--dataset", self.data, "--out", out, check=False).returncode, 0)


if __name__ == "__main__":
    unittest.main()
