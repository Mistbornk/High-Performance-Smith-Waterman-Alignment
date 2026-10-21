# IO 與 Benchmark

目標：在同一份輸入、同一套計時契約下比較 CUDA、多執行緒 striped SIMD、
多執行緒 scalar SW。三者都透過 `sw::BatchAligner` 執行，benchmark 不知道 backend 細節。

```text
gen_dataset.py ──> reference.fa / query.fa / meta.json
                         │
                sw::io::read_sequences ──> sw::io::SequenceSet（連續 buffer + offsets）
                         │
                sw::Workload（zip | all_vs_all，task 即時計算、不展開）
                         │
                sw::make_batch_aligner(backend, scoring, {threads, traceback})
                   ├─ striped / scalar：ThreadedBatchAligner（每 thread 一個 Aligner）
                   └─ cuda：detail::make_cuda_batch（待實作）
                         │
sw-bench ──> 一列 CSV ──> run_benchmarks.py 匯整 results.csv + metadata.json
```

## IO（`include/sw/io.hpp`）

- `read_sequences(path | istream, ReadOptions)`：FASTA（多行）或 FASTQ，依第一筆 header 判斷。
  CRLF、空行、header 註解都可處理；名稱取到第一個空白為止；空序列保留。
- `ReadOptions::unknown`：`reject`（預設，只接受 A/C/G/T/N）或 `as_n`（IUPAC 等字母轉 N）。
  錯誤訊息包含檔名與行號。
- `SequenceSet`：所有序列編碼後放在同一塊 `data()`，第 i 筆是
  `[offsets()[i], offsets()[i+1])`。這和 CUDASW4 `sequence_batch` 的 chars/offsets 相同，
  CUDA backend 可以一次 H2D 上傳整個集合，不需重新打包。
- `write_fasta`：給工具或測試輸出用。

## Workload（`include/sw/workload.hpp`）

- `Pairing::zip`：第 i 個 reference 對第 i 個 query，兩邊數量必須相同。
- `Pairing::all_vs_all`：每個 query 對每個 reference，query-major
  （task i = ref `i % R`、query `i / R`），對應 CUDASW4 的 query vs database 搜尋。
  task 用算的，不會為 R×Q 配置記憶體；CUDA backend 可以讀 `pairing()` 走專用路徑。
- `cells()` = Σ|ref|×|query|，GCUPS 的分子。

## BatchAligner（`include/sw/batch.hpp`）

- `align(workload, results)` 是同步呼叫：返回時所有 thread / kernel / D2H 都已結束。
  `results[i]` 對應 `workload.task(i)`；`results` 的容量會重用。
- `BatchOptions::threads`：CPU worker 數。每個 worker 在建構時建立自己的 `Aligner`
  （含 trace scratch）；thread 在每次 `align()` 建立並 join，呼叫端 thread 當 worker 0。
  task 以 16 筆為一個 chunk 動態分配。
- `BatchOptions::traceback = false`：score-only 模式（CUDASW4 只算分數），CPU backend 改呼叫
  `Aligner::score()`，結果的 CIGAR 為空。不支援的 backend 必須丟 `invalid_argument`，
  不可偷偷照算 CIGAR，以免 benchmark 欄位失真。
  - striped：kernel 的 `TRACEBACK` 樣板參數關閉時不配置、不寫 trace，也不做 backtrace。
  - scalar：rolling row，O(m) 記憶體。
- 不可用的 backend 丟 `runtime_error`，不 fallback。

## sw-bench

```sh
build/sw-bench --reference R.fa --query Q.fa --backend striped --threads 16 \
  [--pairing zip|all_vs_all] [--scoring 1,-4,-6,-1] [--score-only] \
  [--warmup 1] [--repetitions 5] [--results out.tsv] [--label TEXT] [--no-header]
```

計時範圍：

| 欄位 | 內容 |
| --- | --- |
| `load_s` | 讀檔、編碼、建立 Workload |
| `setup_s` | `make_batch_aligner`（CUDA 日後的 device 配置、上傳 DB 也算在這裡） |
| `min_s/median_s/max_s` | 每次 `align()` 的 wall time（warmup 不計） |
| `gcups_median/best` | `cells / 秒 / 1e9` |
| `score_checksum` | 依 task 順序對分數做 FNV-1a；只比分數，因為同分時各 backend 可選不同 CIGAR |

backend 不可用時 exit 77，參數錯誤時 exit 1。`--results` 輸出最後一次重複的
`task/reference/query/score/cigar` TSV（score-only 時 cigar 為 `*`），用來逐筆比對。

## 腳本

```sh
python3 scripts/gen_dataset.py --out data/len512 --pairs 2000 --length 512
python3 scripts/gen_dataset.py --out data/mixed  --pairs 5000 --length 100 --max-length 2000 \
  --substitution 0.1 --indel 0.02 --seed 1

python3 scripts/run_benchmarks.py --bench build/sw-bench \
  --dataset data/len512 --dataset data/mixed \
  --backends striped,scalar,cuda --threads 1,2,4,8,16,32 --repetitions 5 \
  --out results/$(date +%Y%m%d-%H%M%S)
```

- `gen_dataset.py`：query 由同 index 的 reference 加上 substitution 與短 indel 產生；
  固定 seed 可重現；輸出目錄已存在時拒絕寫入。`meta.json` 記錄參數與 zip cells。
- `run_benchmarks.py`：掃 dataset × backend × threads（cuda 只跑 threads=1）。
  輸出 `results.csv`、`metadata.json`（git commit/dirty、CPU、GPU、CMake 設定、每個指令）
  與 `logs/`。exit 77 記為 skipped；同一 dataset 各 backend 的 `score_checksum`
  不一致時列為 `score_mismatches` 並以非 0 結束。

## 接 CUDA 時要做的事

1. 在 `src/backends/cuda/` 實作 `make_cuda_batch`，回傳自己的 `BatchAligner`。
   要求 `threads == 1`；`traceback=false` 是主要路徑。
2. device 配置與（all_vs_all 時）database 上傳放在建構或第一次 `align()`；
   要在報告中註明是否計入計時。`align()` 返回前必須同步。
3. 直接使用 `SequenceSet::data()/offsets()` 上傳，不要再逐筆複製。
4. 用 `--results` 或 `score_checksum` 對 striped 驗證分數。

公平比較：CUDA 只算分數時，CPU 端用 `--score-only` 對比；CSV 的 `traceback` 欄位用來避免混比。

## Backend 一覽

| backend | 實作 | traceback | score-only | 計分限制 |
| --- | --- | --- | --- | --- |
| striped | SSE4.1 striped，8-bit 差值 | ✓ | ✓ | byte 差值範圍（見 `validate_scoring`） |
| scalar | naive Gotoh，逐 cell、32-bit | ✓ | ✓ | `gap_open <= gap_extend <= 0`、int32 範圍 |
| cuda | 待實作 | | | |

## 多執行緒注意事項

在容器裡，`nproc` 可能大於實際可用 CPU。`run_benchmarks.py` 會把 cgroup quota
（`/sys/fs/cgroup/cpu.max`）與 load average 寫進 `metadata.json`；thread 數超過 quota 時會警告。
例如開發機有 32 個 logical CPU，但 quota 只有 8，超過 8 threads 不會再變快。
