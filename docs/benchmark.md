# IO 與 Benchmark

目標：在同一份輸入、同一套計時契約下比較 CUDA、多執行緒 striped SIMD、
多執行緒 scalar SW。

```text
gen_dataset.py ──> reference.fa / query.fa / meta.json
                         │
                sw::io::read_sequences ──> sw::io::SequenceSet（連續 buffer + offsets）
                         │
                sw::Workload（zip | all_vs_all，task 即時計算、不展開）
                         │
sw-bench ── striped / scalar：sw::parallel_for 對每個 task 呼叫 <backend>::align 或 ::score
                         │
                一列 CSV ──> run_benchmarks.py 匯整 results.csv + metadata.json
```

## IO（`include/sw/io.hpp`）

- `read_sequences(path | istream, ReadOptions)`：FASTA（多行）或 FASTQ，依第一筆 header 判斷。
  CRLF、空行、header 註解都可處理；名稱取到第一個空白為止；空序列保留。
- `ReadOptions::unknown`：`reject`（預設，只接受 A/C/G/T/N）或 `as_n`（IUPAC 等字母轉 N）。
  錯誤訊息包含檔名與行號。
- `SequenceSet`：所有序列編碼後放在同一塊 `data()`，第 i 筆是
  `[offsets()[i], offsets()[i+1])`。這和 CUDASW4 `sequence_batch` 的 chars/offsets 相同，
  GPU 可以一次 H2D 上傳整個集合。
- `write_fasta`：給工具或測試輸出用。

## Workload（`include/sw/io.hpp`）

- `Pairing::zip`：第 i 個 reference 對第 i 個 query，兩邊數量必須相同。
- `Pairing::all_vs_all`：每個 query 對每個 reference，query-major
  （task i = ref `i % R`、query `i / R`），對應 CUDASW4 的 query vs database 搜尋。
  task 用算的，不會為 R×Q 配置記憶體。
- `cells()` = Σ|ref|×|query|，GCUPS 的分子。

## CPU 執行（`include/sw/sw.hpp`）

- 每個 backend 是一組 free function：`sw::<backend>::align(ref, query, scoring)` 與
  `::score(...)`。scratch 是 `thread_local`，多個 thread 可同時呼叫。
- `--score-only`：改呼叫 `score()`，不配置、不寫 trace，結果的 CIGAR 為空。
- `sw::parallel_for(n, threads, body)`：呼叫端 thread 也是 worker；task 以 16 筆為一個
  chunk 動態分配；第一個例外會停止剩下的工作並重新丟出。

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
| `setup_s` | CPU backend 為 0；GPU backend 的 device 配置與上傳算在這裡 |
| `min_s/median_s/max_s` | 每次跑完所有 task 的 wall time（warmup 不計） |
| `gcups_median/best` | `cells / 秒 / 1e9` |
| `score_checksum` | 依 task 順序對分數做 FNV-1a；只比分數，因為同分時各 backend 可選不同 CIGAR |

backend 沒有編進來時 exit 77，參數錯誤時 exit 1。`--results` 輸出最後一次重複的
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

## 公平比較

GPU kernel 通常只算分數；CPU 端要用 `--score-only` 對比，CSV 的 `traceback` 欄位用來避免混比。

## Backend 一覽

| backend | 實作 | traceback | score-only | 計分限制 |
| --- | --- | --- | --- | --- |
| striped | SSE4.1 striped，8-bit 差值 | ✓ | ✓ | byte 差值範圍（見 `src/striped.cpp` 的 `validate`） |
| scalar | naive Gotoh，逐 cell、32-bit | ✓ | ✓ | `gap_open <= gap_extend <= 0`、int32 範圍 |
| cuda | 待移植 | | | |

## 多執行緒注意事項

在容器裡，`nproc` 可能大於實際可用 CPU。`run_benchmarks.py` 會把 cgroup quota
（`/sys/fs/cgroup/cpu.max`）與 load average 寫進 `metadata.json`；thread 數超過 quota 時會警告。
例如開發機有 32 個 logical CPU，但 quota 只有 8，超過 8 threads 不會再變快。
