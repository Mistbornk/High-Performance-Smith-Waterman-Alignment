# Backend 擴充

應用程式只依賴 `include/sw/alignment.hpp`。Factory 在 `src/alignment.cpp`，
每個 backend 透過 `src/backend.hpp` 提供 availability 與 factory 函式。
公開型別沒有 CUDA 或 SIMD 標頭，也沒有 virtual device 方法。

```text
應用程式 → sw::Aligner → backend 實作
                          ├─ striped/backend.cpp → kernel.cpp → detail/（依賴 libsimdpp）
                          ├─ scalar/backend.cpp  （naive Gotoh）
                          └─ cuda/backend.cpp    （待填）
```

## Scalar

`src/backends/scalar/backend.cpp` 是 naive Gotoh global affine-gap DP（逐 cell、int32），
作為多執行緒 baseline。`align` 存一個 byte/cell 的 trace；`score` 只用 rolling row。
同分時 S 的來源優先序為 diagonal > insertion > deletion，與 striped backtrace 相同；
gap 同分時優先 open。要求 `gap_open <= gap_extend <= 0`，確保 CIGAR 合併相鄰 gap 後重播仍等於 DP 分數。
每個 backend 都要實作 `align` 與 `score`，並通過 `tests/support/check_alignment.hpp`
（oracle 分數、CIGAR 重播、score-only 一致）。

## 實作 CUDA

1. 在 `src/backends/cuda/` 加入 `.cu` kernel 與擁有 GPU buffer/stream 的 backend 類別。
2. 增加可選 CMake 開關，啟用時才 `enable_language(CUDA)`、查找 toolkit 與編譯 `.cu`。
   關閉時保留清楚的 unavailable 入口；一般 CPU 建置不應要求 CUDA。
3. availability 檢查編譯支援與可用 device；錯誤明確回報，不能悄悄改跑其他 backend。
4. 公開 `align` 接收 host 輸入，同步完成上傳、計算、下載，回傳擁有資料的結果。
   每個 instance 擁有 GPU 資源並以 RAII 清理；device API 不暴露在一般公開標頭中。
5. `sw/base.hpp` 的 `Base`、`EncodedSequenceView` 與編碼 helper 可用於 device。
   view 只是一個 pointer/size，並不傳輸資料；host pointer 不可直接当 device pointer 使用。
6. 對同一 global oracle 與 CIGAR 契約驗證，無 GPU 的 runtime 測試明確 skip。
7. Batch/benchmark 路徑實作 `detail::make_cuda_batch`，契約見 [benchmark.md](benchmark.md)。

## Striped 的界線

`backend.cpp` 以 baseline ISA 驗證 CPU、輸入與 scoring，然後呼叫 kernel。
只有 `kernel.cpp` 加上 `-msse4.1` 與 C++20。
`detail/global_alignment.hpp` 與 `detail/simd_core.hpp` 是本專案自有演算法核心，
直接使用 `sw::SequenceView` 與 `sw::Cigar`，不需要舊 Biovoltron 型別轉接。
核心存在非 inline 函式，因此只允許 `kernel.cpp` 引入其標頭。
第三方相依只有 `third_party/libsimdpp`。
