# Backend 擴充

應用程式只依賴 `include/sw/alignment.hpp`。Factory 在 `src/alignment.cpp`，
每個 backend 透過 `src/backend.hpp` 提供 availability 與 factory 函式。
公開型別沒有 CUDA 或 SIMD 標頭，也沒有 virtual device 方法。

```text
應用程式 → sw::Aligner → backend 實作
                          ├─ striped/backend.cpp → kernel.cpp → detail/（依賴 libsimdpp）
                          ├─ scalar/backend.cpp  （待填）
                          └─ cuda/backend.cpp    （待填）
```

## 實作 scalar

1. 在 `src/backends/scalar/` 實作繼承 `sw::Aligner` 的私有類別。
2. 實作 `align(SequenceView, SequenceView)` 與 `backend()`，沿用共同 global 契約。
3. 替換該目錄 `backend.cpp` 的 unavailable/factory 入口；可用時 reason 回傳空字串。
4. 需要新 source 時明確加入 CMake，不使用遞迴 glob。
5. 讓實作通過 `tests/support/global_oracle.hpp` 分數比較與完整 CIGAR 重播。

不要直接移回舊 scalar：備份的實作是 overlap 語意，與目前 global 契約不同。
測試用 oracle 只供驗證，不作為 production scalar backend。

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

## Striped 的界線

`backend.cpp` 以 baseline ISA 驗證 CPU、輸入與 scoring，然後呼叫 kernel。
只有 `kernel.cpp` 加上 `-msse4.1` 與 C++20。
`detail/global_alignment.hpp` 與 `detail/simd_core.hpp` 是本專案自有演算法核心，
直接使用 `sw::SequenceView` 與 `sw::Cigar`，不需要舊 Biovoltron 型別轉接。
核心存在非 inline 函式，因此只允許 `kernel.cpp` 引入其標頭。
`SOURCE.txt` 保留搬入前的歷史來源 revision，後續可直接在此維護演算法。
第三方相依只有 `third_party/libsimdpp`。
