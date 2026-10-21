# 第三方相依

本目錄只保留外部 SIMD 函式庫 **libsimdpp**，以 Boost Software License 1.0 授權，
授權全文見 `libsimdpp/LICENSE`。

`libsimdpp/include/simdpp/` 由原套件的
`lib/Biovoltron/submodules/libsimdpp/simdpp/` 搬入，內容保持原樣。
完整舊套件與原始校驗清單已打包於 repo 外（`~/backups/sw-alignment-backup-original-20261007.tar.gz`）。

Striped 是本專案作者的自有實作，位於 `src/backends/striped/detail/`，
不屬於第三方程式碼。共用 base helper 的 Biovoltron MIT 聲明
另保留於 `licenses/Biovoltron-MIT.txt`。

從本目錄驗證第三方檔案：

```sh
sha256sum --check SHA256SUMS
```
