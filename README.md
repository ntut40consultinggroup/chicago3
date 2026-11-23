# Chicago 3.0

股票基金的策略性資產配置（SAA）工具：Black–Litterman ＋ 以追蹤誤差為風險的均值–變異數最佳化，放在「指標 → 可投資中性 → 政策 → 實際基金」三層框架裡，以「桶」為單位。

> 版本 0.1.0。教學與研究用途；模型輸出須經投資委員會核准，不構成投資建議（見 [免責聲明](docs/免責聲明.md)）。

## 三種用法

| 你是 | 用法 | 文件 |
|---|---|---|
| 學生、想先試用 | 公開網頁，打開就能用（Streamlit Community Cloud） | [使用手冊](docs/使用手冊.md) |
| 業界使用者，持股屬於機密 | 在公司環境安裝，用命令列或本機網頁 | [安裝說明](docs/安裝說明.md) |
| 審查模型（投資委員會、風控、法遵） | 方法、假設、限制、驗證、變更紀錄 | [模型文件](docs/模型文件.md) |

## 快速開始（本機）

```bash
pip install -e ".[app]"
chicago3 sync --store local:./c3data --funds funds.yaml   # 下載市場資料與持股
chicago3 report my_fund.yaml --out out/                    # 設定檔寫法見安裝說明第 6 節
CHICAGO3_STORE=local:./c3data streamlit run app/streamlit_app.py
```

## 目錄

| 目錄 | 內容 |
|---|---|
| `chicago3/` | 套件：風險、均衡、觀點、中性、最佳化、預算、區間、穩健性、歸因、報告、資料、儲存、同步 |
| `app/` | Streamlit 網頁（政策組合七步驟、系統診斷） |
| `examples/` | 設定檔範例（00981A、00991A、流動性情境） |
| `funds.yaml` | 下拉選單的基金與持股來源 |
| `tests/` | 自動測試（黃金數字、人造市場手算、網頁無頭測試） |
| `tools/` | 凍結黃金數字、雲端探測、資料匯入 |
| `legacy/` | 課程原始程式（只當參考） |
| `docs/` | 使用手冊、安裝說明、模型文件、免責聲明、各期開發報告 |
| `.github/workflows/` | 測試（每次推送）、每日同步（交易日 18:35）、雲端探測 |
