# Chicago 3.0

股票基金的策略性資產配置（SAA）工具：Black–Litterman ＋ 以追蹤誤差為風險的均值–變異數最佳化，放在「指標 → 可投資中性 → 政策 → 實際基金」的三層框架裡。

> 狀態：第 0 期（凍結基準＋可行性試做）。規劃見 `docs/`，與專案文件《SAA套裝軟體_開發規劃》。
> 教學與研究用途；模型輸出需經投資委員會核准，不構成投資建議。

## 目前可用

```bash
pip install -r requirements.txt pytest
python -m pytest -q tests            # 確認凍結的黃金數字與講義一致
streamlit run app/streamlit_app.py   # 第 0 期試做版
```

## 目錄

| 目錄 | 內容 |
|---|---|
| `legacy/` | 課程原始程式（只當參考） |
| `tools/` | 凍結黃金數字的腳本 |
| `tests/` | 黃金數字、核心函數 fixture、測試 |
| `app/` | Streamlit 網頁 |
| `docs/` | 各期報告、設定指南 |
