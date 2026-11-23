"""Chicago 3.0 — 第 0 期試做版（Hello）。

用途：
1. 確認 Streamlit Community Cloud 部署成功。
2. 從雲端測試資料來源能不能連（開發沙盒連不到部分投信網站）。
3. 測試用 OAuth（drive.file 權限）在 Google Drive 建立 chicago3.0 資料夾並寫檔。

需要的 secrets（Streamlit Cloud → App settings → Secrets）：
  [google]
  client_id = "..."
  client_secret = "..."
  refresh_token = "..."   # 第一次授權後才填
"""
import json
import platform
import sys
import time
from urllib.parse import parse_qs, urlencode, urlparse

import requests
import streamlit as st

SCOPE = "https://www.googleapis.com/auth/drive.file"
REDIRECT = "http://localhost"
FOLDER = "chicago3.0"
PROBES = {
    "FinMind API": "https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInfo",
    "證交所 OpenAPI": "https://openapi.twse.com.tw/v1/opendata/t187ap03_L",
    "櫃買 OpenAPI": "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes",
    "證交所 www": "https://www.twse.com.tw/zh/",
    "統一投信 ezmoney": "https://www.ezmoney.com.tw/",
    "元大投信": "https://www.yuantaetfs.com/",
    "國泰投信": "https://www.cathaysite.com.tw/",
    "群益投信": "https://www.capitalfund.com.tw/",
    "復華投信": "https://www.fhtrust.com.tw/",
    "富果 API": "https://api.fugle.tw/marketdata/v1.0/stock/intraday/tickers",
}

st.set_page_config(page_title="Chicago 3.0（試做）", layout="wide")
st.title("Chicago 3.0 — 第 0 期試做")
st.caption("教學與研究用途；模型輸出不構成投資建議。")

tab_env, tab_net, tab_drive = st.tabs(["環境", "資料來源連線", "Google Drive"])

with tab_env:
    st.write({"python": sys.version.split()[0], "platform": platform.platform(), "streamlit": st.__version__})

with tab_net:
    if st.button("測試連線"):
        rows = []
        for name, url in PROBES.items():
            t0 = time.time()
            try:
                r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
                rows.append(dict(來源=name, 狀態=r.status_code, 秒=round(time.time() - t0, 2), 大小=len(r.content)))
            except Exception as e:  # 連線失敗也要列出來，這正是要測的
                rows.append(dict(來源=name, 狀態=f"失敗：{type(e).__name__}", 秒=round(time.time() - t0, 2), 大小=0))
        st.dataframe(rows, use_container_width=True)


def token_from_refresh(cfg):
    r = requests.post("https://oauth2.googleapis.com/token", data=dict(
        client_id=cfg["client_id"], client_secret=cfg["client_secret"],
        refresh_token=cfg["refresh_token"], grant_type="refresh_token"), timeout=20)
    r.raise_for_status()
    return r.json()["access_token"]


def find_or_create_folder(tok):
    h = {"Authorization": f"Bearer {tok}"}
    q = f"name='{FOLDER}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    r = requests.get("https://www.googleapis.com/drive/v3/files", headers=h, params=dict(q=q, fields="files(id,name)"), timeout=20)
    r.raise_for_status()
    files = r.json()["files"]
    if files:
        return files[0]["id"], False
    r = requests.post("https://www.googleapis.com/drive/v3/files", headers=h, timeout=20,
                      json=dict(name=FOLDER, mimeType="application/vnd.google-apps.folder"))
    r.raise_for_status()
    return r.json()["id"], True


def upload_text(tok, parent, name, text):
    h = {"Authorization": f"Bearer {tok}"}
    meta = json.dumps(dict(name=name, parents=[parent]))
    files = {"metadata": ("metadata", meta, "application/json"), "file": (name, text.encode("utf-8"), "text/plain")}
    r = requests.post("https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart&fields=id,webViewLink",
                      headers=h, files=files, timeout=30)
    r.raise_for_status()
    return r.json()


with tab_drive:
    cfg = dict(st.secrets.get("google", {}))
    # 公開 app：Drive 功能要先輸入密碼（Secrets 的 app_password）。沒設密碼就整個關閉，避免任何人寫入你的 Drive。
    pw = st.secrets.get("app_password", "")
    if not pw:
        st.warning("尚未設定 app_password，Drive 功能已關閉。請在 Secrets 加一行 app_password = \"...\"。")
        st.stop()
    if st.text_input("管理密碼", type="password") != pw:
        st.info("輸入管理密碼後才能使用 Drive 功能。")
        st.stop()
    if not cfg.get("client_id"):
        st.info("尚未設定 [google] client_id／client_secret（見 docs/第0期_設定指南.md）。")
    elif not cfg.get("refresh_token"):
        st.subheader("第一次授權")
        url = "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(dict(
            client_id=cfg["client_id"], redirect_uri=REDIRECT, response_type="code", scope=SCOPE,
            access_type="offline", prompt="consent"))
        st.markdown(f"1. [點這裡用 Google 帳號授權]({url})（只要求 drive.file：程式只看得到自己建立的檔案）")
        st.markdown("2. 授權後瀏覽器會跳到一個打不開的 `http://localhost/?code=...` 頁面，這是正常的。把整個網址複製貼到下面。")
        pasted = st.text_input("貼上跳轉後的網址")
        if pasted:
            code = parse_qs(urlparse(pasted).query).get("code", [pasted.strip()])[0]
            r = requests.post("https://oauth2.googleapis.com/token", data=dict(
                code=code, client_id=cfg["client_id"], client_secret=cfg["client_secret"],
                redirect_uri=REDIRECT, grant_type="authorization_code"), timeout=20)
            if r.ok and "refresh_token" in r.json():
                st.success("取得 refresh token。請複製到 Secrets 的 [google] refresh_token，存檔後 app 會重新啟動。這個值只顯示這一次，不要分享給別人。")
                st.code(r.json()["refresh_token"])
            else:
                st.error(f"交換失敗：{r.status_code} {r.text[:300]}")
    else:
        if st.button("建立 chicago3.0 資料夾並寫入測試檔"):
            try:
                tok = token_from_refresh(cfg)
                fid, created = find_or_create_folder(tok)
                info = upload_text(tok, fid, f"hello_{time.strftime('%Y%m%d_%H%M%S')}.txt", "Chicago 3.0 第 0 期 Drive 寫入測試")
                st.success(f"{'新建' if created else '找到既有'}資料夾 {FOLDER}（id {fid}）；已寫入測試檔。")
                st.markdown(f"[開啟測試檔]({info.get('webViewLink', '')})")
            except requests.HTTPError as e:
                st.error(f"Drive 錯誤：{e.response.status_code} {e.response.text[:300]}")
