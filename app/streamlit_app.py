"""Chicago 3.0 網頁入口：主流程與系統診斷兩頁。"""
import sys
from pathlib import Path

# 一律用 repo 裡的 chicago3 程式碼：Streamlit Cloud 只在 requirements.txt 改變時才重裝套件，裝好的副本可能是舊版
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Chicago 3.0", layout="wide")
HERE = Path(__file__).parent
nav = st.navigation([st.Page(HERE / "policy_page.py", title="政策組合", default=True),
                     st.Page(HERE / "diagnostics.py", title="系統診斷")])
nav.run()
