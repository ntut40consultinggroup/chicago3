"""Chicago 3.0 網頁入口：主流程與系統診斷兩頁。"""
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Chicago 3.0", layout="wide")
HERE = Path(__file__).parent
nav = st.navigation([st.Page(HERE / "policy_page.py", title="政策組合", default=True),
                     st.Page(HERE / "diagnostics.py", title="系統診斷")])
nav.run()
