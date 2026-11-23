"""SAA 補充課：畫效率前緣（傳統的總風險前緣 ＋ 相對指標的追蹤誤差前緣）。

用法：python3 -I stage8_frontier.py <同 stage8_saa.py 的 7 個參數>
輸出：<前綴>_效率前緣.png、在 <前綴>.xlsx 加上「15_效率前緣」工作表。

左圖（Markowitz）：x = 總波動，y = Black–Litterman 預期超額報酬 μ_BL。
右圖（Roll 1992 追蹤誤差前緣）：x = 對指標 TE，y = 對指標預期主動報酬 μ_BL'(w − w_B)。
兩張圖的前緣只用「硬限制」求：權重加總 1、權重 ≥ 0、台積電 ≤ 25%（法規）、各桶 ≤ 容量。
不加政策用的「單桶傾斜 ≤ 15 個百分點」與「台積電固定 25%」，因為前緣要畫的是全部可行的機會集合。
"""
import contextlib
import io
import sys
from pathlib import Path

import cvxpy as cp
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage8_saa as saa  # noqa: E402

for f in font_manager.findSystemFonts():
    if "NotoSansCJK-Regular" in f:
        font_manager.fontManager.addfont(f)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
plt.rcParams["axes.unicode_minus"] = False

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"      # 參考色盤第 1、2 槽（已通過驗證）


def frontier(mu, S, wN, cap, wB, kind, n=40):
    """kind='total'：給定總波動上限求最大 μ'w；kind='active'：給定 TE（對指標）上限求最大 μ'(w − w_B)。"""
    k = len(mu)
    L = np.linalg.cholesky(S.values + 1e-10 * np.eye(k))
    w = cp.Variable(k)
    lim = cap.copy()
    lim["台積電"] = 0.25
    base = [cp.sum(w) == 1, w >= 0, w <= lim.values]
    risk = (lambda x: cp.norm(L.T @ x)) if kind == "total" else (lambda x: cp.norm(L.T @ (x - wB.values)))
    # 端點：最小風險、最大報酬
    cp.Problem(cp.Minimize(risk(w)), base).solve()
    r0 = float(risk(w).value)
    cp.Problem(cp.Maximize(mu.values @ w), base).solve()
    r1 = float(risk(w).value)
    rows = []
    for r in np.linspace(r0, r1, n):
        cp.Problem(cp.Maximize(mu.values @ w), base + [risk(w) <= r + 1e-9]).solve()
        x = pd.Series(np.clip(w.value, 0, None), mu.index)
        rows.append(dict(風險=r, 預期報酬=float(mu.values @ x.values) - (0 if kind == "total" else float(mu.values @ wB.values)),
                         **{b: x[b] for b in mu.index}))
    return pd.DataFrame(rows)


def point(mu, S, w, wB):
    w = w.values
    vB = float(wB.values @ S.values @ wB.values)
    return dict(Beta=float(w @ S.values @ wB.values) / vB, 總波動=float(np.sqrt(w @ S.values @ w)), 預期超額=float(mu.values @ w),
                TE=float(np.sqrt((w - wB.values) @ S.values @ (w - wB.values))), 預期主動=float(mu.values @ (w - wB.values)))


def main(*args):
    with contextlib.redirect_stdout(io.StringIO()):
        o = saa.main(*args)
    prefix = args[-1]
    p = o["p27"]
    S, wB, wN, wP = p["S"], p["wB"], p["wN"], p["wP"]
    cap = o["capd"]["容量_占淨資產"]
    mu = p["Pi"] + p["alpha"]                               # 基本觀點：AI 年超額 5%、信心 50%
    _, a10, _ = saa.bl_alpha(S, wB.values, [("AI硬體零組件", 0.10)], 0.75)
    mu10 = p["Pi"] + a10                                    # 對照：AI 年超額 10%、信心 75%
    wF = o["wF"].reindex(wB.index).fillna(0)

    fT, fT10 = frontier(mu, S, wN, cap, wB, "total"), frontier(mu10, S, wN, cap, wB, "total")
    fA, fA10 = frontier(mu, S, wN, cap, wB, "active"), frontier(mu10, S, wN, cap, wB, "active")
    pts = pd.DataFrame({k: point(mu, S, w, wB) for k, w in
                        (("指標", wB), ("可投資中性", wN), ("2027 政策組合", wP), ("基金現況", wF))}).T
    pts10 = pd.DataFrame({k: point(mu10, S, w, wB) for k, w in (("基金現況", wF),)}).T
    print(pts.round(4).to_string())
    print("強烈觀點下的基金現況", pts10.round(4).to_string())
    # 同 TE 下前緣能給多少：在基金現況 TE 處，前緣的預期主動報酬
    te_f = pts.loc["基金現況", "TE"]
    on_f = float(np.interp(te_f, fA.風險, fA.預期報酬))
    on_f10 = float(np.interp(te_f, fA10.風險, fA10.預期報酬))
    for k in ("2027 政策組合", "基金現況"):
        print(k, "前緣同TE主動", round(float(np.interp(pts.loc[k, "TE"], fA.風險, fA.預期報酬)), 4),
              "前緣同波動超額", round(float(np.interp(pts.loc[k, "總波動"], fT.風險, fT.預期報酬)), 4))
    i = (fA.風險 - 0.15).abs().idxmin()
    print("追蹤誤差前緣 TE≈15% 的組合", fA.loc[i].round(3).to_dict())
    vB = float(wB.values @ S.values @ wB.values)
    print("該組合 Beta", round(float(fA.loc[i, list(wB.index)].values @ S.values @ wB.values) / vB, 3))
    print(f"基金 TE {te_f:.4f}：前緣主動 {on_f:.4f}（強烈觀點 {on_f10:.4f}）；基金 {pts.loc['基金現況','預期主動']:.4f}（強烈觀點 {pts10.loc['基金現況','預期主動']:.4f}）")

    fig, axs = plt.subplots(1, 2, figsize=(13, 5.6), facecolor=SURF)
    for ax in axs:
        ax.set_facecolor(SURF)
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(INK2)
        ax.tick_params(colors=INK2)
    pct = matplotlib.ticker.PercentFormatter(1.0, decimals=0)

    ax = axs[0]
    ax.plot(fT.風險, fT.預期報酬, color=BLUE, lw=2, label="效率前緣：基本觀點（AI 年超額 5%，信心 50%）")
    ax.plot(fT10.風險, fT10.預期報酬, color=ORANGE, lw=2, ls="--", label="效率前緣：強烈觀點（10%，75%）")
    for k, mk in (("指標", "s"), ("可投資中性", "o"), ("2027 政策組合", "D"), ("基金現況", "X")):
        x, y = pts.loc[k, "總波動"], pts.loc[k, "預期超額"]
        ax.scatter(x, y, s=90, marker=mk, color=INK, edgecolor=SURF, linewidth=1.5, zorder=5)
    ax.annotate("指標、可投資中性、2027 政策組合\n（三點幾乎重疊，β ≈ 1.0）", (pts.loc["2027 政策組合", "總波動"], pts.loc["2027 政策組合", "預期超額"]),
                xytext=(12, -42), textcoords="offset points", color=INK, fontsize=10,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.annotate(f"基金現況（β {pts.loc['基金現況', 'Beta']:.2f}）", (pts.loc["基金現況", "總波動"], pts.loc["基金現況", "預期超額"]),
                xytext=(-40, -22), textcoords="offset points", color=INK, fontsize=10)
    ax.set_xlabel("總波動（年化）", color=INK2)
    ax.set_ylabel("預期超額報酬（Black–Litterman，年化）", color=INK2)
    ax.xaxis.set_major_formatter(pct)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
    ax.set_title("① 傳統效率前緣：總風險 vs 預期報酬", color=INK, loc="left", fontsize=12)

    ax = axs[1]
    ax.plot(fA.風險, fA.預期報酬, color=BLUE, lw=2, label="效率前緣：基本觀點（AI 年超額 5%，信心 50%）")
    ax.plot(fA10.風險, fA10.預期報酬, color=ORANGE, lw=2, ls="--", label="效率前緣：強烈觀點（AI 年超額 10%，信心 75%）")
    ax.axvline(saa.TE_TOTAL, color=INK2, lw=1, ls=":")
    ax.annotate("總 TE 預算 15%", (saa.TE_TOTAL, ax.get_ylim()[1]), xytext=(4, -14), textcoords="offset points",
                color=INK2, fontsize=9)
    for k, mk in (("指標", "s"), ("可投資中性", "o"), ("2027 政策組合", "D"), ("基金現況", "X")):
        ax.scatter(pts.loc[k, "TE"], pts.loc[k, "預期主動"], s=90, marker=mk, color=INK, edgecolor=SURF, linewidth=1.5, zorder=5)
    lab = {"指標": (8, -4), "可投資中性": (-20, -18), "2027 政策組合": (8, 8), f"基金現況（β {pts.loc['基金現況', 'Beta']:.2f}）": (-28, -20)}
    pts = pts.rename(index={"基金現況": f"基金現況（β {pts.loc['基金現況', 'Beta']:.2f}）"})
    for k, (dx, dy) in lab.items():
        ax.annotate(k, (pts.loc[k, "TE"], pts.loc[k, "預期主動"]), xytext=(dx, dy), textcoords="offset points",
                    color=INK, fontsize=10)
    ax.set_xlabel("追蹤誤差（對加權報酬指數，年化）", color=INK2)
    ax.set_ylabel("預期主動報酬（對指標，年化）", color=INK2)
    ax.xaxis.set_major_formatter(pct)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
    ax.set_title("② 追蹤誤差前緣：主動風險 vs 預期主動報酬", color=INK, loc="left", fontsize=12)

    pts.index = ["指標", "可投資中性", "2027 政策組合", "基金現況"]
    h, l = axs[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=10, labelcolor=INK)
    fig.suptitle("00981A 2027 年 SAA：七桶配置的效率前緣（限制：權重 ≥ 0、台積電 ≤ 25%、各桶 ≤ 容量）",
                 color=INK, fontsize=13, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    png = Path(prefix + "_效率前緣.png")
    fig.savefig(png, dpi=160, facecolor=SURF)

    out = Path(prefix + ".xlsx")
    with pd.ExcelWriter(out, engine="openpyxl", mode="a", if_sheet_exists="replace") as xw:
        r = 0
        for name, df in (("A. 總風險前緣（基本觀點）", fT), ("B. 總風險前緣（強烈觀點）", fT10),
                         ("C. 追蹤誤差前緣（基本觀點）", fA), ("D. 追蹤誤差前緣（強烈觀點）", fA10)):
            pd.DataFrame([[name]]).to_excel(xw, "15_效率前緣", startrow=r, index=False, header=False)
            df.to_excel(xw, "15_效率前緣", startrow=r + 1, index=False)
            r += len(df) + 3
        pd.DataFrame([["E. 四個組合的位置（基本觀點）"]]).to_excel(xw, "15_效率前緣", startrow=r, index=False, header=False)
        pts.rename_axis("組合").reset_index().to_excel(xw, "15_效率前緣", startrow=r + 1, index=False)
    print("輸出", png, out)


if __name__ == "__main__":
    main(*sys.argv[1:])
