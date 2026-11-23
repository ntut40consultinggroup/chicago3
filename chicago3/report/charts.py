"""報告圖表（matplotlib，輸出 PNG）。配色用 dataviz 參考色盤第 1、2 槽（已用 validate_palette.js 驗證）。"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE, BAND = "#2a78d6", "#eb6834", "#b7d3f6"
OFFSETS = {"policy": (8, 6), "neutral": (8, -16), "benchmark": (8, -12), "fund": (8, -12)}   # 中性與政策常重疊
LABELS_ZH = {"benchmark": "指標", "neutral": "可投資中性", "policy": "政策組合", "fund": "基金現況"}


def _font() -> bool:
    """找中文字型；找不到就回傳 False（呼叫端改用英文標籤）。"""
    for f in font_manager.findSystemFonts():
        if "NotoSansCJK-Regular" in f or "NotoSansTC" in f or "msjh" in f.lower():
            font_manager.fontManager.addfont(f)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False


def _style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=9)


def frontier_chart(frontier, points: dict, te_budget: float, path, title: str = ""):
    """追蹤誤差前緣（x：TE，y：預期主動報酬），標出各組合。points：名稱 → portfolio_stats。"""
    zh = _font()
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURF)
    _style(ax)
    ax.plot(frontier.risk * 100, frontier.ret * 100, color=BLUE, lw=2, label="效率前緣" if zh else "Frontier")
    for k, p in points.items():
        is_fund = k == "fund"
        ax.scatter(p["te"] * 100, p["expected_active"] * 100, s=60, color=ORANGE if is_fund else INK,
                   edgecolor=SURF, linewidth=2, zorder=3)
        name = LABELS_ZH.get(k, k) if zh else k
        ax.annotate(f"{name}  β {p['beta']:.2f}", (p["te"] * 100, p["expected_active"] * 100),
                    xytext=OFFSETS.get(k, (6, -12)), textcoords="offset points", color=INK, fontsize=9,
                    ha="left")
    ax.axvline(te_budget * 100, color=MUTED, lw=1, ls=":")
    ax.annotate(f"{'總 TE 預算' if zh else 'TE budget'} {te_budget:.0%}", (te_budget * 100, ax.get_ylim()[1]),
                xytext=(4, -14), textcoords="offset points", color=INK2, fontsize=9)
    ax.set_xlabel("追蹤誤差（%，對指標）" if zh else "Tracking error (%)", color=INK2)
    ax.set_ylabel("預期主動報酬（%／年）" if zh else "Expected active return (%/yr)", color=INK2)
    ax.set_title(title, color=INK, fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def bands_chart(table, path, title: str = ""):
    """每桶：區間（淺色帶）、政策目標（藍色短線）、基金現況（橘點）。table 欄位：policy、band_lower、band_upper、fund（可選）。"""
    zh = _font()
    t = table.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(t) + 1.4), facecolor=SURF)
    _style(ax)
    ax.grid(axis="y", visible=False)
    y = range(len(t))
    ax.barh(y, (t.band_upper - t.band_lower) * 100, left=t.band_lower * 100, height=0.5, color=BAND,
            label="區間" if zh else "Band")
    ax.scatter(t.policy * 100, y, marker="|", s=400, color=BLUE, linewidth=2.5, label="政策目標" if zh else "Policy", zorder=3)
    if "fund" in t:
        ax.scatter(t.fund * 100, y, s=55, color=ORANGE, edgecolor=SURF, linewidth=2, label="基金現況" if zh else "Fund",
                   zorder=4)
    ax.set_yticks(list(y), t.index)
    ax.set_xlabel("權重（%，股票部位）" if zh else "Weight (%)", color=INK2)
    ax.set_title(title, color=INK, fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
