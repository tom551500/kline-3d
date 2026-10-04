import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(page_title="3日K線", layout="wide")
st.title("📈 台股 3日K 線（FinMind）")

API_URL = "https://api.finmindtrade.com/api/v4/data"

RED = "#e53935"  # 台股：紅漲
GREEN = "#00a152"  # 綠跌


# ---------------------------------------------------------------- 資料
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_dataset(dataset: str, stock_id: str, start: str, token: str) -> pd.DataFrame:
    """通用 FinMind 資料抓取。失敗或沒資料時回傳空表。"""
    params = {"dataset": dataset, "data_id": stock_id, "start_date": start}
    if token:
        params["token"] = token
    r = requests.get(API_URL, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    if js.get("status") != 200 or not js.get("data"):
        return pd.DataFrame()
    return pd.DataFrame(js["data"])


@st.cache_data(ttl=3600, show_spinner="下載資料中…")
def load_daily(stock_id: str, start: str, token: str) -> pd.DataFrame:
    df = fetch_dataset("TaiwanStockPrice", stock_id, start, token)
    if df.empty:
        return df
    df = df.rename(columns={"max": "high", "min": "low", "Trading_Volume": "volume"})
    df["date"] = pd.to_datetime(df["date"])
    df = df[["date", "open", "high", "low", "close", "volume"]]
    df = df[df["volume"] > 0].sort_values("date").reset_index(drop=True)
    return df


def safe_fetch(dataset: str, label: str) -> pd.DataFrame:
    try:
        return fetch_dataset(dataset, stock_id, start, token)
    except Exception as e:  # 副資料失敗不影響主圖
        st.warning(f"{label}下載失敗：{e}")
        return pd.DataFrame()


# ---------------------------------------------------------------- N日K合成
def make_fixed_n(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """固定分組：從最新一天往回，每 n 個交易日合成一根（最新一根必為完整 n 日）。"""
    d = df.copy()
    total = len(d)
    d["grp"] = (total - 1 - d.index) // n
    out = d.groupby("grp").agg(
        date=("date", "last"),
        start=("date", "first"),
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        days=("date", "count"),
    )
    out = out[out["days"] == n].sort_values("date").reset_index(drop=True)
    return out


def make_rolling_n(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """滾動式：每天都用最近 n 個交易日重算一根。"""
    out = pd.DataFrame(
        {
            "date": df["date"],
            "open": df["open"].shift(n - 1),
            "high": df["high"].rolling(n).max(),
            "low": df["low"].rolling(n).min(),
            "close": df["close"],
            "volume": df["volume"].rolling(n).sum(),
        }
    ).dropna()
    return out.reset_index(drop=True)


# ---------------------------------------------------------------- 指標
def calc_kd(k: pd.DataFrame, period: int = 9) -> pd.DataFrame:
    """台股常用 KD (9,3,3)：K = 2/3*前K + 1/3*RSV，起始值 50。"""
    low_min = k["low"].rolling(period, min_periods=1).min()
    high_max = k["high"].rolling(period, min_periods=1).max()
    rng = (high_max - low_min).replace(0, float("nan"))
    rsv = ((k["close"] - low_min) / rng * 100).fillna(50)
    kv, dv = 50.0, 50.0
    ks, ds = [], []
    for v in rsv:
        kv = kv * 2 / 3 + v / 3
        dv = dv * 2 / 3 + kv / 3
        ks.append(kv)
        ds.append(dv)
    return pd.DataFrame({"K": ks, "D": ds}, index=k.index)


def calc_macd(close: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9) -> pd.DataFrame:
    dif = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    macd = dif.ewm(span=sig, adjust=False).mean()
    return pd.DataFrame({"DIF": dif, "MACD": macd, "OSC": dif - macd}, index=close.index)


def _who(name: str) -> str:
    if name.startswith("Foreign"):
        return "外資"
    if name.startswith("Investment"):
        return "投信"
    return "自營商"


def _fmt(v, digits: int = 2) -> str:
    return "-" if pd.isna(v) else f"{v:.{digits}f}"


# ---------------------------------------------------------------- 側邊欄（第一段）
with st.sidebar:
    stock_id = st.text_input("股票代號", "2330").strip()
    n = st.number_input("幾日K", min_value=2, max_value=20, value=3, step=1)
    mode = st.radio("合成方式", ["固定分組（每N天一根）", "滾動式（每天重算）"])
    years = st.slider("抓取年數", 1, 5, 2)
    show_ma = st.checkbox("顯示均線 (5 / 10 / 20)", value=True)
    st.markdown("**副圖與標記**")
    show_kd = st.checkbox("KD", value=True)
    show_macd = st.checkbox("MACD", value=True)
    show_inst = st.checkbox("三大法人買賣超", value=True)
    show_exdiv = st.checkbox("除權息日標記", value=True)
    try:
        _default_token = st.secrets.get("FINMIND_TOKEN", "")
    except Exception:  # 沒有設定 secrets 時不要讓程式掛掉
        _default_token = ""
    token = st.text_input("FinMind Token（選填）", value=_default_token, type="password")

start = (dt.date.today() - dt.timedelta(days=365 * years)).isoformat()

try:
    daily = load_daily(stock_id, start, token)
except Exception as e:
    st.error(f"下載失敗：{e}")
    st.stop()

if daily.empty:
    st.warning("查無資料，請確認股票代號（或 FinMind 流量是否用完，可填入 Token）。")
    st.stop()

k = make_fixed_n(daily, int(n)) if mode.startswith("固定") else make_rolling_n(daily, int(n))

if k.empty:
    st.warning("資料筆數不足以合成。")
    st.stop()

# 均線
for w in (5, 10, 20):
    k[f"MA{w}"] = k["close"].rolling(w).mean()

# KD / MACD
kd = calc_kd(k) if show_kd else None
macd = calc_macd(k["close"]) if show_macd else None

# 三大法人：先算每日淨買賣（張），再對齊成「最近 n 個交易日」的合計
inst = None
if show_inst:
    raw = safe_fetch("TaiwanStockInstitutionalInvestorsBuySell", "三大法人資料")
    if raw.empty or not {"date", "buy", "sell", "name"}.issubset(raw.columns):
        st.info("查無三大法人資料（可能是 FinMind 流量用完，或此標的沒有資料）。")
    else:
        raw["date"] = pd.to_datetime(raw["date"])
        raw["net"] = (raw["buy"] - raw["sell"]) / 1000
        raw["who"] = raw["name"].map(_who)
        piv = raw.pivot_table(index="date", columns="who", values="net", aggfunc="sum")
        piv = piv.reindex(pd.Index(daily["date"])).fillna(0.0)
        for c in ("外資", "投信", "自營商"):
            if c not in piv.columns:
                piv[c] = 0.0
        piv = piv[["外資", "投信", "自營商"]].rolling(int(n), min_periods=1).sum()
        inst = piv.reindex(pd.Index(k["date"])).reset_index(drop=True)

# 除權息日
ex_x, ex_y, ex_txt = [], [], []
x = k["date"].dt.strftime("%Y-%m-%d")  # 用類別軸，自動跳過假日空檔
if show_exdiv:
    ed = safe_fetch("TaiwanStockDividendResult", "除權息資料")
    if not ed.empty and "date" in ed.columns:
        ed["date"] = pd.to_datetime(ed["date"])
        for _, r in ed.sort_values("date").iterrows():
            idx = int(k["date"].searchsorted(r["date"]))  # 包含該除息日的那一根
            if idx >= len(k):
                continue

            def g(col):
                v = r.get(col)
                return "-" if v is None or pd.isna(v) else v

            ex_x.append(x.iloc[idx])
            ex_y.append(float(k["high"].iloc[idx]) * 1.015)
            ex_txt.append(
                f"<b>除權息日 {r['date']:%Y-%m-%d}</b><br>"
                f"除權息前收盤 {g('before_price')}<br>"
                f"參考價 {g('after_price')}<br>"
                f"權值＋息值 {g('stock_and_cache_dividend')}"
            )

# ---------------------------------------------------------------- 側邊欄（第二段：盈虧比）
last_close = float(daily["close"].iloc[-1])
with st.sidebar:
    st.markdown("---")
    st.subheader("🧮 開單盈虧比")
    direction = st.radio("方向", ["開多", "開空"], horizontal=True)
    entry = st.number_input("進場價", value=last_close, step=0.05, format="%.2f", key=f"entry_{stock_id}")
    if direction == "開多":
        def_stop, def_target = last_close * 0.95, last_close * 1.10
    else:
        def_stop, def_target = last_close * 1.05, last_close * 0.90
    stop = st.number_input("停損價", value=round(def_stop, 2), step=0.05, format="%.2f", key=f"stop_{stock_id}_{direction}")
    target = st.number_input("目標價", value=round(def_target, 2), step=0.05, format="%.2f", key=f"target_{stock_id}_{direction}")
    lots = st.number_input("張數", min_value=1, value=1, step=1)
    fee_disc = st.number_input("手續費折數（折）", min_value=0.1, max_value=10.0, value=2.8, step=0.1)
    tax_rate = st.number_input("證交稅率 (%)", min_value=0.0, value=0.3, step=0.05, format="%.2f") / 100
    show_rr_lines = st.checkbox("在圖上畫出進場/停損/目標線", value=True)

shares = int(lots) * 1000
is_long = direction == "開多"
rr_valid = (stop < entry < target) if is_long else (target < entry < stop)

risk_ps = (entry - stop) if is_long else (stop - entry)
reward_ps = (target - entry) if is_long else (entry - target)


def net_pnl(exit_price: float) -> float:
    gross = (exit_price - entry) * shares if is_long else (entry - exit_price) * shares
    fee_rate = 0.001425 * fee_disc / 10
    fees = (entry + exit_price) * shares * fee_rate
    sell_value = (exit_price if is_long else entry) * shares  # 賣出那一邊課證交稅
    tax = sell_value * tax_rate
    return gross - fees - tax


# ---------------------------------------------------------------- 滑鼠提示文字
prev_close = k["close"].shift(1)
chg = (k["close"] - prev_close) / prev_close * 100

hover = []
for i in range(len(k)):
    c = chg.iloc[i]
    c_txt = "-" if pd.isna(c) else f"{c:+.2f}%"
    lines = [
        f"<b>{x.iloc[i]}</b>",
        f"開 {k['open'].iloc[i]:.2f}　高 {k['high'].iloc[i]:.2f}",
        f"低 {k['low'].iloc[i]:.2f}　收 {k['close'].iloc[i]:.2f}",
        f"漲跌 {c_txt}",
        f"量 {k['volume'].iloc[i] / 1000:,.0f} 張",
        f"MA5 {_fmt(k['MA5'].iloc[i])}　MA10 {_fmt(k['MA10'].iloc[i])}",
        f"MA20 {_fmt(k['MA20'].iloc[i])}",
    ]
    if kd is not None:
        lines.append(f"K {_fmt(kd['K'].iloc[i], 1)}　D {_fmt(kd['D'].iloc[i], 1)}")
    if macd is not None:
        lines.append(
            f"DIF {_fmt(macd['DIF'].iloc[i])}　MACD {_fmt(macd['MACD'].iloc[i])}　柱 {_fmt(macd['OSC'].iloc[i])}"
        )
    if inst is not None:
        lines.append(
            f"外資 {inst['外資'].iloc[i]:,.0f}　投信 {inst['投信'].iloc[i]:,.0f}　自營 {inst['自營商'].iloc[i]:,.0f}（張）"
        )
    hover.append("<br>".join(lines))

# ---------------------------------------------------------------- 圖表
panels = ["price", "vol"]
if kd is not None:
    panels.append("kd")
if macd is not None:
    panels.append("macd")
if inst is not None:
    panels.append("inst")
row_of = {p: i + 1 for i, p in enumerate(panels)}
weights = {"price": 3.0, "vol": 1.0, "kd": 1.2, "macd": 1.2, "inst": 1.2}
w_sum = sum(weights[p] for p in panels)

fig = make_subplots(
    rows=len(panels),
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.02,
    row_heights=[weights[p] / w_sum for p in panels],
)

# 價格區
fig.add_trace(
    go.Candlestick(
        x=x,
        open=k["open"],
        high=k["high"],
        low=k["low"],
        close=k["close"],
        text=hover,
        hoverinfo="text",
        name=f"{int(n)}日K",
        increasing_line_color=RED,
        increasing_fillcolor=RED,
        decreasing_line_color=GREEN,
        decreasing_fillcolor=GREEN,
    ),
    row=1,
    col=1,
)
if show_ma:
    for w, c in zip((5, 10, 20), ("#fb8c00", "#1e88e5", "#8e24aa")):
        fig.add_trace(
            go.Scatter(x=x, y=k[f"MA{w}"], name=f"MA{w}", line=dict(width=1.2, color=c), hoverinfo="skip"),
            row=1,
            col=1,
        )
if ex_x:
    fig.add_trace(
        go.Scatter(
            x=ex_x,
            y=ex_y,
            mode="markers+text",
            text=["除"] * len(ex_x),
            textposition="top center",
            textfont=dict(size=11, color="#6a1b9a"),
            marker=dict(symbol="triangle-down", size=9, color="#6a1b9a"),
            hovertext=ex_txt,
            hoverinfo="text",
            name="除權息",
        ),
        row=1,
        col=1,
    )

if show_rr_lines and rr_valid:
    for price, label, color in (
        (entry, "進場", "#546e7a"),
        (stop, "停損", "#d32f2f"),
        (target, "目標", "#2e7d32"),
    ):
        fig.add_hline(
            y=price,
            row=1,
            col=1,
            line=dict(color=color, width=1.2, dash="dash"),
            annotation_text=f"{label} {price:.2f}",
            annotation_position="right",
            annotation_font_color=color,
        )

# 成交量
vol_colors = [RED if c >= o else GREEN for o, c in zip(k["open"], k["close"])]
fig.add_trace(
    go.Bar(
        x=x,
        y=k["volume"] / 1000,
        marker_color=vol_colors,
        name="成交量",
        showlegend=False,
        hovertemplate="量 %{y:,.0f} 張<extra></extra>",
    ),
    row=row_of["vol"],
    col=1,
)

# KD
if kd is not None:
    r_ = row_of["kd"]
    fig.add_trace(
        go.Scatter(x=x, y=kd["K"], name="K", line=dict(width=1.3, color="#fb8c00"), hovertemplate="K %{y:.1f}<extra></extra>"),
        row=r_,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=x, y=kd["D"], name="D", line=dict(width=1.3, color="#1e88e5"), hovertemplate="D %{y:.1f}<extra></extra>"),
        row=r_,
        col=1,
    )
    for lv in (20, 80):
        fig.add_hline(y=lv, row=r_, col=1, line=dict(color="#90a4ae", width=1, dash="dot"))
    fig.update_yaxes(range=[0, 100], row=r_, col=1)

# MACD
if macd is not None:
    r_ = row_of["macd"]
    osc_colors = [RED if v >= 0 else GREEN for v in macd["OSC"].fillna(0)]
    fig.add_trace(
        go.Bar(x=x, y=macd["OSC"], marker_color=osc_colors, name="柱狀體", showlegend=False, hovertemplate="柱 %{y:.2f}<extra></extra>"),
        row=r_,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=x, y=macd["DIF"], name="DIF", line=dict(width=1.2, color="#1e88e5"), hovertemplate="DIF %{y:.2f}<extra></extra>"),
        row=r_,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=x, y=macd["MACD"], name="MACD", line=dict(width=1.2, color="#fb8c00"), hovertemplate="MACD %{y:.2f}<extra></extra>"),
        row=r_,
        col=1,
    )

# 三大法人
if inst is not None:
    r_ = row_of["inst"]
    for who, col in (("外資", "#1e88e5"), ("投信", "#fb8c00"), ("自營商", "#8e24aa")):
        fig.add_trace(
            go.Bar(x=x, y=inst[who], name=who, marker_color=col, hovertemplate=f"{who} %{{y:,.0f}} 張<extra></extra>"),
            row=r_,
            col=1,
        )

spike = dict(
    showspikes=True,
    spikemode="across",
    spikesnap="cursor",
    spikethickness=1,
    spikedash="dot",
    spikecolor="#78909c",
)
fig.update_xaxes(type="category", rangeslider_visible=False, nticks=12, **spike)
fig.update_yaxes(**spike)
fig.update_yaxes(title_text="價格", row=1, col=1)
fig.update_yaxes(title_text="量(張)", row=row_of["vol"], col=1)
if kd is not None:
    fig.update_yaxes(title_text="KD", row=row_of["kd"], col=1)
if macd is not None:
    fig.update_yaxes(title_text="MACD", row=row_of["macd"], col=1)
if inst is not None:
    fig.update_yaxes(title_text="法人(張)", row=row_of["inst"], col=1)

extra_panels = len(panels) - 2
fig.update_layout(
    height=560 + 190 * extra_panels,
    dragmode="pan",
    hovermode="x",
    barmode="relative",
    hoverlabel=dict(align="left", font_size=13),
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(orientation="h", y=1.02),
    newshape=dict(line=dict(color="#ff6f00", width=2)),
)

chart_config = {
    "scrollZoom": False,  # 關閉滾輪縮放，避免誤觸
    "doubleClick": "reset",  # 連點兩下回到完整範圍
    "displaylogo": False,
    "modeBarButtonsToAdd": ["drawline", "drawopenpath", "drawrect", "eraseshape"],
    "modeBarButtonsToRemove": ["select2d", "lasso2d"],
}
st.caption("🖱️ 預設為拖曳平移；要縮放請點工具列的放大鏡。滑鼠連點兩下可回到完整範圍。工具列可畫線、刪除選取的線。")
st.plotly_chart(fig, width="stretch", config=chart_config)
if inst is not None:
    st.caption(f"三大法人為「最近 {int(n)} 個交易日」合計的買賣超（張），與 K 棒同一期間；最新一天的資料可能尚未公布。")

# ---------------------------------------------------------------- 盈虧比結果
st.subheader(f"🧮 {direction} 盈虧比（{stock_id}，{int(lots)} 張）")
if not rr_valid:
    if is_long:
        st.warning("開多需要：停損價 < 進場價 < 目標價，請調整側邊欄的價格。")
    else:
        st.warning("開空需要：目標價 < 進場價 < 停損價，請調整側邊欄的價格。")
else:
    profit = net_pnl(target)
    loss = -net_pnl(stop)  # 以正數表示虧損金額
    gross_rr = reward_ps / risk_ps
    net_rr = profit / loss if loss > 0 else float("inf")
    breakeven = 1 / (1 + net_rr) * 100 if net_rr != float("inf") else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("盈虧比（未含成本）", f"1 : {gross_rr:.2f}")
    c2.metric("盈虧比（扣手續費稅）", f"1 : {net_rr:.2f}")
    c3.metric("達標淨獲利", f"{profit:,.0f} 元", f"+{reward_ps / entry * 100:.2f}%")
    c4.metric("停損淨虧損", f"-{loss:,.0f} 元", f"-{risk_ps / entry * 100:.2f}%", delta_color="inverse")
    st.caption(f"損益兩平勝率約 {breakeven:.1f}%（勝率高於此值，長期期望值才為正）。")
    st.caption(
        "試算已扣手續費（買賣各一次）與證交稅（賣出那一邊）；開空另有融券利息／借券費，此處未計入。"
        "此工具僅為試算，不構成投資建議。"
    )

with st.expander("查看合成後資料"):
    st.dataframe(k.tail(60).iloc[::-1], width="stretch")
