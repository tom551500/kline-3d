import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="3日K線", layout="wide")
st.title("📈 台股 3日K 線（FinMind）")

API_URL = "https://api.finmindtrade.com/api/v4/data"


@st.cache_data(ttl=3600, show_spinner="下載資料中…")
def load_daily(stock_id: str, start: str, token: str) -> pd.DataFrame:
    params = {"dataset": "TaiwanStockPrice", "data_id": stock_id, "start_date": start}
    if token:
        params["token"] = token
    r = requests.get(API_URL, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    if js.get("status") != 200 or not js.get("data"):
        return pd.DataFrame()
    df = pd.DataFrame(js["data"])
    df = df.rename(columns={"max": "high", "min": "low", "Trading_Volume": "volume"})
    df["date"] = pd.to_datetime(df["date"])
    df = df[["date", "open", "high", "low", "close", "volume"]]
    df = df[df["volume"] > 0].sort_values("date").reset_index(drop=True)
    return df


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


with st.sidebar:
    stock_id = st.text_input("股票代號", "2330").strip()
    n = st.number_input("幾日K", min_value=2, max_value=20, value=3, step=1)
    mode = st.radio("合成方式", ["固定分組（每N天一根）", "滾動式（每天重算）"])
    years = st.slider("抓取年數", 1, 5, 2)
    show_ma = st.checkbox("顯示均線 (5 / 10 / 20)", value=True)
    token = st.text_input(
        "FinMind Token（選填）",
        value=st.secrets.get("FINMIND_TOKEN", "") if hasattr(st, "secrets") else "",
        type="password",
    )

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

# ---------------- 開單盈虧比計算 ----------------
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

x = k["date"].dt.strftime("%Y-%m-%d")  # 用類別軸，自動跳過假日空檔

# 滑鼠移過去顯示的資訊（開高低收、漲跌幅、成交量、均線）
prev_close = k["close"].shift(1)
chg = (k["close"] - prev_close) / prev_close * 100


def _fmt(v):
    return "-" if pd.isna(v) else f"{v:.2f}"


hover = []
for i in range(len(k)):
    c = chg.iloc[i]
    c_txt = "-" if pd.isna(c) else f"{c:+.2f}%"
    hover.append(
        f"<b>{x.iloc[i]}</b><br>"
        f"開 {k['open'].iloc[i]:.2f}　高 {k['high'].iloc[i]:.2f}<br>"
        f"低 {k['low'].iloc[i]:.2f}　收 {k['close'].iloc[i]:.2f}<br>"
        f"漲跌 {c_txt}<br>"
        f"量 {k['volume'].iloc[i] / 1000:,.0f} 張<br>"
        f"MA5 {_fmt(k['MA5'].iloc[i])}　MA10 {_fmt(k['MA10'].iloc[i])}<br>"
        f"MA20 {_fmt(k['MA20'].iloc[i])}"
    )

fig = go.Figure()
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
        increasing_line_color="#e53935",  # 台股：紅漲
        increasing_fillcolor="#e53935",
        decreasing_line_color="#00a152",  # 綠跌
        decreasing_fillcolor="#00a152",
    )
)
if show_ma:
    for w, c in zip((5, 10, 20), ("#fb8c00", "#1e88e5", "#8e24aa")):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=k[f"MA{w}"],
                name=f"MA{w}",
                line=dict(width=1.2, color=c),
                hoverinfo="skip",
            )
        )

spike = dict(
    showspikes=True,
    spikemode="across",
    spikesnap="cursor",
    spikethickness=1,
    spikedash="dot",
    spikecolor="#78909c",
)
fig.update_layout(
    height=640,
    dragmode="pan",
    hovermode="x",
    hoverlabel=dict(align="left", font_size=13),
    xaxis=dict(type="category", rangeslider=dict(visible=False), nticks=12, **spike),
    yaxis=dict(title="價格", **spike),
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(orientation="h", y=1.05),
    newshape=dict(line=dict(color="#ff6f00", width=2)),
)

if show_rr_lines and rr_valid:
    for price, label, color in (
        (entry, "進場", "#546e7a"),
        (stop, "停損", "#d32f2f"),
        (target, "目標", "#2e7d32"),
    ):
        fig.add_hline(
            y=price,
            line=dict(color=color, width=1.2, dash="dash"),
            annotation_text=f"{label} {price:.2f}",
            annotation_position="right",
            annotation_font_color=color,
        )

chart_config = {
    "scrollZoom": False,  # 關閉滾輪縮放，避免誤觸
    "doubleClick": "reset",  # 連點兩下回到完整範圍
    "displaylogo": False,
    "modeBarButtonsToAdd": ["drawline", "drawopenpath", "drawrect", "eraseshape"],
    "modeBarButtonsToRemove": ["select2d", "lasso2d"],
}
st.caption("🖱️ 預設為拖曳平移；要縮放請點工具列的放大鏡。滑鼠連點兩下可回到完整範圍。工具列可畫線、刪除選取的線。")
st.plotly_chart(fig, use_container_width=True, config=chart_config)

# ---------------- 盈虧比結果 ----------------
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

vol = go.Figure(go.Bar(x=x, y=k["volume"] / 1000, marker_color="#90a4ae"))
vol.update_layout(
    height=180,
    xaxis=dict(type="category", nticks=12),
    yaxis=dict(title="成交量(張)"),
    margin=dict(l=10, r=10, t=10, b=10),
)
st.plotly_chart(vol, use_container_width=True)

with st.expander("查看合成後資料"):
    st.dataframe(k.tail(60).iloc[::-1], use_container_width=True)
