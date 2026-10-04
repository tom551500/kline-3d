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
    hovermode="x",
    hoverlabel=dict(align="left", font_size=13),
    xaxis=dict(type="category", rangeslider=dict(visible=False), nticks=12, **spike),
    yaxis=dict(title="價格", **spike),
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(orientation="h", y=1.05),
    newshape=dict(line=dict(color="#ff6f00", width=2)),
)

chart_config = {
    "scrollZoom": True,  # 滾輪縮放
    "displaylogo": False,
    "modeBarButtonsToAdd": ["drawline", "drawopenpath", "drawrect", "eraseshape"],
}
st.caption("🖱️ 工具列：畫直線 / 自由畫線 / 畫矩形 / 刪除選取的線（點線後按刪除）；滾輪可縮放，拖曳可平移。")
st.plotly_chart(fig, use_container_width=True, config=chart_config)

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
