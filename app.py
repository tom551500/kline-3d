from streamlit_lightweight_charts import renderLightweightCharts
from streamlit_autorefresh import st_autorefresh
import datetime as dt

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="3日K線", layout="wide")
st.title("📈 台股 3日K 線（FinMind · TradingView 風格）")

API_URL = "https://api.finmindtrade.com/api/v4/data"
TWSE_MIS_URL = "https://mis.twse.com.tw/stock/api/getStockInfo.jsp"

RED = "#e53935"    # 台股：紅漲
GREEN = "#00a152"  # 綠跌
BG = "#131722"
GRID = "#1e222d"
TEXT = "#d1d4dc"
BORDER = "#2b2b43"


# ---------------------------------------------------------------- 即時報價（證交所 MIS）
@st.cache_data(ttl=10, show_spinner=False)
def fetch_realtime_price(stock_id: str, market: str) -> dict | None:
    """
    向證交所 MIS API 查即時報價。
    market: "tse"（上市）或 "otc"（上櫃）
    回傳 None 代表查不到（可能是休市、代碼錯、或暫時連不上）。
    """
    ex_ch = f"{market}_{stock_id}.tw"
    params = {"ex_ch": ex_ch, "json": 1, "delay": 0}
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://mis.twse.com.tw/stock/index.jsp",
    }
    try:
        r = requests.get(TWSE_MIS_URL, params=params, headers=headers, timeout=5)
        r.raise_for_status()
        js = r.json()
        arr = js.get("msgArray")
        if not arr:
            return None
        info = arr[0]

        def num(key):
            v = info.get(key)
            if v in (None, "", "-"):
                return None
            try:
                return float(v)
            except ValueError:
                return None

        price = num("z")  # 最新成交價，可能為 "-"（尚無成交）
        if price is None:
            price = num("b") or num("a")  # 退而求其次：買價或賣價
        if price is None:
            return None

        return {
            "price": price,
            "open": num("o"),
            "high": num("h"),
            "low": num("l"),
            "prev_close": num("y"),
            "volume": num("v"),  # 單位：張
            "time": info.get("t") or info.get("tlong"),
        }
    except Exception:
        return None


# ---------------------------------------------------------------- 資料
@st.cache_data(ttl=300, show_spinner=False)
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


@st.cache_data(ttl=300, show_spinner="下載資料中…")
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
    except Exception as e:
        st.warning(f"{label}下載失敗：{e}")
        return pd.DataFrame()


# ---------------------------------------------------------------- N日K合成
def make_fixed_n(df: pd.DataFrame, n: int) -> pd.DataFrame:
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


# ---------------------------------------------------------------- lightweight-charts 小工具
def line_series(times, values, color, title, width=1.3):
    data = [{"time": t, "value": float(v)} for t, v in zip(times, values) if pd.notna(v)]
    return {"type": "Line", "data": data, "options": {"color": color, "lineWidth": width, "title": title}}


def base_chart_options(height: int, show_time_labels: bool = True):
    return {
        "layout": {"background": {"type": "solid", "color": BG}, "textColor": TEXT},
        "grid": {"vertLines": {"color": GRID}, "horzLines": {"color": GRID}},
        "crosshair": {"mode": 0},
        "timeScale": {"borderColor": BORDER, "timeVisible": False, "visible": show_time_labels},
        "rightPriceScale": {"borderColor": BORDER},
        "height": height,
    }


# ---------------------------------------------------------------- 側邊欄（第一段）
with st.sidebar:
    stock_id = st.text_input("股票代號", "2330").strip()
    market = st.radio("市場別", ["上市 (TSE)", "上櫃 (OTC)"], horizontal=True)
    market_code = "tse" if market.startswith("上市") else "otc"
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
    except Exception:
        _default_token = ""
    token = st.text_input("FinMind Token（選填）", value=_default_token, type="password")

    st.markdown("---")
    st.subheader("🔄 自動刷新")
    auto_refresh = st.checkbox("開啟自動刷新", value=True)
    refresh_sec = st.slider("刷新間隔（秒）", min_value=10, max_value=600, value=30, step=10)
    use_realtime = st.checkbox("用即時報價更新最新一根K棒", value=True)

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

# ---------------------------------------------------------------- 用即時報價更新最新K棒
realtime_info = None
if use_realtime:
    realtime_info = fetch_realtime_price(stock_id, market_code)
    if realtime_info:
        last_idx = k.index[-1]
        rt_price = realtime_info["price"]
        # 更新最新一根K棒的 close，並視需要擴張 high/low
        k.loc[last_idx, "close"] = rt_price
        if realtime_info["high"] is not None:
            k.loc[last_idx, "high"] = max(k.loc[last_idx, "high"], realtime_info["high"], rt_price)
        else:
            k.loc[last_idx, "high"] = max(k.loc[last_idx, "high"], rt_price)
        if realtime_info["low"] is not None:
            k.loc[last_idx, "low"] = min(k.loc[last_idx, "low"], realtime_info["low"], rt_price)
        else:
            k.loc[last_idx, "low"] = min(k.loc[last_idx, "low"], rt_price)

for w in (5, 10, 20):
    k[f"MA{w}"] = k["close"].rolling(w).mean()

kd = calc_kd(k) if show_kd else None
macd = calc_macd(k["close"]) if show_macd else None

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

times = k["date"].dt.strftime("%Y-%m-%d").tolist()

ex_markers = []
ex_details = []
if show_exdiv:
    ed = safe_fetch("TaiwanStockDividendResult", "除權息資料")
    if not ed.empty and "date" in ed.columns:
        ed["date"] = pd.to_datetime(ed["date"])
        for _, r in ed.sort_values("date").iterrows():
            idx = int(k["date"].searchsorted(r["date"]))
            if idx >= len(k):
                continue

            def g(col):
                v = r.get(col)
                return "-" if v is None or pd.isna(v) else v

            ex_markers.append(
                {"time": times[idx], "position": "aboveBar", "color": "#ab47bc", "shape": "circle", "text": "除"}
            )
            ex_details.append(
                {
                    "日期": r["date"].strftime("%Y-%m-%d"),
                    "除權息前收盤": g("before_price"),
                    "參考價": g("after_price"),
                    "權值＋息值": g("stock_and_cache_dividend"),
                }
            )

last_close = float(daily["close"].iloc[-1])
with st.sidebar:
    st.markdown("---")
    st.subheader("🧮 開單盈虧比")
    direction = st.radio("方向", ["開多", "開空"], horizontal=True)
    entry_default = realtime_info["price"] if realtime_info else last_close
    entry = st.number_input("進場價", value=entry_default, step=0.05, format="%.2f", key=f"entry_{stock_id}")
    if direction == "開多":
        def_stop, def_target = entry_default * 0.95, entry_default * 1.10
    else:
        def_stop, def_target = entry_default * 1.05, entry_default * 0.90
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
    sell_value = (exit_price if is_long else entry) * shares
    tax = sell_value * tax_rate
    return gross - fees - tax


# ---------------------------------------------------------------- 組圖表資料
candle_data = [
    {"time": t, "open": float(o), "high": float(h), "low": float(l), "close": float(c)}
    for t, o, h, l, c in zip(times, k["open"], k["high"], k["low"], k["close"])
]

candle_options = {
    "upColor": RED,
    "downColor": GREEN,
    "borderVisible": False,
    "wickUpColor": RED,
    "wickDownColor": GREEN,
}

price_lines = []
if show_rr_lines and rr_valid:
    price_lines = [
        {"price": entry, "color": "#90a4ae", "lineWidth": 1, "lineStyle": 2, "axisLabelVisible": True, "title": f"進場 {entry:.2f}"},
        {"price": stop, "color": "#ef5350", "lineWidth": 1, "lineStyle": 2, "axisLabelVisible": True, "title": f"停損 {stop:.2f}"},
        {"price": target, "color": "#26a69a", "lineWidth": 1, "lineStyle": 2, "axisLabelVisible": True, "title": f"目標 {target:.2f}"},
    ]

candlestick_series = {"type": "Candlestick", "data": candle_data, "options": candle_options}
if ex_markers:
    candlestick_series["markers"] = ex_markers
if price_lines:
    candlestick_series["priceLines"] = price_lines

price_series = [candlestick_series]
if show_ma:
    for w, c in zip((5, 10, 20), ("#fb8c00", "#1e88e5", "#8e24aa")):
        price_series.append(line_series(times, k[f"MA{w}"], c, f"MA{w}"))

vol_colors = [RED if c >= o else GREEN for o, c in zip(k["open"], k["close"])]
volume_data = [{"time": t, "value": float(v) / 1000, "color": col} for t, v, col in zip(times, k["volume"], vol_colors)]
volume_series = [{"type": "Histogram", "data": volume_data, "options": {"priceFormat": {"type": "volume"}}}]

charts = [
    {"chart": base_chart_options(420), "series": price_series},
    {"chart": base_chart_options(130), "series": volume_series},
]

if kd is not None:
    kd_series = [line_series(times, kd["K"], "#fb8c00", "K"), line_series(times, kd["D"], "#1e88e5", "D")]
    charts.append({"chart": base_chart_options(150), "series": kd_series})

if macd is not None:
    osc_colors = [RED if v >= 0 else GREEN for v in macd["OSC"].fillna(0)]
    osc_data = [{"time": t, "value": float(v), "color": c} for t, v, c in zip(times, macd["OSC"], osc_colors)]
    macd_series = [
        {"type": "Histogram", "data": osc_data, "options": {"title": "柱狀體"}},
        line_series(times, macd["DIF"], "#1e88e5", "DIF"),
        line_series(times, macd["MACD"], "#fb8c00", "MACD"),
    ]
    charts.append({"chart": base_chart_options(150), "series": macd_series})

if inst is not None:
    inst_series = [
        line_series(times, inst["外資"], "#1e88e5", "外資", width=1.6),
        line_series(times, inst["投信"], "#fb8c00", "投信", width=1.6),
        line_series(times, inst["自營商"], "#8e24aa", "自營商", width=1.6),
    ]
    charts.append({"chart": base_chart_options(150), "series": inst_series})


# ---------------------------------------------------------------- 自動刷新的圖表區塊
@st.fragment(run_every=refresh_sec if auto_refresh else None)
def render_chart_section():
    if use_realtime:
        rt = fetch_realtime_price(stock_id, market_code)
        if rt:
            prev = rt["prev_close"]
            chg = rt["price"] - prev if prev else None
            chg_pct = (chg / prev * 100) if (chg is not None and prev) else None
            c1, c2, c3 = st.columns([1, 1, 2])
            c1.metric(
                f"{stock_id} 即時價",
                f"{rt['price']:.2f}",
                f"{chg:+.2f} ({chg_pct:+.2f}%)" if chg is not None else None,
            )
            c2.metric("今日成交量", f"{rt['volume']:,.0f} 張" if rt["volume"] else "-")
            c3.caption(f"資料來源：證交所即時行情　更新時間：{dt.datetime.now().strftime('%H:%M:%S')}")
        else:
            st.caption("目前查無即時報價（可能休市中，或此代碼/市場別設定不符）。")

    st.caption(
        "🖱️ 拖曳平移、滾輪縮放、滑鼠移到圖上會顯示十字線與對應數值；各面板時間軸同步連動。"
        + (f"　🔄 每 {refresh_sec} 秒自動刷新一次。" if auto_refresh else "")
    )
    renderLightweightCharts(charts, key="multipane")


render_chart_section()

if inst is not None:
    st.caption(f"三大法人為「最近 {int(n)} 個交易日」合計的買賣超（張），與 K 棒同一期間；最新一天的資料可能尚未公布。")

if ex_details:
    with st.expander("查看除權息明細"):
        st.dataframe(pd.DataFrame(ex_details), width="stretch")

# ---------------------------------------------------------------- 盈虧比結果
st.subheader(f"🧮 {direction} 盈虧比（{stock_id}，{int(lots)} 張）")
if not rr_valid:
    if is_long:
        st.warning("開多需要：停損價 < 進場價 < 目標價，請調整側邊欄的價格。")
    else:
        st.warning("開空需要：目標價 < 進場價 < 停損價，請調整側邊欄的價格。")
else:
    profit = net_pnl(target)
    loss = -net_pnl(stop)
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
