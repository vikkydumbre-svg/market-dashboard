"""India Market Dashboard: global-news sentiment + Nifty price/Supertrend in one view.
Local:  py -m streamlit run app.py
Cloud:  deploy on Streamlit Community Cloud (see steps given in chat).
"""
import os, re, json, math, hmac, hashlib, sqlite3
from datetime import datetime, timezone
from urllib.parse import quote

import feedparser
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

IST = "Asia/Kolkata"
DB = "sentiment.db"
MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-5-5")
MAX_NEW_PER_CYCLE = 60
SYMBOLS = {"Nifty 50": "^NSEI", "Bank Nifty": "^NSEBANK"}

st.set_page_config(page_title="India Market Dashboard", layout="wide")

# ---------------------------------------------------------------- secrets + password
def secret(k):
    try:
        return st.secrets[k]
    except Exception:
        return os.getenv(k)

if secret("ANTHROPIC_API_KEY"):
    os.environ["ANTHROPIC_API_KEY"] = secret("ANTHROPIC_API_KEY")

_pw = secret("APP_PASSWORD")
if _pw and not st.session_state.get("ok"):
    st.title("🔒 India Market Dashboard")
    x = st.text_input("Password", type="password")
    if x and hmac.compare_digest(x, _pw):
        st.session_state.ok = True
        st.rerun()
    elif x:
        st.error("Wrong password")
    st.stop()

# ---------------------------------------------------------------- news sources
def gnews(q):
    return f"https://news.google.com/rss/search?q={quote(q + ' when:1d')}&hl=en-IN&gl=IN&ceid=IN:en"

FEEDS = {
    "Economic Times": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "Moneycontrol": "https://www.moneycontrol.com/rss/marketreports.xml",
    "CNBC": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664",
    "GNews: Nifty/Sensex": gnews("Nifty OR Sensex OR \"Bank Nifty\""),
    "GNews: Fed/rates": gnews("Federal Reserve OR \"interest rates\" OR inflation"),
    "GNews: Crude/OPEC": gnews("crude oil OR OPEC OR Brent"),
    "GNews: FII/Rupee": gnews("FII OR FPI OR rupee OR RBI"),
    "GNews: China/Geopolitics": gnews("China economy OR war OR sanctions OR tariffs"),
    "GNews: Global markets": gnews("Wall Street OR \"Asian markets\" OR \"global markets\""),
    "GNews: India policy": gnews("India budget OR SEBI OR India GDP OR India inflation"),
}

# ---------------------------------------------------------------- storage
def db():
    c = sqlite3.connect(DB)
    c.execute("""CREATE TABLE IF NOT EXISTS news(
        id TEXT PRIMARY KEY, title TEXT, source TEXT, link TEXT,
        published TEXT, score REAL, relevance REAL, reason TEXT, method TEXT)""")
    return c

def known_ids():
    with db() as c:
        return {r[0] for r in c.execute("SELECT id FROM news")}

def save(rows):
    with db() as c:
        c.executemany("INSERT OR IGNORE INTO news VALUES (?,?,?,?,?,?,?,?,?)", rows)

def load():
    with db() as c:
        df = pd.read_sql("SELECT * FROM news", c)
    if df.empty:
        return df
    df["published"] = pd.to_datetime(df["published"], utc=True)
    return df[df.published >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=7)].sort_values("published")

@st.cache_data(ttl=240)
def fetch_news():
    items, seen = [], set()
    for name, url in FEEDS.items():
        try:
            feed = feedparser.parse(url)
        except Exception:
            continue
        for e in feed.entries[:40]:
            title = re.sub(r"\s+", " ", e.get("title", "")).strip()
            if not title:
                continue
            nid = hashlib.md5(re.sub(r"\W+", "", title.lower())[:80].encode()).hexdigest()
            if nid in seen:
                continue
            seen.add(nid)
            p = e.get("published_parsed") or e.get("updated_parsed")
            ts = datetime(*p[:6], tzinfo=timezone.utc) if p else datetime.now(timezone.utc)
            if (datetime.now(timezone.utc) - ts).total_seconds() > 48 * 3600:
                continue
            items.append(dict(id=nid, title=title, source=name, link=e.get("link", ""), published=ts))
    return items

# ---------------------------------------------------------------- scorers
POS = {"rate cut": 1, "cuts rates": 1, "ceasefire": 1, "oil falls": .8, "crude falls": .8,
       "oil prices fall": .8, "fii buying": .8, "inflows": .5, "rally": .6, "surge": .4,
       "record high": .6, "trade deal": .7, "stimulus": .6, "soft landing": .5,
       "upgrade": .4, "inflation eases": .7, "cools": .3, "gains": .3, "rebound": .5}
NEG = {"rate hike": -1, "hikes rates": -1, "war": -.7, "attack": -.6, "oil surges": -.8,
       "crude surges": -.8, "oil prices rise": -.7, "fii selling": -.8, "outflows": -.5,
       "selloff": -.8, "sell-off": -.8, "crash": -.9, "recession": -.8, "tariff": -.6,
       "sanctions": -.5, "downgrade": -.5, "inflation rises": -.7, "plunge": -.7,
       "slump": -.6, "falls": -.3, "default": -.6, "bank failure": -.9, "tumbles": -.6}
REL = ["nifty", "sensex", "india", "rbi", "fed", "federal reserve", "crude", "oil", "opec",
       "fii", "fpi", "rupee", "dollar", "china", "tariff", "war", "sanction", "wall street",
       "inflation", "interest rate", "treasury", "yield", "gold", "sebi", "budget", "gdp",
       "bank", "asian markets", "middle east", "iran", "russia", "ukraine", "trade"]

def score_keyword(titles):
    out = []
    for t in titles:
        low = t.lower()
        s = sum(w for k, w in {**POS, **NEG}.items() if k in low)
        rel = 0.7 if any(k in low for k in REL) else 0.0
        out.append(dict(score=math.tanh(s), relevance=rel, reason="keyword match"))
    return out

def score_claude(titles):
    import anthropic
    client = anthropic.Anthropic()
    out = []
    for i in range(0, len(titles), 25):
        chunk = titles[i:i + 25]
        numbered = "\n".join(f"{j}. {t}" for j, t in enumerate(chunk))
        prompt = f"""You are an Indian equity-market analyst. For each headline, judge the likely
near-term effect on the Indian stock market (Nifty/Bank Nifty). Consider crude oil (India is a net
importer), US Fed/dollar/rupee, FII flows, China/geopolitics, global risk sentiment, Indian policy.

Return ONLY a JSON array, one object per headline:
{{"i": 0, "score": -1.0 to 1.0, "relevance": 0.0 to 1.0, "reason": "max 12 words"}}
score: -1 very negative, 0 neutral, +1 very positive for Indian markets.
relevance: how directly this affects Indian markets (0 = irrelevant).

Headlines:
{numbered}"""
        try:
            msg = client.messages.create(model=MODEL, max_tokens=4000,
                                         messages=[{"role": "user", "content": prompt}])
            text = msg.content[0].text
            by_i = {d["i"]: d for d in json.loads(text[text.index("["): text.rindex("]") + 1])}
        except Exception as ex:
            st.warning(f"Claude scoring failed for a batch ({ex}); used keywords instead.")
            out += score_keyword(chunk)
            continue
        for j in range(len(chunk)):
            d = by_i.get(j, {})
            out.append(dict(score=max(-1, min(1, float(d.get("score", 0)))),
                            relevance=max(0, min(1, float(d.get("relevance", 0)))),
                            reason=str(d.get("reason", ""))))
    return out

def update_news(mode):
    seen = known_ids()
    new = [n for n in fetch_news() if n["id"] not in seen]
    new = sorted(new, key=lambda n: n["published"], reverse=True)[:MAX_NEW_PER_CYCLE]
    if not new:
        return
    titles = [n["title"] for n in new]
    scores = score_claude(titles) if mode == "Claude API" else score_keyword(titles)
    save([(n["id"], n["title"], n["source"], n["link"], n["published"].isoformat(),
           s["score"], s["relevance"], s["reason"], mode) for n, s in zip(new, scores)])

# ---------------------------------------------------------------- price + indicators
@st.cache_data(ttl=60)
def get_candles(symbol, cm):
    try:
        raw = yf.download(symbol, period="5d", interval="1m", progress=False, auto_adjust=False)
    except Exception:
        return pd.DataFrame()
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.index = raw.index.tz_convert(IST) if raw.index.tz else raw.index.tz_localize(IST)
    df = (raw.resample(f"{cm}min")
          .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna())
    return df[df.index + pd.Timedelta(minutes=cm) <= pd.Timestamp.now(tz=IST)]  # closed candles only

def supertrend(df, period, mult):
    h, l, c = df.High.to_numpy(), df.Low.to_numpy(), df.Close.to_numpy()
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum.reduce([h - l, abs(h - pc), abs(l - pc)])
    atr = pd.Series(tr).ewm(alpha=1 / period, adjust=False).mean().to_numpy()
    hl2 = (h + l) / 2
    ub, lb = hl2 + mult * atr, hl2 - mult * atr
    up, lo, d = ub.copy(), lb.copy(), np.ones(len(c), int)
    for i in range(1, len(c)):
        up[i] = ub[i] if ub[i] < up[i - 1] or c[i - 1] > up[i - 1] else up[i - 1]
        lo[i] = lb[i] if lb[i] > lo[i - 1] or c[i - 1] < lo[i - 1] else lo[i - 1]
        if d[i - 1] == -1 and c[i] > up[i - 1]:
            d[i] = 1
        elif d[i - 1] == 1 and c[i] < lo[i - 1]:
            d[i] = -1
        else:
            d[i] = d[i - 1]
    return pd.Series(d, index=df.index), pd.Series(np.where(d == 1, lo, up), index=df.index)

def sentiment_at(news, times, halflife_h):
    n = news[news.relevance >= 0.3]
    vals = []
    for t in times:
        d = n[n.published <= t]
        if d.empty:
            vals.append(0.0)
            continue
        age = (t - d.published).dt.total_seconds() / 3600
        w = d.relevance * 0.5 ** (age / halflife_h)
        vals.append(float((d.score * w).sum() / (w.sum() + 0.5)))
    return pd.Series(vals, index=times)

def label(v):
    return ("Strong Positive" if v > .5 else "Positive" if v > .15 else
            "Neutral" if v > -.15 else "Negative" if v > -.5 else "Strong Negative")

def move_after(c, t, cm, mins=30):
    i = c.index.searchsorted(t)
    if i >= len(c):
        return None
    j = c.index.searchsorted(c.index[i] + pd.Timedelta(minutes=max(mins - cm, 0)))
    if j >= len(c):
        return None  # not enough time has passed yet
    return (c.Close.iloc[j] / c.Open.iloc[i] - 1) * 100

# ---------------------------------------------------------------- sidebar
has_key = bool(os.getenv("ANTHROPIC_API_KEY"))
with st.sidebar:
    name = st.selectbox("Index", list(SYMBOLS))
    cm = st.selectbox("Candle (minutes)", [3, 5, 15], index=0)
    days = st.slider("Days to show", 1, 5, 2)
    mode = st.radio("News scoring", ["Claude API", "Keywords (free)"], index=0 if has_key else 1)
    if mode == "Claude API" and not has_key:
        st.error("No API key set. Using keywords.")
        mode = "Keywords (free)"
    st_p = st.number_input("Supertrend period", 5, 30, 10)
    st_m = st.number_input("Supertrend multiplier", 1.0, 6.0, 3.0, 0.5)
    halflife = st.slider("News half-life (hours)", 1, 12, 3)
    refresh = st.slider("Auto-refresh (minutes)", 1, 30, 3)
    auto = st.checkbox("Auto-refresh", value=True)
    st.caption("Not financial advice. Data may be delayed; confirm on your broker chart before trading.")

st.title("📈 India Market Dashboard")

@st.fragment(run_every=(refresh * 60 if auto else None))
def dashboard():
    with st.spinner("Updating..."):
        update_news(mode)
    news = load()
    candles = get_candles(SYMBOLS[name], cm)
    if candles.empty:
        st.warning("Could not get price data from Yahoo Finance right now. Showing news only.")
    if news.empty:
        st.info("No news stored yet.")
        return

    if not candles.empty and len(candles) > st_p + 5:
        candles = candles.copy()
        candles["dir"], candles["st"] = supertrend(candles, st_p, st_m)
        keep = sorted(set(candles.index.date))[-days:]
        view = candles[[d in keep for d in candles.index.date]]
        sent = sentiment_at(news, view.index, halflife)
        cur_s, d_now = sent.iloc[-1], int(view.dir.iloc[-1])
        flips = view.index[view.dir.ne(view.dir.shift())][1:]
        since = flips[-1].strftime("%d %b %H:%M") if len(flips) else "start of window"
        prev_close = candles[candles.index.date < view.index[-1].date()].Close
        chg = (view.Close.iloc[-1] / prev_close.iloc[-1] - 1) * 100 if len(prev_close) else 0.0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(name, f"{view.Close.iloc[-1]:,.2f}", f"{chg:+.2f}% vs prev close")
        m2.metric("Supertrend", "BUY 🟢" if d_now == 1 else "SELL 🔴", f"since {since}", delta_color="off")
        m3.metric("News sentiment", f"{cur_s:+.2f}", label(cur_s))
        m4.metric("Last candle", view.index[-1].strftime("%d %b %H:%M"))

        if d_now == 1 and cur_s > .15:
            st.success("Aligned bullish: Supertrend is BUY and news sentiment is positive.")
        elif d_now == -1 and cur_s < -.15:
            st.error("Aligned bearish: Supertrend is SELL and news sentiment is negative.")
        else:
            st.warning("Mixed signals: price trend and news sentiment don't agree. Be cautious.")

        x = view.index.tz_localize(None)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28],
                            vertical_spacing=0.03)
        fig.add_trace(go.Candlestick(x=x, open=view.Open, high=view.High, low=view.Low,
                                     close=view.Close, name=name), row=1, col=1)
        for dv, col, nm in [(1, "#26a69a", "Supertrend up"), (-1, "#ef5350", "Supertrend down")]:
            fig.add_trace(go.Scatter(x=x, y=view.st.where(view.dir == dv), mode="lines",
                                     line=dict(color=col, width=2), name=nm), row=1, col=1)
        ev = news[(news.relevance >= .5) & (news.score.abs() >= .4)]
        pos = view.index.searchsorted(ev.published)
        ok = pos < len(view)
        ev, pos = ev[ok], pos[ok]
        for sgn, col, sym in [(1, "#00c853", "triangle-up"), (-1, "#ff1744", "triangle-down")]:
            m = (np.sign(ev.score) == sgn).to_numpy()
            if m.any():
                fig.add_trace(go.Scatter(
                    x=x[pos[m]], y=view.High.iloc[pos[m]].to_numpy() * 1.0006, mode="markers",
                    marker=dict(symbol=sym, size=11, color=col), name="Major news",
                    text=[t[:110] for t in ev.title[m]], hovertemplate="%{text}<extra></extra>"),
                    row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=sent.values, mode="lines", fill="tozeroy",
                                 line=dict(width=2), name="Sentiment"), row=2, col=1)
        fig.add_hline(y=0, line_width=1, line_color="gray", row=2, col=1)
        fig.update_yaxes(range=[-1, 1], title_text="Sentiment", row=2, col=1)
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"]),
                                      dict(bounds=[15.5, 9.25], pattern="hour")])
        fig.update_layout(height=680, hovermode="x unified", xaxis_rangeslider_visible=False,
                          margin=dict(t=10, b=10), legend=dict(orientation="h", y=1.04))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("▲/▼ markers = major news (shown on the candle after it was published). "
                   "Times are IST; nights and weekends are hidden.")
    else:
        cur_s = sentiment_at(news, [pd.Timestamp.now(tz="UTC")], halflife).iloc[-1]
        st.metric("News sentiment", f"{cur_s:+.2f}", label(cur_s))

    st.subheader("Headlines and what the market did next")
    show = news[news.relevance >= .3].sort_values("published", ascending=False).head(50).copy()
    show["Impact"] = show.score.apply(lambda s: "🟢" if s > .15 else "🔴" if s < -.15 else "⚪")
    show["Time"] = show.published.dt.tz_convert(IST).dt.strftime("%d %b %H:%M")
    if not candles.empty:
        show["move"] = [move_after(candles, t, cm) for t in show.published.dt.tz_convert(IST)]
    else:
        show["move"] = None
    st.dataframe(show[["Time", "Impact", "score", "move", "title", "reason", "source", "link"]],
                 hide_index=True, use_container_width=True,
                 column_config={"link": st.column_config.LinkColumn("Link"),
                                "score": st.column_config.NumberColumn("Score", format="%.2f"),
                                "move": st.column_config.NumberColumn(f"{name} 30m after %", format="%+.2f"),
                                "title": st.column_config.TextColumn("Headline", width="large")})
    st.caption(f"Updated {datetime.now(tz=None).strftime('%H:%M:%S')} (server time)")

dashboard()
