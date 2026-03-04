import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as GO
from google import genai
import datetime
from datetime import timedelta
from bs4 import BeautifulSoup
import requests
import feedparser

st.set_page_config(page_title="USD/JPY 予測アプリ", layout="wide")
st.title("USD/JPY AI予測アプリ (Gemini)")

# Session State Initialization
if "last_price" not in st.session_state:
    st.session_state.last_price = None
if "last_updated" not in st.session_state:
    st.session_state.last_updated = None
if "news_text" not in st.session_state:
    st.session_state.news_text = ""
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

# Function to fetch current price
def fetch_current_price():
    try:
        ticker = yf.Ticker("JPY=X")
        # Fast historical fetch for latest price
        data = ticker.history(period="1d", interval="1m")
        if not data.empty:
            return data['Close'].iloc[-1]
    except Exception as e:
        st.error(f"価格レート取得エラー: {e}")
    return None

# Initial Load
if st.session_state.last_price is None:
    initial_price = fetch_current_price()
    if initial_price:
        st.session_state.last_price = initial_price
        st.session_state.last_updated = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Sidebar Settings
st.sidebar.header("設定")

# View Selector (PC vs Smartphone)
view_mode = st.sidebar.radio("レイアウト表示", ["PC (標準)", "スマートフォン (文字小さめ)"])

# Apply CSS if Smartphone view is selected
if "スマートフォン" in view_mode:
    st.markdown("""
        <style>
        html, body, [class*="css"] {
            font-size: 13px !important;
        }
        h1 { font-size: 1.8rem !important; }
        h2 { font-size: 1.5rem !important; }
        h3 { font-size: 1.2rem !important; }
        /* Reduce padding/margins for compact view */
        .st-emotion-cache-16txtl3 { padding: 2rem 1rem !important; }
        </style>
    """, unsafe_allow_html=True)

api_key = st.sidebar.text_input("Gemini API Key", type="password")

# Timeframe Selection
st.sidebar.subheader("予測条件")
timeframe_options = {
    "1時間後": ("1d", "15m", "USD/JPY - 24 Hours"),
    "4時間後": ("3d", "1h", "USD/JPY - 3 Days"),
    "8時間後": ("5d", "1h", "USD/JPY - 5 Days"),
    "1日後": ("6mo", "1d", "USD/JPY - 6 Months"),
    "1週間後": ("1y", "1d", "USD/JPY - 1 Year")
}
selected_timeframe = st.sidebar.selectbox("予測する時間枠", list(timeframe_options.keys()))
st.session_state.current_period, st.session_state.current_interval, st.session_state.chart_title = timeframe_options[selected_timeframe]
st.session_state.selected_timeframe = selected_timeframe

# Display Current Price
st.subheader("USD/JPY 最新レート")
if st.session_state.last_price:
    st.metric(label=f"取得日時: {st.session_state.last_updated}", value=f"{st.session_state.last_price:.3f} 円")
else:
    st.write("レート取得中...")

# --- Helper Functions ---

def fetch_recent_news():
    """Fetches Forex news from the past 8 hours via Google News RSS for better volume."""
    news_items = []
    try:
        url = 'https://news.google.com/rss/search?q=USD%2FJPY+OR+%E3%83%89%E3%83%AB%E5%86%86&hl=ja&gl=JP&ceid=JP:ja'
        feed = feedparser.parse(url)
        now = datetime.datetime.now(datetime.timezone.utc)
        eight_hours_ago = now - timedelta(hours=8)
        
        for entry in feed.entries:
            try:
                # Parse RSS pubDate e.g. "Wed, 04 Mar 2026 12:06:33 GMT"
                pub_time = datetime.datetime.strptime(entry.published, '%a, %d %b %Y %H:%M:%S %Z')
                pub_time = pub_time.replace(tzinfo=datetime.timezone.utc)
            except ValueError:
                continue
                
            if pub_time >= eight_hours_ago:
                news_items.append(f"- [{pub_time.strftime('%H:%M')}] {entry.title}")
                
    except Exception as e:
        st.warning(f"ニュースの取得に失敗しました: {e}")
        
    if not news_items:
        news_items.append("(過去8時間に関連するニュースは見つかりませんでした。)")
        
    # Gemini Context Limit (prevent too massive prompts if news spikes)
    return "\n".join(news_items[:50])

def create_chart(period, interval, title):
    ticker = yf.Ticker("JPY=X")
    data = ticker.history(period=period, interval=interval)
    if data.empty:
        return None
        
    fig = GO.Figure(data=[GO.Candlestick(x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'])])
    fig.update_layout(title=title, yaxis_title='USD/JPY', xaxis_title='Time', template='plotly_dark', margin=dict(l=20, r=20, t=40, b=20))
    # Disable rangeslider for cleaner view
    fig.update_layout(xaxis_rangeslider_visible=False)
    
    # Hide weekends to remove empty gaps in the chart
    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"]) # hide weekends
        ]
    )
    return fig

def predict_with_gemini(api_key, current_price, news_text, timeframe):
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        return f"Gemini API 初期化エラー: {e}"
    
    # Try different models in case of version/region restrictions
    models_to_try = ['gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-1.5-flash', 'gemini-1.5-pro']
    
    prompt = f"""
    あなたはプロのFXアナリストです。
    現在のUSD/JPYの価格は {current_price:.3f} 円です。
    以下は、過去8時間のUSD/JPYおよび経済に関連するニュースの見出しです：
    
    {news_text}
    
    この情報と現在の市場の一般的なトレンドに基づいて、USD/JPYが「{timeframe}」に「上昇(UP)」するか「下降(DOWN)」するかを予測してください。
    また、その簡潔な理由も添えてください。
    
    フォーマット：
    {timeframe}の予測：[上昇 or 下降] - 理由...
    """
    
    last_error = ""
    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            if response.text:
                return f"【使用モデル: {model_name}】\n\n{response.text}"
            else:
                return f"Gemini APIエラー: レスポンスが空です（安全フィルター制限等）。モデル: {model_name}"
        except Exception as e:
            last_error = str(e)
            # 404 error means model not found/supported, continue to next
            if "404" in last_error or "not found" in last_error.lower():
                continue
            else:
                # Other errors like Quota/Permission should stop immediately
                return f"Gemini APIエラー ({model_name} 実行時): {last_error}"
                
    return f"Gemini APIエラー: 利用可能なモデルが見つかりませんでした。詳細: {last_error}"

# --- Main Action ---

if st.button("最新値とニュースを読み込んで予測 (Update & Predict)"):
    if not api_key:
        st.error("左側のサイドバーからGemini API Keyを入力してください。")
    else:
        with st.spinner("データを取得・予測中..."):
            # Update Price
            new_price = fetch_current_price()
            if new_price:
                st.session_state.last_price = new_price
                st.session_state.last_updated = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Fetch News
            st.session_state.news_text = fetch_recent_news()
            
            # Predict
            st.session_state.prediction_result = predict_with_gemini(api_key, st.session_state.last_price, st.session_state.news_text, selected_timeframe)
        st.rerun()

# --- Display Sections ---

if st.session_state.prediction_result:
    st.markdown("---")
    st.subheader("🤖 Gemini予測結果")
    if "Gemini APIエラー" in st.session_state.prediction_result:
        st.error(st.session_state.prediction_result)
    else:
        st.info(st.session_state.prediction_result)
    
    with st.expander("参考にした過去8時間のニュース"):
        st.text(st.session_state.news_text)

st.markdown("---")
st.subheader(f"📈 チャート: {st.session_state.selected_timeframe}予測用")

fig = create_chart(period=st.session_state.current_period, 
                   interval=st.session_state.current_interval, 
                   title=st.session_state.chart_title)
if fig: 
    st.plotly_chart(fig, use_container_width=True)
