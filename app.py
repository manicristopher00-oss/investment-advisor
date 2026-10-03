import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from openai import OpenAI

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Investment Advisor EU/IT", layout="wide", page_icon="📈")

# --- FUNZIONI DI CALCOLO TECNICO ---
def calculate_rsi(data, window=14):
    delta = data.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=window-1, adjust=False).mean()
    ema_down = down.ewm(com=window-1, adjust=False).mean()
    rs = ema_up / ema_down
    return 100 - (100 / (1 + rs))

def fetch_data(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    hist = ticker.history(period="6mo")
    if hist.empty:
        return None, None
    
    # Calcolo indicatori
    hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
    hist['SMA_200'] = hist['Close'].rolling(window=200).mean()
    hist['RSI'] = calculate_rsi(hist['Close'])
    
    info = ticker.info
    return hist, info

# --- GESTIONE STATO (WATCHLIST) ---
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []

# --- SIDEBAR: IMPOSTAZIONI E WATCHLIST ---
with st.sidebar:
    st.header("⚙️ Impostazioni")
    # Cerca la chiave API nei secrets di Streamlit, altrimenti permette l'inserimento manuale
    api_key_input = st.text_input("OpenAI API Key", type="password", 
                                  value=st.secrets.get("OPENAI_API_KEY", ""),
                                  help="Inserisci qui la tua chiave per i suggerimenti AI")
    
    st.markdown("---")
    st.header("⭐ Watchlist")
    if st.session_state.watchlist:
        for item in st.session_state.watchlist:
            st.write(f"- **{item}**")
    else:
        st.info("Watchlist vuota.")

# --- UI PRINCIPALE ---
st.title("📈 Investment Advisor EU/IT")
st.markdown("Analizza asset finanziari europei, italiani e globali con l'aiuto dell'Intelligenza Artificiale.")

# Input dell'utente
col1, col2 = st.columns([2, 1])
with col1:
    symbol = st.text_input("Inserisci il Simbolo (es. ENI.MI, AAPL, ^GDAXI, BTC-USD):", "ENI.MI").upper()
with col2:
    market = st.selectbox("Mercato di riferimento", ["Italia (Borsa Italiana)", "Europa", "Globale", "Crypto"])

if st.button("Analizza 🚀", type="primary"):
    with st.spinner(f"Recupero dati per {symbol}..."):
        hist, info = fetch_data(symbol)
        
        if hist is None:
            st.error(f"Impossibile trovare dati per il simbolo {symbol}. Verifica che sia corretto.")
        else:
            # Layout Risultati
            current_price = hist['Close'].iloc[-1]
            prev_price = hist['Close'].iloc[-2]
            daily_change = ((current_price - prev_price) / prev_price) * 100
            
            st.subheader(f"Risultati per: {info.get('shortName', symbol)}")
            
            # Metriche principali
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Prezzo Attuale", f"{current_price:.2f}", f"{daily_change:.2f}%")
            m2.metric("RSI (14g)", f"{hist['RSI'].iloc[-1]:.1f}")
            m3.metric("P/E Ratio", info.get('trailingPE', 'N/A'))
            m4.metric("Div. Yield", f"{info.get('dividendYield', 0)*100:.2f}%" if info.get('dividendYield') else "N/A")

            # Grafico interattivo Plotly
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name='Prezzo', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_50'], mode='lines', name='SMA 50', line=dict(color='orange', dash='dash')))
            fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_200'], mode='lines', name='SMA 200', line=dict(color='red', dash='dash')))
            fig.update_layout(title="Andamento 6 Mesi", xaxis_title="Data", yaxis_title="Prezzo", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            
            # ALERT SIMULATO (Logica di base)
            rsi_val = hist['RSI'].iloc[-1]
            if not np.isnan(rsi_val):
                if rsi_val < 30:
                    st.warning("🔔 **ALERT SIMULATO:** RSI in zona Ipervenduto (<30). Possibile opportunità di acquisto.")
                elif rsi_val > 70:
                    st.warning("🔔 **ALERT SIMULATO:** RSI in zona Ipercomprato (>70). Possibile rischio correzione.")

            # Pulsante Watchlist
            if symbol not in st.session_state.watchlist:
                if st.button("➕ Aggiungi alla Watchlist"):
                    st.session_state.watchlist.append(symbol)
                    st.rerun()

            # --- ANALISI AI ---
            st.markdown("---")
            st.subheader("🤖 Raccomandazione AI")
            
            if api_key_input:
                try:
                    client = OpenAI(api_key=api_key_input)
                    prompt = f"""
                    Agisci come un consulente finanziario esperto. Analizza questo asset: {symbol}.
                    Dati attuali:
                    - Prezzo: {current_price:.2f}
                    - RSI (14g): {hist['RSI'].iloc[-1]:.2f}
                    - SMA 50: {hist['SMA_50'].iloc[-1]:.2f}
                    - SMA 200: {hist['SMA_200'].iloc[-1]:.2f}
                    
                    Fornisci un'analisi sintetica (massimo 4 frasi) e un suggerimento operativo chiaro (Buy/Hold/Sell) 
                    basato sull'analisi tecnica di questi parametri.
                    """
                    
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=150
                    )
                    st.info(response.choices[0].message.content)
                except Exception as e:
                    st.error(f"Errore con l'API OpenAI: {str(e)}")
            else:
                st.warning("Inserisci una chiave API OpenAI nelle impostazioni (sidebar) per generare l'analisi AI.")
