from flask import Flask, render_template, request, session, redirect, url_for
import yfinance as yf
import pandas as pd
import numpy as np
from langchain_community.llms import CTransformers
import logging
import os
from datetime import datetime, timedelta
import time
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb
import io
import traceback
import tempfile
import uuid
from pathlib import Path

app = Flask(__name__)
app.secret_key = "your_secret_key"
app.permanent_session_lifetime = timedelta(minutes=30)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_PATH = os.path.abspath(r"C:\Users\91636\Desktop\chat_CSV\llama-2-7b-chat.ggmlv3.q8_0.bin")
MAX_TOKENS = 512
CHUNK_SIZE = 400

# ================== Helper Functions ==================

def cleanup_old_files():
    """Remove temporary files older than 1 hour"""
    temp_dir = Path(tempfile.gettempdir())
    for file in temp_dir.glob("stock_data_*.json"):
        if file.stat().st_mtime < (time.time() - 3600):
            try:
                file.unlink()
            except Exception as e:
                logger.error(f"Cleanup error for {file}: {e}")

def load_llm():
    """Load the Llama model with optimized settings."""
    try:
        # Verify the model path exists
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model file not found at {MODEL_PATH}")
            raise FileNotFoundError(f"Llama model file not found at {MODEL_PATH}")

        return CTransformers(
            model=MODEL_PATH,
            model_type="llama",
            max_new_tokens=128,
            temperature=0.1,
            context_length=MAX_TOKENS,
            batch_size=1,
            threads=4,
        )
    except Exception as e:
        logger.error(f"Error loading model: {e}\n{traceback.format_exc()}")
        return None

def fetch_stock_data(ticker):
    """Fetch stock data with error handling."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="6mo")
        if df.empty:
            return None
        df["SMA_50"] = df["Close"].rolling(window=50).mean()
        df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
        return df
    except Exception as e:
        logger.error(f"Error fetching {ticker}: {e}\n{traceback.format_exc()}")
        return None

def process_csv_data(uploaded_file):
    """Process CSV with robust error handling."""
    try:
        df = pd.read_csv(uploaded_file)
        date_cols = [col for col in df.columns if "date" in col.lower()]
        if date_cols:
            try:
                df[date_cols[0]] = pd.to_datetime(df[date_cols[0]])
                if date_cols[0] in df.columns:
                    df.set_index(date_cols[0], inplace=True)
            except Exception as date_err:
                logger.error(f"Error converting date column: {date_err}")
        if all(col in df.columns for col in ["Close", "Open", "High", "Low"]):
            df["SMA_50"] = df["Close"].rolling(window=50).mean()
            df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
        return df.dropna()
    except Exception as e:
        logger.error(f"CSV processing error: {e}\n{traceback.format_exc()}")
        return None

# ================== Updated Data Handling ==================

def save_data_to_temp(df):
    """Save DataFrame to temporary file and return path"""
    try:
        cleanup_old_files() # Cleanup before creating new files
        temp_dir = Path(tempfile.gettempdir())
        file_id = uuid.uuid4().hex
        filename = temp_dir / f"stock_data_{file_id}.json"
        df.to_json(filename)
        return str(filename)
    except Exception as e:
        logger.error(f"Data save error: {e}")
        return None

def load_data_from_temp():
    """Load DataFrame from temporary file"""
    try:
        if "data_file" not in session:
            return None, "No data loaded"
        filename = Path(session["data_file"])
        if not filename.exists():
            return None, "Data expired or deleted"
        return pd.read_json(filename), None
    except Exception as e:
        logger.error(f"Data load error: {e}")
        return None, str(e)

# ================== Updated Routes ==================

@app.route("/", methods=["GET", "POST"])
def index():
    """Main entry point with data upload handling."""
    if request.method == "POST":
        try:
            # Cleanup previous data file
            if "data_file" in session:
                old_file = Path(session["data_file"])
                if old_file.exists():
                    try:
                        old_file.unlink()
                    except Exception as e:
                        logger.error(f"Cleanup error: {e}")
                session.clear()

            data_source = request.form["data_source"]
            session["data_source"] = data_source

            if data_source == "CSV File":
                file = request.files["file"]
                if not file or file.filename == "":
                    return render_template("index.html", error="No file selected")
                df = process_csv_data(io.StringIO(file.read().decode("utf-8")))
            else: # Stock Ticker
                ticker = request.form["ticker_input"].strip().upper()
                if not ticker:
                    return render_template("index.html", error="Enter a stock ticker")
                df = fetch_stock_data(ticker)

            if df is None:
                return render_template("index.html", error="Data loading failed")

            # Save to temp file and store path in session
            temp_file = save_data_to_temp(df)
            if not temp_file:
                return render_template("index.html", error="Data storage failed")

            session["data_file"] = temp_file
            return render_template("index.html", success="Data loaded successfully!")

        except Exception as e:
            logger.error(f"Index error: {e}\n{traceback.format_exc()}")
            return render_template("index.html", error=f"Error: {str(e)}")

    return render_template("index.html")

def preprocess_data(df):
    """Preprocess data for LSTM model."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df["Close"].values.reshape(-1, 1))
    X_train, y_train = [], []
    for i in range(60, len(scaled_data)):
        X_train.append(scaled_data[i - 60:i, 0])
        y_train.append(scaled_data[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    return X_train, y_train, scaler

def build_lstm_model(input_shape):
    """Build LSTM model for prediction."""
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(input_shape, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

def predict_stock_prices(model_type, df, days):
    """Generate price predictions using selected model."""
    try:
        if 'Close' not in df.columns:
            raise ValueError("The 'Close' column is missing in the DataFrame.")

        if model_type == "LSTM":
            X_train, y_train, scaler = preprocess_data(df)
            model = build_lstm_model(X_train.shape[1])
            model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
            last_60_days = df["Close"][-60:].values.reshape(-1, 1)
            last_60_days_scaled = scaler.transform(last_60_days)
            predictions = []
            current_batch = last_60_days_scaled.reshape((1, 60, 1))

            for _ in range(days):
                next_pred = model.predict(current_batch)[0, 0]
                predictions.append(scaler.inverse_transform([[next_pred]])[0][0])
                current_batch = np.concatenate((current_batch[:, 1:, :], np.array([[[next_pred]]])), axis=1)

        elif model_type == "ARIMA":
            model = ARIMA(df["Close"], order=(5, 1, 0))
            model_fit = model.fit()
            predictions = model_fit.forecast(steps=days)

        elif model_type == "XGBoost":
            model = xgb.XGBRegressor(objective="reg:squarederror")
            X = np.arange(len(df)).reshape(-1, 1)
            model.fit(X, df["Close"])
            future_dates = np.arange(len(df), len(df) + days).reshape(-1, 1)
            predictions = model.predict(future_dates)

        else:
            raise ValueError("Invalid model type selected.")

        return predictions

    except Exception as e:
        logger.error(f"Prediction error: {e}\n{traceback.format_exc()}")
        return None

def generate_buying_advice(predictions, current_price):
    """Generate buying advice based on predictions."""
    predicted_increase = (predictions[-1] - current_price) / current_price
    if predicted_increase > 0.10:
        return f"Strong Buy: Predicted increase of {predicted_increase:.2%}."
    elif 0.05 < predicted_increase <= 0.10:
        return f"Buy: Predicted increase of {predicted_increase:.2%}."
    elif -0.05 <= predicted_increase <= 0.05:
        return "Hold: Little predicted change."
    elif -0.10 <= predicted_increase < -0.05:
        return f"Sell: Predicted decrease of {abs(predicted_increase):.2%}."
    else:
        return f"Strong Sell: Predicted decrease of {abs(predicted_increase):.2%}."

@app.route("/prediction", methods=["GET", "POST"])

def prediction():
    """Prediction route with model selection."""
    df, error = load_data_from_temp()
    if error:
        return render_template("prediction.html", error=error)
    if df is None:
        return render_template("prediction.html", error="No data loaded. Please upload a CSV or enter a ticker on the home page.")
    
    if request.method == "POST":
        try:
            days_to_predict = int(request.form.get("days_to_predict"))
            model_type = request.form.get("model_type")
            predictions = predict_stock_prices(model_type, df, days_to_predict)
            
            if predictions is None:
                return render_template("prediction.html", error="Prediction failed. Check data or model.")
            
            current_price = df["Close"].iloc[-1]
            advice = generate_buying_advice(predictions, current_price)
            
            pred_df = pd.DataFrame({
                "Date": pd.date_range(df.index[-1] + timedelta(days=1), periods=days_to_predict),
                "Predicted Price": predictions,
            })
            
            # Ensure proper date handling
            last_date = df.index[-1]
            if isinstance(last_date, str):
                last_date = pd.to_datetime(last_date)
                
            future_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=days_to_predict
            )
            
            # Update the figure creation
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df.index, 
                y=df["Close"], 
                mode='lines', 
                name='Historical'
            ))
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=predictions,
                mode='lines+markers',
                name='Prediction',
                line=dict(color='orange', dash='dot')
            ))
            fig.update_layout(
                title='Stock Price Prediction',
                xaxis_title='Date',
                yaxis_title='Price',
                hovermode='x unified'
            )
            plot_html = fig.to_html(full_html=False)
            
            return render_template(
                "prediction.html",
                advice=advice,
                pred_table=pred_df.to_html(classes='table table-striped'),
                fig=plot_html
            )
        
        except Exception as e:
            logger.error(f"Prediction POST error: {e}\n{traceback.format_exc()}")
            return render_template("prediction.html", error=f"Prediction error: {str(e)}")

    return render_template("prediction.html")

def create_optimized_prompt(query, data_summary, columns=None):
    """Create a concise prompt that fits within token limits."""
    if columns:
        return f"""Analyze this data with columns {', '.join(columns)}:
{data_summary}
Question: {query}
Answer:"""
    else:
        return f"""Analyze this market data:
{data_summary}
Question: {query}
Answer:"""

def summarize_data(df, query):
    """Create a relevant summary of the data based on the query."""
    summary = []
    try:
        latest = df.iloc[-1]
        summary.append(
            f"Latest values: {', '.join([f'{col}: {latest[col]:.2f}' for col in ['Close', 'SMA_50', 'EMA_50'] if col in df.columns])}"
        )

        if any(word in query.lower() for word in ["trend", "movement", "change"]):
            pct_change = ((df["Close"].iloc[-1] / df["Close"].iloc[0]) - 1) * 100
            summary.append(f"Overall trend: {pct_change:.2f}% change")

        if any(word in query.lower() for word in ["volume", "trading"]):
            avg_volume = df["Volume"].mean() if "Volume" in df.columns else None
            if avg_volume:
                summary.append(f"Average volume: {avg_volume:.0f}")

        if any(word in query.lower() for word in ["high", "low", "range"]):
            summary.append(
                f"Range: High {df['High'].max():.2f}, Low {df['Low'].min():.2f}"
            )

        return "\n".join(summary)[:CHUNK_SIZE]
    except Exception as e:
        logger.error(f"Error in summarize_data: {e}")
        return str(df.describe().round(2).head())[:CHUNK_SIZE]

@app.route("/visualization")
def visualization():
    """Data visualization route with descriptions."""
    df, error = load_data_from_temp()
    if error:
        return render_template("visualization.html", error=error)
    if df is None or df.empty:
        return render_template("visualization.html", error="No data loaded. Please upload a CSV or enter a ticker.")

    try:
        # 📈 Candlestick Chart - Shows the opening, closing, high, and low prices for each day.
        fig_candlestick = go.Figure(data=[go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']
        )])
        fig_candlestick.update_layout(
            title="📈 Candlestick Chart - Price Trends Over Time",
            template="plotly_dark"
        )

        # 📉 Closing Prices Line Chart - Tracks the stock's closing price each day.
        fig_close = px.line(df, x=df.index, y="Close", 
                            title="📉 Closing Prices Over Time - Track Daily Trends",
                            template="plotly_dark")

        # 📊 Trading Volume Chart - Indicates how many shares were traded each day.
        fig_volume = px.bar(df, x=df.index, y="Volume", 
                            title="📊 Trading Volume Over Time - Market Activity",
                            template="plotly_dark")

        # 📉 Moving Average (20-day) - Smooths price trends to highlight direction.
        df["MA20"] = df["Close"].rolling(window=20).mean()
        fig_ma = go.Figure()
        fig_ma.add_trace(go.Scatter(x=df.index, y=df["Close"], 
                                    mode="lines", name="Close Price", line=dict(color='white')))
        fig_ma.add_trace(go.Scatter(x=df.index, y=df["MA20"], 
                                    mode="lines", name="20-Day MA", line=dict(color='orange')))
        fig_ma.update_layout(title="📉 20-Day Moving Average - Identifying Trends", template="plotly_dark")

        # 📊 Bollinger Bands - Helps determine whether stock prices are high or low.
        df["Upper"] = df["MA20"] + (df["Close"].rolling(window=20).std() * 2)
        df["Lower"] = df["MA20"] - (df["Close"].rolling(window=20).std() * 2)
        fig_bb = go.Figure()
        fig_bb.add_trace(go.Scatter(x=df.index, y=df["Upper"], name="Upper Band", line=dict(color='red')))
        fig_bb.add_trace(go.Scatter(x=df.index, y=df["Lower"], name="Lower Band", line=dict(color='blue')))
        fig_bb.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Closing Price", line=dict(color='white')))
        fig_bb.update_layout(title="📊 Bollinger Bands - Detecting Overbought & Oversold Conditions", template="plotly_dark")

        # 🛑 RSI (Relative Strength Index) - Identifies if a stock is overbought or oversold.
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df.index, y=df["RSI"], mode="lines", name="RSI", line=dict(color='cyan')))
        fig_rsi.add_hline(y=70, line_dash="dot", line_color="red", annotation_text="Overbought")
        fig_rsi.add_hline(y=30, line_dash="dot", line_color="green", annotation_text="Oversold")
        fig_rsi.update_layout(title="🛑 RSI (Relative Strength Index) - Momentum Indicator", template="plotly_dark")

        return render_template(
            "visualization.html",
            fig_candlestick=fig_candlestick.to_html(full_html=False),
            fig_close=fig_close.to_html(full_html=False),
            fig_volume=fig_volume.to_html(full_html=False),
            fig_ma=fig_ma.to_html(full_html=False),
            fig_bb=fig_bb.to_html(full_html=False),
            fig_rsi=fig_rsi.to_html(full_html=False)
        )

    except Exception as e:
        logger.error(f"Visualization error: {e}\n{traceback.format_exc()}")
        return render_template("visualization.html", error=f"Visualization error: {str(e)}")


@app.route("/chat", methods=["GET", "POST"])
def chat():
    """Chat analysis route."""
    df, error = load_data_from_temp()
    if error:
        return render_template("chat.html", error=error)
    if df is None:
        return render_template("chat.html", error="No data loaded. Please upload a CSV or enter a ticker on the home page.")

    llm = load_llm()
    if not llm:
        return render_template("chat.html", error="LLM loading failed.")

    response = None
    if request.method == "POST":
        try:
            query = request.form["query"]
            data_summary = summarize_data(df, query)
            prompt = create_optimized_prompt(query, data_summary, df.columns.tolist())

            start_time = time.time()
            response = llm(prompt)
            end_time = time.time()
            logger.info(f"LLM Inference Time: {end_time - start_time:.2f} seconds")

            if not response:
                response = "No response generated. Please try a different query."

        except Exception as e:
            logger.error(f"Chat POST error: {e}\n{traceback.format_exc()}")
            return render_template("chat.html", error=str(e), response=response)

    return render_template("chat.html", response=response, error=error)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('index.html', error="Page not found"), 404

if __name__ == "__main__":
    cleanup_old_files() # Initial cleanup
    app.run(debug=True)
