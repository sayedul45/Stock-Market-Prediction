import pandas as pd
import numpy as np
from datetime import datetime
import logging
import hashlib
from typing import Optional, List, Dict, Any
# import yfinance as yf

logger = logging.getLogger(__name__)

def validate_company_code(company_code: str, label_encoder) -> bool:
    try:
        label_encoder.transform([company_code])
        return True
    except ValueError:
        return False

def get_company_list(label_encoder) -> List[str]:
    try:
        return list(label_encoder.classes_)
    except Exception as e:
        logger.error(f"Error getting company list: {str(e)}")
        return []

def generate_features_for_inference(date_str: str, encoded_code: int) -> Optional[pd.DataFrame]:
    try:
        # Generate deterministic seed from date and code
        key = f"{date_str}_{encoded_code}"
        seed = int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)

        date = pd.to_datetime(date_str)
        base_price = rng.uniform(100, 500)
        volume = rng.integers(1_000_000, 10_000_000)

        features = {
            'code': [encoded_code],
            'date': [date.strftime('%Y-%m-%d')],
            'open': [base_price * rng.uniform(0.95, 1.05)],
            'high': [base_price * rng.uniform(1.01, 1.05)],
            'low': [base_price * rng.uniform(0.95, 0.99)],
            'close': [base_price],
            'volume': [volume],
            'SMA_5': [base_price + rng.uniform(-2, 2)],
            'SMA_10': [base_price + rng.uniform(-3, 3)],
            'EMA_5': [base_price + rng.uniform(-2, 2)],
            'EMA_10': [base_price + rng.uniform(-3, 3)],
            'MACD': [rng.uniform(-1, 1)],
            'RSI': [rng.uniform(30, 70)],
            'OBV': [rng.integers(-1_000_000, 1_000_000)],
            'day_of_week': [date.weekday()],
            'month': [date.month],
            'quarter': [date.quarter]
        }

        for lag in range(1, 6):
            features[f'close_lag_{lag}'] = [base_price * rng.uniform(0.95, 1.05)]
            features[f'volume_lag_{lag}'] = [rng.integers(500_000, 10_000_000)]

        prev_close = base_price * rng.uniform(0.95, 1.05)
        price_change = base_price - prev_close

        features.update({
            'price_change': [price_change],
            'price_change_pct': [price_change / prev_close * 100],
            'daily_range': [features['high'][0] - features['low'][0]],
            'daily_range_pct': [(features['high'][0] - features['low'][0]) / base_price * 100],
            'sma_cross': [int(features['SMA_5'][0] > features['SMA_10'][0])],
            'ema_cross': [int(features['EMA_5'][0] > features['EMA_10'][0])],
            'volume_sma5': [rng.integers(1_000_000, 5_000_000)],
            'volume_change': [rng.uniform(-20, 20)],
            'macd_rsi_signal': [int(features['MACD'][0] > 0 and features['RSI'][0] > 50)],
        })

        for i in range(384):  # or however many BERT features your model needs
            features[f'bert_feature_{i}'] = [rng.uniform(-1, 1)]

        return pd.DataFrame(features)

    except Exception as e:
        logger.error(f"Error generating features: {str(e)}")
        return None

def get_real_market_data(symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)

        if data.empty:
            logger.warning(f"No data found for symbol {symbol}")
            return None

        data.reset_index(inplace=True)
        data.rename(columns={
            'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low',
            'Close': 'close', 'Volume': 'volume'
        }, inplace=True)

        return data

    except Exception as e:
        logger.error(f"Error fetching market data for {symbol}: {str(e)}")
        return None

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.copy()
        df['SMA_5'] = df['close'].rolling(window=5).mean()
        df['SMA_10'] = df['close'].rolling(window=10).mean()
        df['EMA_5'] = df['close'].ewm(span=5).mean()
        df['EMA_10'] = df['close'].ewm(span=10).mean()
        df['MACD'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()

        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        return df

    except Exception as e:
        logger.error(f"Error calculating indicators: {str(e)}")
        return df

def generate_real_features_for_inference(date_str: str, encoded_code: int, historical_data: pd.DataFrame) -> Optional[pd.DataFrame]:
    try:
        date = pd.to_datetime(date_str)
        historical_data = calculate_technical_indicators(historical_data)
        historical_data = historical_data[historical_data['date'] < date]

        if historical_data.empty:
            return generate_features_for_inference(date_str, encoded_code)

        latest = historical_data.iloc[-1]

        features = {
            'code': [encoded_code],
            'date': [date_str],
            'open': [latest['open']],
            'high': [latest['high']],
            'low': [latest['low']],
            'close': [latest['close']],
            'volume': [latest['volume']],
            'SMA_5': [latest.get('SMA_5', latest['close'])],
            'SMA_10': [latest.get('SMA_10', latest['close'])],
            'EMA_5': [latest.get('EMA_5', latest['close'])],
            'EMA_10': [latest.get('EMA_10', latest['close'])],
            'MACD': [latest.get('MACD', 0)],
            'RSI': [latest.get('RSI', 50)],
            'OBV': [latest.get('OBV', 0)],
            'day_of_week': [date.weekday()],
            'month': [date.month],
            'quarter': [date.quarter]
        }

        for lag in range(1, 6):
            if len(historical_data) > lag:
                features[f'close_lag_{lag}'] = [historical_data.iloc[-lag]['close']]
                features[f'volume_lag_{lag}'] = [historical_data.iloc[-lag]['volume']]
            else:
                features[f'close_lag_{lag}'] = [latest['close']]
                features[f'volume_lag_{lag}'] = [latest['volume']]

        prev_close = features['close_lag_1'][0]
        price_change = latest['close'] - prev_close

        features.update({
            'price_change': [price_change],
            'price_change_pct': [price_change / prev_close * 100],
            'daily_range': [latest['high'] - latest['low']],
            'daily_range_pct': [(latest['high'] - latest['low']) / latest['close'] * 100],
            'sma_cross': [int(features['SMA_5'][0] > features['SMA_10'][0])],
            'ema_cross': [int(features['EMA_5'][0] > features['EMA_10'][0])],
            'volume_sma5': [historical_data['volume'].tail(5).mean()],
            'volume_change': [(latest['volume'] - historical_data['volume'].tail(5).mean()) / historical_data['volume'].tail(5).mean() * 100],
            'macd_rsi_signal': [int(features['MACD'][0] > 0 and features['RSI'][0] > 50)],
        })

        for i in range(384):
            features[f'bert_feature_{i}'] = [np.random.uniform(-1, 1)]

        return pd.DataFrame(features)

    except Exception as e:
        logger.error(f"Error generating real features: {str(e)}")
        return generate_features_for_inference(date_str, encoded_code)
