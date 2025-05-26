import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, datetime, timedelta
import json
import time

# Page configuration
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .sidebar-header {
        color: #1f77b4;
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'backend_url' not in st.session_state:
    st.session_state.backend_url = "http://localhost:5000"

# Sidebar configuration
st.sidebar.markdown('<div class="sidebar-header">⚙️ Configuration</div>', unsafe_allow_html=True)

# Backend URL configuration
backend_url = st.sidebar.text_input(
    "Backend URL", 
    value=st.session_state.backend_url,
    help="URL of the prediction API backend"
)
st.session_state.backend_url = backend_url

# Check backend health
def check_backend_health():
    try:
        response = requests.get(f"{backend_url}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

# Display backend status
backend_status = check_backend_health()
status_color = "🟢" if backend_status else "🔴"
st.sidebar.markdown(f"**Backend Status:** {status_color} {'Connected' if backend_status else 'Disconnected'}")

# Main app header
st.markdown('<div class="main-header">📈 Stock Price Predictor</div>', unsafe_allow_html=True)
st.markdown("---")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Single Prediction", "📊 Batch Prediction", "📈 History", "ℹ️ About"])

with tab1:
    st.header("Single Stock Prediction")
    
    # Create columns for better layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Get available companies
        companies = []
        if backend_status:
            try:
                response = requests.get(f"{backend_url}/companies", timeout=10)
                if response.status_code == 200:
                    companies_data = response.json()
                    companies = companies_data.get('companies', [])
            except:
                pass
        
        # Company selection
        if companies:
            company_code = st.selectbox(
                "Select Company Code",
                options=companies,
                index=0 if companies else None,
                help="Choose from available company codes in the model"
            )
        else:
            company_code = st.text_input(
                "Company Code", 
                value="AAPL",
                placeholder="e.g., AAPL, GOOGL, MSFT",
                help="Enter the stock symbol for prediction"
            ).upper()
        
        # Date selection
        selected_date = st.date_input(
            "Select Date for Prediction",
            value=date.today(),
            min_value=date.today() - timedelta(days=365),
            max_value=date.today() + timedelta(days=365),
            help="Choose the date for stock price prediction"
        )
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            show_confidence = st.checkbox("Show prediction confidence", value=True)
            auto_refresh = st.checkbox("Auto-refresh prediction", value=False)
    
    with col2:
        st.markdown("### 📊 Quick Stats")
        if companies:
            st.metric("Available Companies", len(companies))
        
        # Prediction button
        predict_button = st.button(
            "🎯 Predict Stock Price", 
            type="primary",
            use_container_width=True,
            disabled=not backend_status
        )
    
    # Make prediction
    if predict_button or (auto_refresh and backend_status):
        if not backend_status:
            st.error("❌ Backend is not connected. Please check the backend URL and ensure the server is running.")
        else:
            with st.spinner("🔮 Making prediction..."):
                payload = {
                    "code": company_code,
                    "date": selected_date.strftime("%Y-%m-%d")
                }
                
                try:
                    response = requests.post(
                        f"{backend_url}/predict", 
                        json=payload,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display prediction result
                        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                        st.success(f"✅ Prediction completed successfully!")
                        
                        # Main prediction display
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Predicted Price",
                                f"${result['predicted_price']:.2f}",
                                delta=None
                            )
                        
                        with col2:
                            st.metric("Company", result['company'])
                        
                        with col3:
                            st.metric("Date", result['date'])
                        
                        # Show confidence if available
                        if show_confidence and result.get('confidence'):
                            st.info(f"🎯 Prediction Confidence: {result['confidence']:.2%}")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Add to history
                        result['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.session_state.prediction_history.append(result)
                        
                        # Limit history to last 50 predictions
                        if len(st.session_state.prediction_history) > 50:
                            st.session_state.prediction_history = st.session_state.prediction_history[-50:]
                    
                    else:
                        error_data = response.json()
                        st.error(f"❌ Prediction failed: {error_data.get('error', 'Unknown error')}")
                
                except requests.exceptions.Timeout:
                    st.error("⏱️ Request timed out. The backend might be busy.")
                except requests.exceptions.ConnectionError:
                    st.error("🚫 Failed to connect to backend. Please check if the server is running.")
                except Exception as e:
                    st.error(f"🚨 Unexpected error: {str(e)}")

with tab2:
    st.header("Batch Predictions")
    st.markdown("Upload a CSV file or enter multiple predictions manually.")
    
    # File upload option
    uploaded_file = st.file_uploader(
        "Upload CSV file with columns: code, date",
        type=['csv'],
        help="CSV should have 'code' and 'date' columns"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.subheader("📋 Data Preview")
            st.dataframe(df.head())
            
            if st.button("🚀 Run Batch Predictions", type="primary"):
                if not backend_status:
                    st.error("❌ Backend is not connected.")
                else:
                    predictions_data = df.to_dict('records')
                    batch_payload = {"predictions": predictions_data}
                    
                    with st.spinner("Processing batch predictions..."):
                        try:
                            response = requests.post(
                                f"{backend_url}/batch_predict",
                                json=batch_payload,
                                timeout=60
                            )
                            
                            if response.status_code == 200:
                                results = response.json()
                                
                                st.success(f"✅ Batch prediction completed!")
                                st.info(f"📊 {results['successful_predictions']} out of {results['total_predictions']} predictions successful")
                                
                                # Convert results to DataFrame
                                results_df = pd.DataFrame(results['results'])
                                st.subheader("📈 Batch Results")
                                st.dataframe(results_df)
                                
                                # Download results
                                csv = results_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Results",
                                    data=csv,
                                    file_name=f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                            else:
                                st.error(f"❌ Batch prediction failed: {response.json().get('error', 'Unknown error')}")
                        
                        except Exception as e:
                            st.error(f"🚨 Error processing batch: {str(e)}")
        
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {str(e)}")
    
    else:
        # Manual batch input
        st.subheader("✍️ Manual Entry")
        
        num_predictions = st.number_input("Number of predictions", min_value=1, max_value=10, value=3)
        
        batch_data = []
        for i in range(num_predictions):
            col1, col2 = st.columns(2)
            with col1:
                code = st.text_input(f"Company Code {i+1}", key=f"batch_code_{i}")
            with col2:
                pred_date = st.date_input(f"Date {i+1}", key=f"batch_date_{i}")
            
            if code and pred_date:
                batch_data.append({
                    'code': code.upper(),
                    'date': pred_date.strftime('%Y-%m-%d')
                })
        
        if st.button("🚀 Run Manual Batch Predictions", type="primary") and batch_data:
            if not backend_status:
                st.error("❌ Backend is not connected.")
            else:
                batch_payload = {"predictions": batch_data}
                
                with st.spinner("Processing manual batch predictions..."):
                    try:
                        response = requests.post(
                            f"{backend_url}/batch_predict",
                            json=batch_payload,
                            timeout=60
                        )
                        
                        if response.status_code == 200:
                            results = response.json()
                            st.success(f"✅ Manual batch prediction completed!")
                            st.info(f"📊 {results['successful_predictions']} out of {results['total_predictions']} predictions successful")
                            
                            results_df = pd.DataFrame(results['results'])
                            st.dataframe(results_df)
                        else:
                            st.error(f"❌ Batch prediction failed: {response.json().get('error')}")
                    
                    except Exception as e:
                        st.error(f"🚨 Error: {str(e)}")

with tab3:
    st.header("📈 Prediction History")
    
    if st.session_state.prediction_history:
        # Convert history to DataFrame
        history_df = pd.DataFrame(st.session_state.prediction_history)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Predictions", len(history_df))
        with col2:
            avg_price = history_df['predicted_price'].mean()
            st.metric("Average Price", f"${avg_price:.2f}")
        with col3:
            unique_companies = history_df['company'].nunique()
            st.metric("Unique Companies", unique_companies)
        with col4:
            latest_prediction = history_df.iloc[-1]['predicted_price']
            st.metric("Latest Prediction", f"${latest_prediction:.2f}")
        
        # Plot history
        if len(history_df) > 1:
            fig = px.line(
                history_df, 
                x='timestamp', 
                y='predicted_price',
                color='company',
                title="Prediction History Over Time",
                labels={'predicted_price': 'Predicted Price ($)', 'timestamp': 'Time'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Display history table
        st.subheader("📋 Detailed History")
        
        # Add filters
        col1, col2 = st.columns(2)
        with col1:
            companies_filter = st.multiselect(
                "Filter by Company",
                options=history_df['company'].unique(),
                default=history_df['company'].unique()
            )
        with col2:
            if st.button("🗑️ Clear History"):
                st.session_state.prediction_history = []
                st.rerun()
        
        # Filter and display
        filtered_df = history_df[history_df['company'].isin(companies_filter)]
        st.dataframe(
            filtered_df.sort_values('timestamp', ascending=False),
            use_container_width=True
        )
        
        # Download history
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download History",
            data=csv,
            file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("📝 No predictions made yet. Make some predictions to see the history!")

with tab4:
    st.header("ℹ️ About This Application")
    
    st.markdown("""
    ### 🎯 Stock Price Predictor
    
    This application uses machine learning to predict stock prices based on historical data and technical indicators.
    
    #### 📊 Features Used in the Model:
    - **Basic OHLCV Data**: Open, High, Low, Close, Volume
    - **Technical Indicators**: SMA, EMA, MACD, RSI, OBV
    - **Lag Features**: Historical prices and volumes (1-5 days)
    - **Date Features**: Day of week, month, quarter
    - **Derived Features**: Price changes, ranges, technical signals
    - **News Analysis**: BERT-based sentiment features (384 dimensions)
    
    #### 🔧 How It Works:
    1. **Data Input**: Enter company code and prediction date
    2. **Feature Generation**: Calculate technical indicators and features
    3. **ML Prediction**: Use trained LightGBM model for prediction
    4. **Result Display**: Show predicted price with confidence metrics
    
    #### ⚠️ Important Notes:
    - This is for educational/demonstration purposes only
    - Real trading decisions should not be based solely on these predictions
    - Model performance depends on training data quality and market conditions
    - Historical performance does not guarantee future results
    
    #### 🛠️ Technical Stack:
    - **Backend**: Flask + LightGBM
    - **Frontend**: Streamlit
    - **Features**: Technical Analysis + NLP (BERT)
    - **Deployment**: Docker-ready
    
    #### 📈 Model Information:
    - **Algorithm**: LightGBM (Gradient Boosting)
    - **Features**: {len(['code', 'date', 'open', 'high', 'low', 'close', 'volume', 'SMA_5', 'SMA_10', 'EMA_5', 'EMA_10', 'MACD', 'RSI', 'OBV', 'day_of_week', 'month', 'quarter'] + [f'close_lag_{i}' for i in range(1,6)] + [f'volume_lag_{i}' for i in range(1,6)] + ['price_change', 'price_change_pct', 'daily_range', 'daily_range_pct', 'sma_cross', 'ema_cross', 'volume_sma5', 'volume_change', 'macd_rsi_signal', 'combined_headlines'] + [f'bert_feature_{i}' for i in range(384)])} total features
    - **Target**: Next day closing price
    - **Preprocessing**: StandardScaler normalization
    """)
    
    # System status
    st.subheader("🔧 System Status")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Backend Status", "🟢 Connected" if backend_status else "🔴 Disconnected")
        st.metric("Session Predictions", len(st.session_state.prediction_history))
    
    with col2:
        st.metric("Backend URL", backend_url)
        if backend_status:
            try:
                health_response = requests.get(f"{backend_url}/health", timeout=5)
                health_data = health_response.json()
                st.metric("Backend Health", "✅ Healthy")
                st.text(f"Last checked: {health_data.get('timestamp', 'Unknown')}")
            except:
                st.metric("Backend Health", "❌ Unhealthy")

# Auto-refresh for real-time updates (optional)
if st.sidebar.checkbox("🔄 Auto-refresh (30s)", value=False):
    time.sleep(30)
    st.rerun()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        📈 Stock Price Predictor | Built with Streamlit & Flask | 
        <a href='#' style='color: #1f77b4;'>Documentation</a> | 
        <a href='#' style='color: #1f77b4;'>GitHub</a>
    </div>
    """, 
    unsafe_allow_html=True
)