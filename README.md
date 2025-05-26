# 📈 Stock Market Prediction using News Headlines (Flask + Streamlit)

An intelligent stock price forecasting system powered by machine learning and NLP. This project predicts future stock prices based on historical trends and financial news headlines.

---

## 📂 Project Structure

├── backend/ # Flask backend API
│ ├── app.py # API server
│ ├── utils.py # Feature builder for inference
│ └── model/ # Saved model + encoders
│ ├── best_model_xgboost.pkl
│ ├── scaler_X.pkl
│ ├── scaler_y.pkl
│ ├── label_encoder_code.pkl
│ └── features.json
├── frontend/ # Streamlit frontend
│ ├── app.py # Streamlit UI
│ ├── static/ # Optional static files (CSS)
│ └── components/ # Streamlit display components
├── assets/ # Screenshots for documentation
│ ├── streamlit_ui.png
│ └── flask_api_docs.png
├── docker-compose.yml # Docker integration (optional)
├── sample_input.csv # Example input file
├── .gitignore
└── README.md # Project documentation



---

## 🚀 Features

- 📅 Predict stock price based on **date** and **company**
- 🧠 NLP embeddings for news headlines using **Sentence-BERT**
- 💾 Support for pretrained ML models (XGBoost, LightGBM, LSTM)
- 🧮 Feature scaling and label encoding included in pipeline
- 💬 Streamlit web interface for user interaction
- 🧪 RESTful Flask API for backend predictions
- 🐳 Optional Docker support for full deployment

---

## 🧰 Tech Stack

| Layer      | Tools                                     |
|------------|--------------------------------------------|
| Backend    | Flask, Scikit-learn, XGBoost, LightGBM     |
| NLP        | Sentence-Transformers, BERT                |
| Frontend   | Streamlit                                  |
| DevOps     | Docker, Docker Compose                     |

---

## 💻 Manual Local Run (Without Docker)

### 🧠 Backend (Flask API)

```bash
cd backend
pip install -r requirements.txt
python app.py
```

### Frontend (Streamlit)
```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

## 🖼️ Screenshots

### 🔸 Streamlit App UI
![Streamlit UI](assets/Streamlit_UI.png)