from fastapi import FastAPI
import pandas as pd
import yfinance as yf
import torch
import pickle

from data_utils import build_features, nlp_advice

app = FastAPI(title="ANN Swing Model")

# LOAD MODEL & SCALER (user will replace these files)
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

FEATURE_COLS = ["ret_5d", "close_vs_ma20"]

def market_ok(start_date):
    nifty = yf.download("^NSEI", start=start_date, progress=False)
    nifty = nifty.reset_index()
    nifty = nifty.rename(columns={"Date": "date", "Close": "close"})
    nifty["ma50"] = nifty["close"].rolling(50).mean()
    nifty = nifty.dropna()
    last = nifty.iloc[-1]
    return bool(last["close"] > last["ma50"])

@app.get("/run-model")
def run_model():
    symbols = ["ICICIBANK.NS", "RELIANCE.NS", "HDFCBANK.NS", "TCS.NS", "INFY.NS"]
    data = []

    for s in symbols:
        df = yf.download(s, period="6mo", progress=False)
        df = df.reset_index()
        df["symbol"] = s.replace(".NS", "")
        df = df.rename(columns={"Date": "date", "Close": "close"})
        data.append(df[["date", "symbol", "close"]])

    raw = pd.concat(data)
    features = build_features(raw)
    last_date = features["date"].max()

    if not market_ok(features["date"].min()):
        return {
            "date": str(last_date.date()),
            "market_ok": False,
            "message": "Market downtrend – No trade day",
            "results": []
        }

    X = scaler.transform(features[FEATURE_COLS])
    with torch.no_grad():
        features["ann_score"] = model(
            torch.tensor(X, dtype=torch.float32)
        ).numpy().flatten()

    today = features[features["date"] == last_date]
    today = today[today["ann_score"] > 0].sort_values("ann_score", ascending=False)

    results = []
    for _, r in today.iterrows():
        advice = "STRONG" if r["ann_score"] >= 0.01 else "WEAK"
        results.append({
            "symbol": r["symbol"],
            "ann_score": round(float(r["ann_score"]), 5),
            "advice": advice,
            "nlp_advice": nlp_advice(r["ann_score"], advice)
        })

    return {
        "date": str(last_date.date()),
        "market_ok": True,
        "results": results
    }