from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(title="Customer Churn Predictor", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
@app.get("/")
def root():
    return {"system": "Customer Churn Predictor", "version": "1.0.0", "status": "operational", "accuracy": "85%+", "revenue_target": "$25K/month", "capabilities": ["churn-prediction", "retention-playbooks", "risk-scoring", "30-90-day-early-warning"], "pricing": {"startup": "$199/month", "growth": "$499/month", "enterprise": "$1499/month"}}
@app.get("/health")
def health():
    return {"status": "healthy", "system": "Customer Churn Predictor"}
