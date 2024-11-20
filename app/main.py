from fastapi import FastAPI
from app.model import predict_color
from app.schemas import ColorAnalysisRequest, ColorAnalysisResponse

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "¡Servidor de IA activo!"}

@app.post("/analyze-color", response_model=ColorAnalysisResponse)
def analyze_color(data: ColorAnalysisRequest):
    prediction = predict_color(data.image_data)
    return {"health": prediction}
