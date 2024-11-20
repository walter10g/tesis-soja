from pydantic import BaseModel

class ColorAnalysisRequest(BaseModel):
    image_data: bytes  # Imagen en formato binario

class ColorAnalysisResponse(BaseModel):
    health: str  # Resultado del análisis ("Good" o "Bad")
