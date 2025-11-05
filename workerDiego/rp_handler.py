import os
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI
from model import CropInference
from pydantic import BaseModel

app = FastAPI()

cropper = CropInference()


class InputData(BaseModel):
    image: str


class RunPodRequest(BaseModel):
    input: InputData


@app.post("/")
async def predict(request: RunPodRequest):
    """Main endpoint for predictions"""
    try:
        result = cropper.infer(request.input.image)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


@app.get("/ping")
async def ping():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
