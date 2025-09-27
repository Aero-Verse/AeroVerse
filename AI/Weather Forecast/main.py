from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from all_pred import router as all_pred_router
from Weather3 import router as weather3_router
from yoloo import router as yoloo_router

app = FastAPI(
    title="AI Services API Gateway",
    description="Integrated API for all AI projects",
    version="1.0.0",
    docs_url="/api-docs",
    redoc_url=None
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# التعديل الرئيسي هنا - tags موحدة
app.include_router(all_pred_router, prefix="/all-pred", tags=["Weather Forecast"])
app.include_router(weather3_router, prefix="/weather3", tags=["Image Classification"])
app.include_router(yoloo_router, prefix="/yoloo", tags=["Object Detection"])

@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "Welcome to AI Services API Gateway",
        "services": {
            "Weather Forecast": "/all-pred",
            "Image Classification": "/weather3",
            "Object Detection": "/yoloo"
        },
        "documentation": "/api-docs"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)