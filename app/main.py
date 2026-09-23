from fastapi import FastAPI

from app.config import get_settings
from app.routers import documents, query

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="A small retrieval-augmented generation API.",
)

app.include_router(documents.router)
app.include_router(query.router)


@app.get("/health", tags=["health"])
async def health_check() -> dict[str, str]:
    return {"status": "ok", "environment": settings.environment}
