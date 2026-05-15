"""Entrypoint: uvicorn api.main:app --reload"""
from fastapi import FastAPI

from api.endpoints import router

app = FastAPI(title="Chest X-Ray Intelligence", version="0.1.0")
app.include_router(router)
