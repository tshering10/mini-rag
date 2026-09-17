# Mini RAG

A small retrieval-augmented generation API built with FastAPI.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```powershell
uv sync
```

Create a local environment file when you need to override defaults:

```powershell
Copy-Item .env.example .env
```

## Run the API

```powershell
uv run uvicorn app.main:app --reload
```

The API is available at <http://127.0.0.1:8000>. Check the health endpoint:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Interactive API documentation is available at <http://127.0.0.1:8000/docs>.

## Run tests

```powershell
uv run pytest
```
