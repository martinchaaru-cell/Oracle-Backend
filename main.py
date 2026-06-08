from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import your M25 orchestrator
from module25 import app as oracle_app

# Configure CORS for frontend
oracle_app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://the-oracle-nlbk.vercel.app"],  # Your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(oracle_app, host="0.0.0.0", port=8000)
