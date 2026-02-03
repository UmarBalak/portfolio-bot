import uvicorn
from ast import Dict
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException
from starlette.responses import Response
from typing import Any, Dict
import logging
from dotenv import load_dotenv
from fastapi import HTTPException
import os
from starlette.responses import JSONResponse, Response
from pydantic import BaseModel


# Imports from your project structure
from chatbot_pipeline import ChatbotPipeline

logging.basicConfig(level=logging.INFO)
load_dotenv()

# --- Main App Setup ---
app = FastAPI()

FRONTEND_URL = os.getenv("FRONTEND_URL")
LLM_MODEL = os.getenv("LLM_MODEL")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["FRONTEND_URL"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Set-Cookie"],
)

chatbot_pipeline = ChatbotPipeline(llm_model=LLM_MODEL)

class QueryRequest(BaseModel):
    query: str
    temperature: float = 0.5

class QueryResponse(BaseModel):
    query: str
    answer: str
    tokens_used: Dict[str, Any]

user_llm_cache = {}

def get_user_llm(user_id):
    if user_id not in user_llm_cache:
        from llm_models import LLM
        user_llm_cache[user_id] = LLM(llm_model=LLM_MODEL)
    return user_llm_cache[user_id]


# --- Endpoints ---

@app.get("/", response_model=dict)
async def root():
    return {"message": "Portfolio Chatbot Service is running."}

@app.get("/health", response_class=JSONResponse)
async def health_check():
    return {"status": "healthy"}

@app.head("/health")
async def health_check_monitor():
    return Response(status_code=200)
  
@app.post("/chatbot/query", response_model=QueryResponse)
async def query_chatbot(
    query_request: QueryRequest,
):
    try:
        result = chatbot_pipeline.query_with_template_method(
            query_text=query_request.query
        )
        
        return QueryResponse(
            query=query_request.query,
            answer=result.get('answer', ''),
            tokens_used=result.get("tokens_used", {}),
        )
        
    except Exception as e:
        logging.error(f"Query failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=5000, log_level="info", reload=True)