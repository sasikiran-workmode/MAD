"""
FastAPI application for the Multi-Agent Debate framework.
Task 4: keeps endpoints async def but calls the sync pipeline
        via run_in_executor so we don't block the server event loop.
"""
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from loguru import logger

from app import solve as sync_solve

app = FastAPI(
    title="Multi-Agent Debate API",
    description="Evidence-triggered debate with conflict classification",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)


class QueryRequest(BaseModel):
    """Request model for /solve endpoint."""
    query: str = Field(..., description="The question to answer")
    demo_mode: bool = Field(default=False, description="Use fixture/demo mode")


class QueryResponse(BaseModel):
    """Response model for /solve endpoint."""
    query: str
    answer: str
    reasoning: str
    evidence_state: str
    debate_triggered: bool
    conflict_type: Optional[str] = None
    sources: list = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "name": "Multi-Agent Debate API",
        "version": "2.0.0",
        "status": "running",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/api/solve", response_model=QueryResponse)
@app.post("/solve", response_model=QueryResponse)
async def solve_query(request: QueryRequest):
    """
    Solve a query using the Multi-Agent Debate system (sync pipeline via threadpool).
    """
    try:
        logger.info(f"API request: {request.query!r}")

        loop = asyncio.get_event_loop()
        trace = await loop.run_in_executor(
            None, lambda: sync_solve(request.query, demo_mode=request.demo_mode)
        )

        if trace.error:
            raise HTTPException(status_code=500, detail=trace.error)

        decision = trace.evidence_decision
        answer_obj = trace.final_answer

        response = QueryResponse(
            query=trace.query,
            answer=answer_obj.answer if answer_obj else "No answer generated",
            reasoning=answer_obj.reasoning if answer_obj else "",
            evidence_state=decision.state.value if decision else "unknown",
            debate_triggered=trace.debate_triggered,
            conflict_type=trace.conflict.type.value if trace.conflict else None,
            metrics=trace.metrics,
        )

        if answer_obj and answer_obj.sources:
            for source in answer_obj.sources:
                response.sources.append({"url": source.url, "title": source.title})

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)