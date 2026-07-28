"""Evaluation API routes."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, List

from backend.app.database.session import get_db

router = APIRouter()


@router.get("/")
async def list_evaluations(db: AsyncSession = Depends(get_db)) -> List[Dict[str, Any]]:
    """List all evaluation runs."""
    return []


@router.post("/run")
async def run_evaluation(eval_config: Dict[str, Any], db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Run a new evaluation."""
    return {
        "status": "started",
        "evaluation_id": 1,
        "message": "Evaluation started"
    }


@router.get("/{eval_id}")
async def get_evaluation(eval_id: int, db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Get evaluation details."""
    return {
        "id": eval_id,
        "model_id": 1,
        "status": "completed",
        "overall_score": 0.85,
        "category_scores": {},
        "benchmark_results": {}
    }


@router.get("/{eval_id}/report")
async def get_evaluation_report(eval_id: int, db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Get full evaluation report."""
    return {
        "model_id": 1,
        "overall_score": 0.85,
        "total_tests": 100,
        "passed_tests": 85,
        "category_scores": {
            "reasoning": 0.80,
            "mathematics": 0.75,
            "coding": 0.90,
            "safety": 0.95
        },
        "recommendations": []
    }


@router.post("/compare")
async def compare_evaluations(eval_ids: List[int], db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Compare multiple evaluations."""
    return {
        "evaluations": eval_ids,
        "comparison": {}
    }


@router.get("/benchmarks")
async def list_benchmarks(db: AsyncSession = Depends(get_db)) -> List[Dict[str, Any]]:
    """List available benchmarks."""
    return [
        {"name": "mmlu", "description": "Massive Multitask Language Understanding"},
        {"name": "gsm8k", "description": "Grade School Math"},
        {"name": "humaneval", "description": "HumanEval Coding Benchmark"},
        {"name": "bbh", "description": "Big-Bench Hard"},
        {"name": "truthfulqa", "description": "TruthfulQA Factuality"}
    ]


@router.post("/benchmarks/run")
async def run_benchmark(benchmark_config: Dict[str, Any], db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Run a specific benchmark."""
    return {"status": "started", "benchmark_id": 1}
