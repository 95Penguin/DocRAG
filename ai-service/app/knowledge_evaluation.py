"""Reproducible, document-grounded evaluation for uploaded knowledge bases.

Retrieval evaluation is offline. ``--answers`` is opt-in because it calls the
configured paid model and still requires human review of semantic answer points.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from pydantic import BaseModel, Field, model_validator

from app.config import get_settings
from app.generation.runtime import get_answer_service, get_embedding_encoder
from app.knowledge_agent import get_knowledge_agent_service
from app.retrieval.bm25 import BM25Index
from app.retrieval.dense import DenseIndex
from app.retrieval.graph import HeadingGraphIndex
from app.retrieval.hybrid import MultiRouteRRFRetriever
from app.retrieval.reranker import BGERerankerScorer, RerankingRetriever


class EvidenceTarget(BaseModel):
    document: str
    section_path: list[str] = Field(min_length=1)
    anchor: str = Field(min_length=4)


class EvaluationCase(BaseModel):
    id: str
    knowledge_base: str
    category: str
    question: str
    expected_evidence: list[EvidenceTarget] = Field(default_factory=list)
    required_answer_points: list[str] = Field(default_factory=list)
    should_refuse: bool = False

    @model_validator(mode="after")
    def check_gold(self) -> EvaluationCase:
        if self.should_refuse and self.expected_evidence:
            raise ValueError("refusal case must not have expected evidence")
        if not self.should_refuse and not self.expected_evidence:
            raise ValueError("answerable case needs expected evidence")
        return self


@dataclass(frozen=True)
class ResolvedCase:
    case: EvaluationCase
    kb_id: str
    owner_id: str
    relevant_chunk_ids: frozenset[str]


def load_cases(path: Path) -> list[EvaluationCase]:
    cases = [
        EvaluationCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not cases or len({case.id for case in cases}) != len(cases):
        raise ValueError("evaluation dataset must contain unique case IDs")
    return cases


def resolve_cases(root: Path, cases: list[EvaluationCase]) -> tuple[list[ResolvedCase], dict[str, Path]]:
    """Fail before evaluation when a human gold label no longer matches the active index."""
    metadata_by_name: dict[str, list[tuple[dict[str, Any], Path]]] = defaultdict(list)
    for path in root.glob("*/metadata.json"):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata_by_name[metadata["name"]].append((metadata, path.parent))
    resolved: list[ResolvedCase] = []
    active_by_id: dict[str, Path] = {}
    for case in cases:
        matches = metadata_by_name.get(case.knowledge_base, [])
        if len(matches) != 1:
            raise ValueError(f"case {case.id}: knowledge base name must resolve uniquely")
        metadata, folder = matches[0]
        kb_id = str(uuid.UUID(metadata["id"]))
        pointer = folder / "active_revision.json"
        active = folder
        if pointer.is_file():
            revision = str(uuid.UUID(json.loads(pointer.read_text(encoding="utf-8"))["revision"]))
            active = folder / "revisions" / revision
        documents = json.loads((active / "documents.json").read_text(encoding="utf-8"))
        chunks = json.loads((active / "chunks.json").read_text(encoding="utf-8"))
        relevant: set[str] = set()
        for target in case.expected_evidence:
            document_ids = {item["id"] for item in documents if item["filename"] == target.document}
            hit = {
                item["chunk_id"]
                for item in chunks
                if any(item["source_path"].startswith(f"{kb_id}/{document_id}-") for document_id in document_ids)
                and item["section_path"] == target.section_path
                and target.anchor.casefold() in item["content"].casefold()
            }
            if not hit:
                raise ValueError(f"case {case.id}: gold evidence not found: {target.document} / {target.section_path}")
            relevant.update(hit)
        active_by_id[kb_id] = active
        resolved.append(ResolvedCase(case, kb_id, metadata["owner_id"], frozenset(relevant)))
    return resolved, active_by_id


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = max(0, math.ceil(fraction * len(ordered)) - 1)
    return round(ordered[position], 3)


def evaluate_retrieval(retriever_by_kb: dict[str, Any], cases: list[ResolvedCase], top_k: int) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    latencies: list[float] = []
    for resolved in cases:
        started = time.perf_counter()
        results = retriever_by_kb[resolved.kb_id].search(resolved.case.question, top_k=top_k)
        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        latencies.append(latency_ms)
        retrieved = [item.chunk_id for item in results]
        relevant = resolved.relevant_chunk_ids
        item: dict[str, Any] = {
            "id": resolved.case.id,
            "knowledge_base": resolved.case.knowledge_base,
            "category": resolved.case.category,
            "question": resolved.case.question,
            "should_refuse": resolved.case.should_refuse,
            "latency_ms": latency_ms,
            "retrieved_chunk_ids": retrieved,
        }
        if relevant:
            ranks = [index for index, chunk_id in enumerate(retrieved, 1) if chunk_id in relevant]
            dcg = sum(1 / math.log2(rank + 1) for rank in ranks)
            ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(relevant), top_k) + 1))
            item.update(
                recall_at_k=round(len(ranks) / len(relevant), 6),
                reciprocal_rank=round(1 / ranks[0], 6) if ranks else 0.0,
                ndcg_at_k=round(dcg / ideal, 6) if ideal else 0.0,
                first_relevant_rank=ranks[0] if ranks else None,
                relevant_chunk_ids=sorted(relevant),
                failure_type=None if ranks else "gold_evidence_not_recalled",
            )
        else:
            item["failure_type"] = "not_scored_retrieval_only"
        details.append(item)
    positives = [item for item in details if not item["should_refuse"]]
    by_kb: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in positives:
        by_kb[item["knowledge_base"]].append(item)
    return {
        "questions_total": len(details),
        "answerable_questions": len(positives),
        "recall_at_k": round(mean(item["recall_at_k"] for item in positives), 6),
        "mrr": round(mean(item["reciprocal_rank"] for item in positives), 6),
        "ndcg_at_k": round(mean(item["ndcg_at_k"] for item in positives), 6),
        "latency_ms": {"p50": percentile(latencies, 0.5), "p95": percentile(latencies, 0.95)},
        "by_knowledge_base": {
            name: {
                "recall_at_k": round(mean(item["recall_at_k"] for item in items), 6),
                "mrr": round(mean(item["reciprocal_rank"] for item in items), 6),
                "misses": [item["id"] for item in items if item["failure_type"]],
            }
            for name, items in sorted(by_kb.items())
        },
        "details": details,
    }


def evaluate_answers(service: Any, cases: list[ResolvedCase]) -> dict[str, Any]:
    """Record objective status/citation signals; semantic answer points remain manual."""
    details: list[dict[str, Any]] = []
    latencies: list[float] = []
    for resolved in cases:
        started = time.perf_counter()
        try:
            answer = service.answer(
                resolved.case.question, [], [resolved.kb_id], resolved.owner_id
            )
            error = None
        except Exception as exc:  # noqa: BLE001 - preserve a batch's completed cases
            answer, error = None, f"{type(exc).__name__}: {exc}"
        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        latencies.append(latency_ms)
        expected_status = "refused" if resolved.case.should_refuse else "answered"
        citation_ids = [item.chunk_id for item in answer.citations] if answer else []
        hit = sorted(resolved.relevant_chunk_ids.intersection(citation_ids))
        actual_status = answer.status if answer else "error"
        details.append({
            "id": resolved.case.id,
            "knowledge_base": resolved.case.knowledge_base,
            "expected_status": expected_status,
            "actual_status": actual_status,
            "status_correct": actual_status == expected_status,
            "citation_recall": round(len(hit) / len(resolved.relevant_chunk_ids), 6)
            if resolved.relevant_chunk_ids and answer and actual_status == "answered" else None,
            "required_answer_points": resolved.case.required_answer_points,
            "manual_answer_point_review": "pending" if not resolved.case.should_refuse else "not_applicable",
            "answer": answer.answer if answer else None,
            "citations": citation_ids,
            "error": error,
            "latency_ms": latency_ms,
            "failure_type": error or ("wrong_answerability" if actual_status != expected_status else None),
        })
    positives = [item for item in details if item["expected_status"] == "answered"]
    negatives = [item for item in details if item["expected_status"] == "refused"]
    return {
        "answerability_accuracy": round(mean(item["status_correct"] for item in details), 6),
        "positive_answer_rate": round(mean(item["status_correct"] for item in positives), 6)
        if positives else None,
        "negative_refusal_accuracy": round(mean(item["status_correct"] for item in negatives), 6)
        if negatives else None,
        "citation_recall_answered": round(mean(item["citation_recall"] for item in positives if item["citation_recall"] is not None), 6)
        if any(item["citation_recall"] is not None for item in positives) else None,
        "latency_ms": {"p50": percentile(latencies, 0.5), "p95": percentile(latencies, 0.95)},
        "details": details,
    }


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Evaluate uploaded knowledge-base questions")
    parser.add_argument("--questions", type=Path, default=settings.project_root / "data/eval/uploaded-knowledge.v1.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--answers", action="store_true", help="Call the configured LLM; may incur cost")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    cases, active_by_id = resolve_cases(settings.knowledge_base_path, load_cases(args.questions))
    encoder = get_embedding_encoder()
    indexes: dict[str, dict[str, Any]] = {name: {} for name in ("bm25", "dense", "rrf", "rerank")}
    scorer = BGERerankerScorer()
    for kb_id, active in active_by_id.items():
        bm25 = BM25Index.load(active / "bm25.json")
        dense = DenseIndex.load(active / "dense.json", encoder)
        graph = HeadingGraphIndex(bm25.documents)
        rrf = MultiRouteRRFRetriever({"bm25": (bm25, 1.0), "dense": (dense, 1.0), "graph": (graph, 0.5)})
        indexes["bm25"][kb_id] = bm25
        indexes["dense"][kb_id] = dense
        indexes["rrf"][kb_id] = rrf
        indexes["rerank"][kb_id] = RerankingRetriever(rrf, scorer, rerank_k=10)
    report: dict[str, Any] = {
        "schema_version": 1,
        "dataset": str(args.questions),
        "top_k": args.top_k,
        "retrieval": {
            name: evaluate_retrieval(retrievers, cases, args.top_k)
            for name, retrievers in indexes.items()
        },
    }
    if args.answers:
        report["answers"] = evaluate_answers(get_knowledge_agent_service(get_answer_service()), cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "report": str(args.output),
        "retrieval": {
            name: {key: value for key, value in result.items() if key != "details"}
            for name, result in report["retrieval"].items()
        },
        "answers": {key: value for key, value in report.get("answers", {}).items() if key != "details"},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
