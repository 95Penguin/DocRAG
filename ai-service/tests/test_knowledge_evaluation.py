import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.knowledge_evaluation import (
    EvaluationCase,
    ResolvedCase,
    evaluate_answers,
    evaluate_retrieval,
    load_cases,
    resolve_cases,
)


def test_gold_labels_resolve_against_active_chunks(tmp_path: Path) -> None:
    kb_id = "899c0ad7-e6fd-49ea-b31e-12354686e220"
    doc_id = "a1768a63-69ed-4c24-98d3-78d941aa3d43"
    folder = tmp_path / kb_id
    folder.mkdir()
    (folder / "metadata.json").write_text(
        json.dumps({"id": kb_id, "name": "Demo", "owner_id": "owner"}), encoding="utf-8"
    )
    (folder / "documents.json").write_text(
        json.dumps([{"id": doc_id, "filename": "guide.md"}]), encoding="utf-8"
    )
    (folder / "chunks.json").write_text(
        json.dumps([{
            "chunk_id": "chunk-1",
            "source_path": f"{kb_id}/{doc_id}-guide.md",
            "section_path": ["Guide", "Topic"],
            "content": "One verifiable answer.",
        }]), encoding="utf-8"
    )
    case = EvaluationCase.model_validate({
        "id": "q1", "knowledge_base": "Demo", "category": "fact", "question": "Question?",
        "expected_evidence": [{"document": "guide.md", "section_path": ["Guide", "Topic"], "anchor": "verifiable"}],
    })
    resolved, active = resolve_cases(tmp_path, [case])
    assert resolved[0].relevant_chunk_ids == {"chunk-1"}
    assert active[kb_id] == folder

    bad = case.model_copy(update={"expected_evidence": [
        case.expected_evidence[0].model_copy(update={"anchor": "does not exist"})
    ]})
    with pytest.raises(ValueError, match="gold evidence not found"):
        resolve_cases(tmp_path, [bad])


def test_retrieval_report_excludes_refusal_cases_and_records_misses() -> None:
    positive = EvaluationCase(id="yes", knowledge_base="Demo", category="fact", question="yes",
                              expected_evidence=[{"document": "a", "section_path": ["x"], "anchor": "fact"}])
    negative = EvaluationCase(id="no", knowledge_base="Demo", category="out_of_scope",
                              question="no", should_refuse=True)
    cases = [
        ResolvedCase(positive, "kb", "owner", frozenset({"gold"})),
        ResolvedCase(negative, "kb", "owner", frozenset()),
    ]

    class Retriever:
        def search(self, query: str, top_k: int):
            return [SimpleNamespace(chunk_id="wrong"), SimpleNamespace(chunk_id="gold")]

    report = evaluate_retrieval({"kb": Retriever()}, cases, 2)
    assert report["answerable_questions"] == 1
    assert report["recall_at_k"] == 1
    assert report["mrr"] == 0.5
    assert report["details"][1]["failure_type"] == "not_scored_retrieval_only"


def test_answer_report_keeps_semantic_review_manual() -> None:
    case = EvaluationCase(id="q", knowledge_base="Demo", category="fact", question="Q?",
                          expected_evidence=[{"document": "a", "section_path": ["x"], "anchor": "fact"}],
                          required_answer_points=["fact"])

    class AnswerService:
        def answer(self, question: str, history: list[str], kb_ids: list[str], owner: str):
            return SimpleNamespace(status="answered", answer="An answer",
                                   citations=[SimpleNamespace(chunk_id="gold")])

    result = evaluate_answers(AnswerService(), [ResolvedCase(case, "kb", "owner", frozenset({"gold"}))])
    assert result["answerability_accuracy"] == 1
    assert result["citation_recall_answered"] == 1
    assert result["details"][0]["manual_answer_point_review"] == "pending"


def test_duplicate_case_ids_fail(tmp_path: Path) -> None:
    line = '{"id":"x","knowledge_base":"D","category":"c","question":"Q",' \
           '"expected_evidence":[],"should_refuse":true}'
    path = tmp_path / "questions.jsonl"
    path.write_text(f"{line}\n{line}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unique case IDs"):
        load_cases(path)
