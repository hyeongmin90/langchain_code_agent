import sys
import os
from typing import Any, Optional

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# 프로젝트 루트 경로를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.retriever import query_hybrid
from agent.state import AgentState
from agent.prompts import (
    ANALYZE_PROMPT,
    REWRITE_PROMPT,
    GENERATE_PROMPT,
    GRADE_PROMPT,
    SUPPORTED_CATEGORIES,
)


# ──────────────────────────────────────────────
# Structured Output 스키마 정의
# ──────────────────────────────────────────────
class AnalyzeOutput(BaseModel):
    category: Optional[str] = Field(default=None, description="The most relevant Spring documentation category, or null")
    reason: str = Field(description="Brief reason for the category selection")

class GradeOutput(BaseModel):
    should_rewrite: bool = Field(description="Whether the retrieved docs are insufficient and the query needs to be rewritten")

class RewriteOutput(BaseModel):
    rewritten_query: str = Field(description="Rewritten search query optimized for semantic search")
    category: Optional[str] = Field(default=None, description="The most relevant Spring documentation category for filtering")

class GenerateOutput(BaseModel):
    answer: str = Field(description="Final answer to the user's question in Korean")

def _get_llm(temperature: float = 0) -> ChatOpenAI:
    return ChatOpenAI(model="gpt-5-mini", temperature=temperature)

def _format_docs(docs: list) -> str:
    """검색 결과 문서를 컨텍스트 문자열로 포맷팅"""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        header = doc.metadata.get("header", "N/A")
        category = doc.metadata.get("category", "Unknown")
        formatted.append(
            f"[Document {i}]\n"
            f"Source: {source}\n"
            f"Header: {header}\n"
            f"Category: {category}\n"
            f"Content:\n{doc.page_content}\n"
        )
    return "\n".join(formatted)

# ──────────────────────────────────────────────
# Node: Grade Docs
#   - 검색 결과의 품질을 판단하여 재검색 여부 결정
# ──────────────────────────────────────────────
def grade_docs_node(state: AgentState) -> dict[str, Any]:
    # 이미 재작성을 한 번 수행했다면 더 이상 재작성하지 않음 (루프 방지)
    if state.get("is_rewritten", False):
        return {"should_rewrite": False}

    llm = _get_llm().with_structured_output(GradeOutput)
    chain = GRADE_PROMPT | llm

    context = _format_docs(state["documents"])
    result: GradeOutput = chain.invoke({
        "question": state["question"],
        "context": context,
    })

    return {"should_rewrite": result.should_rewrite}


# ──────────────────────────────────────────────
# Node 2: Rewrite
#   - 정보가 부족할 때만 카테고리 추론 + 쿼리 최적화 수행
# ──────────────────────────────────────────────
def rewrite_node(state: AgentState) -> dict[str, Any]:
    # 1. 먼저 카테고리 분석 수행
    analyze_llm = _get_llm().with_structured_output(AnalyzeOutput)
    analyze_chain = ANALYZE_PROMPT | analyze_llm
    
    categories_str = "\n".join(f"  - {c}" for c in SUPPORTED_CATEGORIES)
    analyze_result: AnalyzeOutput = analyze_chain.invoke({
        "question": state["question"],
        "categories": categories_str,
    })
    
    category = analyze_result.category
    if category not in SUPPORTED_CATEGORIES:
        category = None

    # 2. 분석된 카테고리를 바탕으로 쿼리 재작성
    rewrite_llm = _get_llm().with_structured_output(RewriteOutput)
    rewrite_chain = REWRITE_PROMPT | rewrite_llm

    rewrite_result: RewriteOutput = rewrite_chain.invoke({
        "question": state["question"],
        "category": category or "null",
    })

    return {
        "rewritten_query": rewrite_result.rewritten_query,
        "category": category, # 이제서야 카테고리 필터링 적용
        "is_rewritten": True,
    }


# ──────────────────────────────────────────────
# Node 3: Retrieve
#   - 첫 번째 시도는 원본 질문 + 전체 검색
#   - 두 번째 시도는 재작성 쿼리 + 카테고리 필터링
# ──────────────────────────────────────────────
def retrieve_node(state: AgentState) -> dict[str, Any]:
    query = state.get("rewritten_query") or state["question"]
    category = state.get("category")

    docs = query_hybrid(
        query=query,
        k=5,
        category=category,
        use_reranker=True,
    )

    return {"documents": docs}


# ──────────────────────────────────────────────
# Node 4: Generate
#   - 검색 결과를 바탕으로 최종 답변 생성
# ──────────────────────────────────────────────
def generate_node(state: AgentState) -> dict[str, Any]:
    llm = _get_llm(temperature=0).with_structured_output(GenerateOutput)
    chain = GENERATE_PROMPT | llm

    context = _format_docs(state["documents"])
    result: GenerateOutput = chain.invoke({
        "question": state["question"],
        "context": context,
    })

    return {"answer": result.answer}
