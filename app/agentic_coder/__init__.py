"""
Agentic Coder 모듈

3개의 에이전트를 통한 자동 코드 생성 시스템:
1. Specification Writer Agent - 요청사항 → 명세서 작성 (API 시그니처 추출 및 주입)
2. Code Generator Agent - 명세서 → 코드 작성
3. Static Reviewer Agent - 작성된 코드 → 정적 리뷰

오케스트라 에이전트가 전체 워크플로우를 조율합니다.
"""

from .schemas import (
    AgenticCoderState,
    Specification,
    APISignature,
    FilePlan,
    SingleFileGeneration,
    GeneratedCodeFile,
    StaticReviewResult,
    CodeIssue,
    OrchestratorDecision,
    TokenUsage,
)

from .agents import (
    specification_writer_agent,
    code_generator_agent,
    static_reviewer_agent,
    orchestrator_agent,
)

from .workflow import (
    orchestrator_router,
    create_agentic_coder_workflow,
    run_agentic_coder,
    export_code_to_files,
)

__all__ = [
    # Schemas
    "AgenticCoderState",
    "Specification",
    "APISignature",
    "FilePlan",
    "SingleFileGeneration",
    "GeneratedCodeFile",
    "StaticReviewResult",
    "CodeIssue",
    "OrchestratorDecision",
    "TokenUsage",
    
    # Agents
    "specification_writer_agent",
    "code_generator_agent",
    "static_reviewer_agent",
    
    # Workflow & Orchestrator
    "orchestrator_agent",
    "orchestrator_router",
    "create_agentic_coder_workflow",
    "run_agentic_coder",
    "export_code_to_files",
]

__version__ = "1.0.0"

