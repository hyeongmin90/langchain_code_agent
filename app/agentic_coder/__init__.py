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
    TokenUsage,
    RequirementAnalysisResult,
    SkeletonCodeList,
    BuildGradleKts,
)

from .agents import (
    requirement_analyst_agent,
    code_file_generator_agent,
    skeleton_code_generator_agent,
    file_writer_node,
    setup_project,
)

from .workflow import (
    create_agentic_coder_workflow,
    generate_java_spring_boot_project
)

__all__ = [
    # Schemas
    "AgenticCoderState",
    "RequirementAnalysisResult",
    "SkeletonCodeList",
    "BuildGradleKts",
    "TokenUsage",
    # Agents
    "requirement_analyst_agent",
    "skeleton_code_generator_agent",
    "code_file_generator_agent",
    "file_writer_node",
    "setup_project",
    # Workflow & Orchestrator
    "create_agentic_coder_workflow",
    "generate_java_spring_boot_project",
]

__version__ = "1.0.0"

