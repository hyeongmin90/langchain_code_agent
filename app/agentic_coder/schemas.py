"""
Agentic Coder의 데이터 스키마 정의
"""

from typing import List, Optional, TypedDict, Literal
from pydantic import BaseModel, Field


# ============================================
# 토큰 사용량 모델
# ============================================

class TokenUsage(BaseModel):
    """토큰 사용량"""
    step_name: str = Field(description="단계 이름")
    input_tokens: int = Field(description="입력 토큰 수")
    output_tokens: int = Field(description="출력 토큰 수")
    total_tokens: int = Field(description="총 토큰 수")


# ============================================
# 요구사항 분석 관련 모델
# ============================================

class RequirementAnalysisResult(BaseModel):
    """요구사항 분석 결과"""
    project_name: str = Field(description="프로젝트 이름")
    requirement_analysis_result: str = Field(description="요구사항 분석 결과")


# ============================================
# 프로젝트 설정 관련 모델
# ============================================

class BuildGradleKts(BaseModel):
    """프로젝트 설정"""
    build_gradle_kts: str = Field(description="build.gradle.kts 파일 전체 내용")
    application_yml: str = Field(description="application.yml 파일 전체 내용")

# ============================================
# 스켈레톤 코드 관련 모델
# ============================================

class SkeletonCode(BaseModel):
    """스켈레톤 코드"""
    file_name: str = Field(description="파일명")
    file_path: str = Field(description="파일 경로")
    skeleton_code: str = Field(description="스켈레톤 코드")
    need_to_generate: bool = Field(description="완전한 코드를 생성해야 하는지 여부")
    
class SkeletonCodeList(BaseModel):
    """스켈레톤 코드 목록"""
    skeleton_code_list: List[SkeletonCode] = Field(description="스켈레톤 코드 목록")
    

# ============================================
# State 정의
# ============================================

class AgenticCoderState(TypedDict):
    """
    Agentic Coder 워크플로우의 상태를 관리하는 State
    """
    # 입력
    orchestrator_request: str

    # 요구사항 분석 결과
    requirement_analysis_result: Optional[dict]

    # 스켈레톤 코드 목록
    skeleton_code_list: Optional[dict]
    all_skeleton: Optional[dict]

    # 완료된 스켈레톤 코드 목록
    completed_skeleton_list: Optional[dict]

    # 생성된 코드 파일 목록
    generated_files: Optional[dict]

    # 토큰 사용량 목록
    token_usage_list: List[TokenUsage]