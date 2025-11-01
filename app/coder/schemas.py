from typing import TypedDict, List, Optional, Literal
from pydantic import BaseModel, Field

# ============================================
# Epic 관련 스키마
# ============================================

class Epic(BaseModel):
    """에픽: 큰 기능 단위"""
    id: str = Field(description="에픽 고유 식별자 (예: epic-1, epic-2)")
    title: str = Field(description="에픽 제목 (예: User Domain (Auth))")
    description: str = Field(description="에픽 상세 설명")
    priority: int = Field(description="우선순위 (낮을수록 먼저 구현)")

class EpicList(BaseModel):
    """에픽 리스트"""
    epics: List[Epic] = Field(description="에픽 목록")


# ============================================
# 프로젝트 설정 관련 스키마
# ============================================

class ProjectSetup(BaseModel):
    """프로젝트 설정"""
    project_name: str = Field(description="프로젝트 이름 (예: TodoList)")
    build_gradle_kts: str = Field(description="build.gradle.kts 파일 전체 내용")
    application_yml: str = Field(description="application.yml 파일 전체 내용")

# ============================================
# Task 관련 스키마
# ============================================

class Task(BaseModel):
    """작업: 파일 단위 작업"""
    id: str = Field(description="작업 고유 식별자")
    file_name: str = Field(description="생성할 파일명 (예: User.java)")
    file_path: str = Field(description="파일 경로 (예: src/main/java/com/example/domain/)")
    description: str = Field(description="파일 설명 및 구현할 내용")
    dependencies: List[str] = Field(
        default=[],
        description="의존하는 다른 Task ID 목록"
    )

class TaskList(BaseModel):
    """작업 목록"""
    epic_id: str = Field(description="이 Task List가 속한 Epic ID")
    tasks: List[Task] = Field(description="작업 목록")

# ============================================
# 코드 생성 결과
# ============================================

class GeneratedFile(BaseModel):
    """생성된 파일"""
    task_id: str = Field(description="작업 ID")
    file_name: str = Field(description="파일명")
    file_path: str = Field(description="전체 파일 경로")
    code_content: str = Field(description="파일 코드 내용")
    status: Literal["success", "failed"] = Field(description="생성 상태")
    error_message: Optional[str] = Field(default=None, description="에러 메시지 (실패시)")

class CodeGenerationResult(BaseModel):
    """코드 생성 결과"""
    epic_id: str = Field(description="에픽 ID")
    generated_files: List[GeneratedFile] = Field(description="생성된 파일 목록")

# ============================================
# 검증 결과
# ============================================

class VerificationResult(BaseModel):
    """검증 결과"""
    epic_id: str = Field(description="검증한 에픽 ID")
    status: Literal["SUCCESS", "FAILED"] = Field(description="검증 상태")
    build_log: str = Field(description="빌드 로그")
    error_files: Optional[List[str]] = Field(
        default=None,
        description="에러가 발생한 파일 목록"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="에러 메시지"
    )

# ============================================
# Agent State
# ============================================

class TokenUsage(BaseModel):
    """토큰 사용량"""
    step_name: str = Field(description="단계 이름")
    input_tokens: int = Field(description="입력 토큰 수")
    output_tokens: int = Field(description="출력 토큰 수")
    total_tokens: int = Field(description="총 토큰 수")

class MultiAgentState(TypedDict):
    """멀티 에이전트 시스템 전체 상태"""
    
    project_uuid: str

    project_dir: str
    # 입력
    user_request: str

    analyzed_user_request: str

    project_setup_files: Optional[List[GeneratedFile]]
    project_setup_status: Literal["success", "failed"]
    
    # Analyst Agent 산출물
    epic_list: Optional[EpicList]

    current_epic_index: int  # 현재 처리중인 에픽 인덱스

    # Planner Agent 산출물
    current_task_list: Optional[TaskList]
    
    # Coder Agent 산출물
    current_code_result: Optional[CodeGenerationResult]
    
    # Verifier Agent 산출물
    current_verification: Optional[VerificationResult]
    
    # 전체 진행 상태
    completed_epics: List[str]  # 완료된 에픽 ID 목록
    current_status: Literal["analyzing", "setup_project", "planning", "coding", "verifying", "completed", "failed"]
    
    # 재시도 관련
    retry_count: int  # 현재 에픽 재시도 횟수
    max_retries: int  # 최대 재시도 횟수
    
    # 최종 결과
    all_generated_files: List[GeneratedFile]  # 전체 생성된 파일 목록
    final_message: Optional[str]
    
    # 토큰 사용량 추적
    token_usage_list: List[TokenUsage]  # 단계별 토큰 사용량

