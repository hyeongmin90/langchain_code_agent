"""
Agentic Coder의 데이터 스키마 정의
"""

from typing import List, Optional, TypedDict
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
# 코드 생성 관련 모델 (먼저 정의 - AgenticCoderState에서 참조)
# ============================================

class FilePlan(BaseModel):
    """오케스트라가 계획한 파일 정보"""
    file_name: str = Field(description="파일명")
    file_path: str = Field(description="파일 경로")
    signature: str = Field(description="API 시그니처")
    description: str = Field(description="파일 설명")
    dependencies: Optional[List[str]] = Field(default=None, description="의존하는 파일명들")
    priority: int = Field(description="생성 우선순위 (낮을수록 먼저)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "file_name": "Todo.java",
                "file_path": "src/main/java/com/example/todo/domain",
                "signature": "Class Todo { id: Long, title: String, description: String, priority: Int }",
                "description": "Todo 엔티티 클래스",
                "dependencies": [],
                "priority": 1
            }
        }


# ============================================
# State 정의
# ============================================

class AgenticCoderState(TypedDict):
    """
    Agentic Coder 워크플로우의 상태를 관리하는 State
    """
    # 입력
    user_request: str
    
    # 이전 단계 제안사항
    previous_suggestions: Optional[str]

    # 명세서 작성 단계 (한번에 전체 생성)
    specification: Optional[str]
    
    # 파일 계획 (오케스트라가 관리)
    files_plan: Optional[List[dict]]  # 전체 파일 계획 (dict 형태로 저장)
    current_file_index: int  # 현재 생성 중인 파일 인덱스
    next_file_to_generate: Optional[dict]  # 오케스트라가 지정한 다음 파일
    
    # 코드 작성 단계 (파일 하나씩 생성)
    generated_files: Optional[List[dict]]  # 이미 생성된 파일들
    current_file_code: Optional[str]  # 방금 생성된 파일의 코드
    
    # 정적 리뷰 단계
    review_result: Optional[str]
    review_passed: bool
    issues_found: Optional[List[str]]
    
    # 상태 관리
    current_status: str  # "spec", "code_single", "review", "completed", "failed"
    retry_count: int
    max_retries: int
    
    # 최종 결과
    final_code: Optional[str]
    final_message: Optional[str]

    pre_result: Optional[str]

    token_usage_list: List[TokenUsage]


# ============================================
# 명세서 관련 모델
# ============================================

class Specification(BaseModel):
    """명세서"""
    title: str = Field(description="프로젝트 제목")
    api_signatures: List[FilePlan] = Field(description="API 시그니처 목록")
    technical_stack: str = Field(description="기술 스택")
    
    class Config:
        json_schema_extra = {
            "example": {
                "title": "TodoList",
                "api_signatures": [
                    {
                        "file_name": "Todo.java",
                        "file_path": "src/main/java/com/example/todo/domain",
                        "signature": "Class Todo { id: Long, title: String, description: String, priority: Int }",
                        "description": "Todo 엔티티 클래스",
                        "dependencies": [],
                        "priority": 1
                    },
                    {
                        "file_name": "TodoService.java",
                        "file_path": "src/main/java/com/example/todo/service",
                        "signature": "Class TodoService { getTodo(Long id): Todo, createTodo(Todo todo): Todo }",
                        "description": "Todo 서비스 클래스",
                        "dependencies": ["Todo.java", "TodoRepository.java"],
                        "priority": 2
                    }
                ],
                "technical_stack": "Spring Boot 3.x, Java 17, JPA, H2"
            }
        }


class SingleFileGeneration(BaseModel):
    """단일 파일 생성 결과"""
    file_name: str = Field(description="파일명")
    file_path: str = Field(description="파일 경로")
    code_content: str = Field(description="생성된 코드")
    
    class Config:
        json_schema_extra = {
            "example": {
                "file_name": "TodoService.java",
                "file_path": "src/main/java/com/example/todo/service",
                "code_content": "package com.example.todo.service;\n\n@Service..."
            }
        }


class GeneratedCodeFile(BaseModel):
    """생성된 코드 파일"""
    file_name: str = Field(description="파일명")
    file_path: str = Field(description="파일 경로")
    code_content: str = Field(description="코드 내용")
    description: str = Field(description="파일 설명")
    
    class Config:
        json_schema_extra = {
            "example": {
                "file_name": "TodoController.java",
                "file_path": "src/main/java/com/example/todo/controller",
                "code_content": "package com.example.todo.controller;\n\n@RestController...",
                "description": "Todo REST API 컨트롤러"
            }
        }


class CodeGenerationOutput(BaseModel):
    """코드 생성 결과"""
    files: List[GeneratedCodeFile] = Field(description="생성된 파일 목록")
    summary: str = Field(description="생성 요약")
    
    class Config:
        json_schema_extra = {
            "example": {
                "files": [
                    {
                        "file_name": "TodoController.java",
                        "file_path": "src/main/java/com/example/todo/controller",
                        "code_content": "...",
                        "description": "Todo REST API 컨트롤러"
                    }
                ],
                "summary": "3개 파일 생성: Controller, Service, Repository"
            }
        }


# ============================================
# 정적 리뷰 관련 모델
# ============================================

class CodeIssue(BaseModel):
    """코드 이슈"""
    severity: str = Field(description="심각도 (CRITICAL, MAJOR, MINOR, INFO)")
    file_name: str = Field(description="파일명")
    line_number: Optional[int] = Field(default=None, description="줄 번호")
    issue_type: str = Field(description="이슈 유형 (예: NullPointer, SecurityVulnerability, CodeSmell)")
    description: str = Field(description="이슈 설명")
    suggestion: Optional[str] = Field(default=None, description="개선 제안")
    
    class Config:
        json_schema_extra = {
            "example": {
                "severity": "MAJOR",
                "file_name": "TodoService.java",
                "line_number": 45,
                "issue_type": "NullPointerException",
                "description": "todoRepository.findById() 결과에 대한 null 체크 누락",
                "suggestion": "Optional.orElseThrow()를 사용하여 예외 처리 추가"
            }
        }


class StaticReviewResult(BaseModel):
    """정적 리뷰 결과"""
    passed: bool = Field(description="리뷰 통과 여부")
    issues: List[CodeIssue] = Field(description="발견된 이슈 목록")
    summary: str = Field(description="리뷰 요약")
    recommendations: Optional[List[str]] = Field(default=None, description="전반적인 개선 권장사항")
    
    class Config:
        json_schema_extra = {
            "example": {
                "passed": False,
                "issues": [
                    {
                        "severity": "MAJOR",
                        "file_name": "TodoService.java",
                        "issue_type": "NullPointerException",
                        "description": "Null 체크 누락"
                    }
                ],
                "summary": "2개의 MAJOR 이슈, 3개의 MINOR 이슈 발견",
                "recommendations": ["예외 처리 강화", "로깅 추가", "트랜잭션 관리 개선"]
            }
        }


# ============================================
# 오케스트라 에이전트 관련 모델
# ============================================

class OrchestratorDecision(BaseModel):
    """오케스트라 에이전트의 결정"""
    next_action: str = Field(
        description="다음 행동 (specification_writer, code_generator, static_reviewer, completed, failed)"
    )
    reasoning: str = Field(description="결정 이유 및 분석")
    should_retry: bool = Field(default=False, description="재시도 필요 여부")
    
    # 파일 계획 관련
    next_file: Optional[FilePlan] = Field(default=None, description="다음에 생성할 파일 (code_generator로 갈 때마다)")
    dependent_files: Optional[List[FilePlan]] = Field(default=None, description="의존하는 파일 목록")
    
    # 완료 시 최종 메시지 (completed일 때만 사용)
    final_message: Optional[str] = Field(default=None, description="워크플로우 완료 시 최종 메시지")
    
    suggestions: Optional[str] = Field(default=None, description="다음 단계를 위한 제안사항")
    
    class Config:
        json_schema_extra = {
            "example": {
                "next_action": "code_generator",
                "reasoning": "명세서 완성. 첫 번째 파일인 Entity부터 생성 시작.",
                "should_retry": False,
                "files_plan": [{"file_name": "Todo.java", "priority": 1}],
                "next_file": {"file_name": "Todo.java", "priority": 1},
                "final_message": None,
                "suggestions": "Entity부터 순차적으로 생성"
            }
        }

