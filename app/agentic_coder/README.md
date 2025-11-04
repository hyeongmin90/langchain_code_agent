# Agentic Coder

**3개의 에이전트와 1개의 오케스트라 에이전트로 구성된 자동 코드 생성 시스템**

## 📋 개요

Agentic Coder는 사용자의 간단한 요청을 받아 완전한 Spring Boot 코드를 자동으로 생성하고 리뷰하는 시스템입니다.

## 🤖 에이전트 구성

### 1️⃣ Specification Writer Agent (명세서 작성 에이전트)
- **역할**: 요청사항 → 명세서 작성 (**한번에 전체**)
- **주요 작업**:
  - 사용자 요청 분석 및 요구사항 도출
  - API 엔드포인트 설계 및 시그니처 추출
  - 기술 스택 결정
  - 아키텍처 설계
  - 전체 명세서 완성

### 2️⃣ Code Generator Agent (코드 생성 에이전트)
- **역할**: 명세서 → 코드 작성 (**파일 하나씩**)
- **특징**: 오케스트라가 지정한 **단일 파일만** 생성
- **주요 작업**:
  - 오케스트라가 준 파일 정보로 한 파일 생성
  - 이미 생성된 파일들 참조하여 일관성 유지
  - Entity, Repository, DTO, Service, Controller 등
  - API 시그니처에 정확히 맞는 구현
  - Spring Boot 베스트 프랙티스 준수

### 3️⃣ Static Reviewer Agent (정적 리뷰 에이전트)
- **역할**: 작성된 코드 → 정적 리뷰 (**전체 파일 종합**)
- **주요 작업**:
  - 모든 생성된 파일 종합 리뷰
  - 잠재적 버그 탐지 (NullPointer, Resource Leak 등)
  - 보안 취약점 검사
  - 코드 스멜 탐지
  - 베스트 프랙티스 준수 여부 확인
  - 구체적인 개선 제안

### 🎯 Orchestrator Agent (오케스트라 에이전트) - 에이전틱! ⭐
- **역할**: 전체 워크플로우 조율 + **파일 계획 수립**
- **특징**: LLM 기반 의사결정 (단순 조건문 아님!)
- **주요 작업**:
  - **📋 파일 계획 수립**: 명세서 분석 → 전체 파일 목록 계획
  - **🎯 파일 하나씩 지정**: "다음은 이 파일 만들어" 지시
  - 현재 상태 분석 및 평가
  - 의존성 고려한 생성 순서 결정
  - 각 파일 생성 후 결과 검토
  - 다음 파일 동적 결정
  - 재시도 필요성 판단
  - 확신도와 이유를 포함한 결정

## 🔄 워크플로우 (에이전틱 오케스트레이션)

```
START
  ↓
[Specification Writer] - 명세서 한번에 전체 생성
  ↓
[Orchestrator] - 명세서 분석, 전체 파일 계획 수립
  ↓              ↓ "Todo.java 만들어"
[Code Generator] - Todo.java 생성
  ↓
[Orchestrator] - 결과 확인
  ↓              ↓ "TodoRepository.java 만들어"  
[Code Generator] - TodoRepository.java 생성
  ↓
[Orchestrator] - 결과 확인
  ↓              ↓ "TodoService.java 만들어"
[Code Generator] - TodoService.java 생성
  ↓
... (파일 하나씩 반복)
  ↓
[Orchestrator] - 모든 파일 완료 확인
  ↓              ↓ "리뷰 시작"
[Static Reviewer] - 전체 코드 리뷰
  ↓
END
```

**핵심 특징**:
- 명세서: **한번에 전체** 생성
- 코드: **파일 하나씩** 생성 (오케스트라가 지시)
- 오케스트라: 매번 다음 파일 결정
- 의존성 고려한 순서 (Entity → Repository → Service → Controller)

## 🚀 사용 방법

### 기본 사용

```python
from app.agentic_coder import run_agentic_coder

# 간단한 요청으로 코드 생성
result = run_agentic_coder(
    user_request="""
    간단한 Todo 관리 API를 만들어줘.
    
    필요한 기능:
    - Todo 생성, 조회, 수정, 삭제
    - 제목, 내용, 완료 여부, 우선순위
    """,
    max_retries=2  # 최대 재시도 횟수
)

# 결과 확인
print(result["final_message"])
```

### 코드 파일로 저장

```python
from app.agentic_coder import run_agentic_coder, export_code_to_files

# 코드 생성
result = run_agentic_coder("사용자 관리 API")

# 생성된 코드를 파일로 저장
export_code_to_files(result, output_dir="./generated_code")
```

### 고급 사용

```python
from app.agentic_coder import (
    create_agentic_coder_workflow,
    AgenticCoderState
)

# 워크플로우 직접 생성 및 실행
workflow = create_agentic_coder_workflow()

initial_state = {
    "user_request": "블로그 API를 만들어줘",
    "current_status": "spec",
    "retry_count": 0,
    "max_retries": 3,
    # ... 기타 상태
}

final_state = workflow.invoke(initial_state)
```

## 📦 주요 컴포넌트

### Schemas (`schemas.py`)
- `AgenticCoderState`: 워크플로우 상태 관리
- `Specification`: 명세서 데이터 모델
- `APISignature`: API 시그니처 정보
- `CodeGenerationOutput`: 코드 생성 결과
- `StaticReviewResult`: 정적 리뷰 결과

### Agents (`agents.py`)
- `specification_writer_agent()`: 명세서 작성
- `code_generator_agent()`: 코드 생성
- `static_reviewer_agent()`: 정적 리뷰

### Workflow (`workflow.py`)
- `orchestrator_router()`: 오케스트라 라우터
- `create_agentic_coder_workflow()`: 워크플로우 생성
- `run_agentic_coder()`: 실행 함수
- `export_code_to_files()`: 파일 저장

## ⚙️ 설정

### 환경 변수
```bash
# OpenAI API Key (Code Generator, Static Reviewer)
OPENAI_API_KEY=your_openai_key

# Google API Key (선택사항)
GOOGLE_API_KEY=your_google_key
```

### LLM 모델 변경
`agents.py`의 `get_llm()` 함수에서 모델 변경 가능:
```python
def get_llm(model: str = "gpt-4o-mini"):
    return ChatOpenAI(model=model)
```

## 📊 출력 예시

```
================================================================================
🚀 Agentic Coder 시스템 시작
================================================================================
📝 사용자 요청: 간단한 Todo 관리 API를 만들어줘
🔄 최대 재시도 횟수: 2
================================================================================

================================================================================
📝 [Specification Writer Agent] 명세서 작성 시작
================================================================================
✅ 명세서 작성 완료
  - 프로젝트: Todo Management API
  - API 개수: 5개
  - 기술 스택: Spring Boot 3.x, Java 17, JPA, H2

📋 API 시그니처 목록:
  - POST /api/todos: Todo 생성
  - GET /api/todos: Todo 목록 조회
  - GET /api/todos/{id}: Todo 조회
  - PUT /api/todos/{id}: Todo 수정
  - DELETE /api/todos/{id}: Todo 삭제

================================================================================
💻 [Code Generator Agent] 코드 생성 시작
================================================================================
✅ 코드 생성 완료
  - 생성된 파일 수: 7개
  - 요약: Entity, Repository, DTO, Service, Controller, Exception Handler

📄 생성된 파일 목록:
  - com/example/todo/domain/Todo.java
  - com/example/todo/repository/TodoRepository.java
  - com/example/todo/dto/TodoRequestDto.java
  - com/example/todo/dto/TodoResponseDto.java
  - com/example/todo/service/TodoService.java
  - com/example/todo/controller/TodoController.java
  - com/example/todo/exception/GlobalExceptionHandler.java

================================================================================
🔍 [Static Reviewer Agent] 정적 리뷰 시작
================================================================================
✅ 정적 리뷰 완료
  - 통과 여부: ✅ PASS
  - 발견된 이슈: 2개
  - 요약: 2개의 MINOR 이슈 발견, 전반적으로 양호

================================================================================
🎉 Agentic Coder 시스템 완료
================================================================================
📢 ✅ 모든 단계 완료! 코드 생성 및 리뷰 통과. 7개 파일 생성됨.
```

## 🔧 재시도 메커니즘

- Static Reviewer Agent가 FAIL을 반환하면 Code Generator로 돌아가 재시도
- 최대 재시도 횟수를 초과하면 현재 코드로 완료 처리
- CRITICAL 이슈: 0개, MAJOR 이슈: 3개 이하일 때 통과

## 🛠️ 확장 가능성

### 새로운 에이전트 추가
```python
def new_agent(state: AgenticCoderState) -> Dict[str, Any]:
    # 에이전트 로직
    return {"current_status": "next_step"}

# workflow.py에서 노드 추가
workflow.add_node("new_agent", new_agent)
workflow.add_edge("previous_agent", "new_agent")
```

### 커스텀 리뷰 기준
`static_reviewer_agent()`의 `system_prompt` 수정

## 📝 라이센스

MIT License

## 🤝 기여

Pull Request 환영합니다!

