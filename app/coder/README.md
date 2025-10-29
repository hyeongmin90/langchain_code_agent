# 멀티 에이전트 코드 생성 시스템

4개의 전문화된 에이전트가 협력하여 Spring Boot 프로젝트를 자동으로 생성합니다.

## 🤖 에이전트 구성

### 1. 📊 Analyst Agent (분석 에이전트)
- **역할**: 전략가
- **입력**: 사용자의 모호한 요청
- **출력**: Epic List (간결한 에픽 목록)
- **기능**:
  - 사용자 요청을 도메인 단위로 분해
  - 우선순위 설정
  - 첫 번째 에픽만 Planner Agent에게 전달

### 2. 📋 Planner Agent (계획 에이전트)
- **역할**: 설계자
- **입력**: Epic 1개
- **출력**: Task List (파일 목록)
- **기능**:
  - 에픽을 구현하는 데 필요한 모든 파일 정의
  - 파일 간 의존성 설정
  - 구현 순서 결정

### 3. 💻 Coder Agent (코더 에이전트)
- **역할**: 작업자
- **입력**: Task List
- **출력**: 생성된 파일들
- **기능**:
  - Task를 순회하며 파일을 순차적으로 생성
  - LLM을 이용해 파일별 코드 생성
  - 파일 시스템에 저장

### 4. 🔍 Verifier Agent (검증 에이전트)
- **역할**: 품질 검증자
- **입력**: 도메인 단위 코드
- **출력**: 검증 결과 (SUCCESS/FAILED)
- **기능**:
  - 빌드 실행 및 검증
  - 실패 시 Coder Agent에게 수정 요청
  - 성공 시 다음 에픽으로 진행

## 🔄 워크플로우

```
사용자 요청
    ↓
┌─────────────────────────────────────────┐
│  1. Analyst Agent                       │
│  → Epic List 생성                        │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  2. Planner Agent                       │
│  → Task List 생성 (Epic 1개)             │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  3. Coder Agent                         │
│  → 파일 생성 (Task 순회)                  │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  4. Verifier Agent                      │
│  → 검증 (빌드)                           │
└─────────────────────────────────────────┘
    ↓
   성공? ──────→ 다음 Epic으로 (2번으로)
    │
   실패? ──────→ 재시도 (3번으로)
    │
  최대 재시도 초과? → 다음 Epic으로
```

## 📦 설치

```bash
# 가상환경 활성화
.\venv\Scripts\activate

# 이미 requirements.txt가 있으므로 설치 완료되어 있어야 함
```

## 🚀 사용법

### 방법 1: 직접 실행

```bash
cd app/coder
python -m coder
```

### 방법 2: 모듈로 실행

```bash
python -m app.coder.coder
```

### 방법 3: 코드에서 사용

```python
from app.coder import run_multi_agent_system

result = run_multi_agent_system(
    "회원가입, 로그인, 게시판 기능이 있는 블로그 MVP를 만들어줘"
)
```

## 📝 예시

### 입력
```
요청사항을 입력하세요: 회원가입, 로그인, 게시판 기능이 있는 블로그 MVP
```

### 출력 (Epic List)
```
✅ 생성된 Epic 목록 (4개):
  - [epic-1] Project Setup (우선순위: 1)
  - [epic-2] User Domain (Auth) (우선순위: 2)
  - [epic-3] Post Domain (Core) (우선순위: 3)
  - [epic-4] Comment Domain (Sub) (우선순위: 4)
```

### 생성된 파일 구조 예시
```
generated/
├── src/
│   └── main/
│       └── java/
│           └── com/
│               └── example/
│                   ├── domain/
│                   │   ├── user/
│                   │   │   ├── User.java
│                   │   │   ├── UserRepository.java
│                   │   │   ├── UserService.java
│                   │   │   └── UserController.java
│                   │   └── post/
│                   │       ├── Post.java
│                   │       ├── PostRepository.java
│                   │       ├── PostService.java
│                   │       └── PostController.java
│                   └── config/
│                       └── SecurityConfig.java
└── application.yml
```

## 🎯 주요 특징

1. **간결한 산출물**
   - 각 에이전트는 명확하고 간결한 결과만 생성
   - 불필요한 설명 제거

2. **단계적 진행**
   - Epic 단위로 진행
   - Epic 내에서는 Task 단위로 진행

3. **자동 검증**
   - 도메인 단위로 빌드 검증
   - 실패 시 자동 재시도

4. **재시도 메커니즘**
   - 최대 3회 재시도
   - 재시도 초과 시 다음 Epic으로 진행

## ⚙️ 환경 변수

`.env` 파일에 다음을 설정하세요:

```
GOOGLE_API_KEY=your_api_key_here
```

## 🔧 설정

### 재시도 횟수 변경
`app/coder/schemas.py`의 `max_retries` 기본값을 수정하거나, 실행 시 초기 상태에서 설정할 수 있습니다.

### LLM 모델 변경
`app/coder/agents.py`의 각 에이전트에서 모델을 변경할 수 있습니다:

```python
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",  # 원하는 모델로 변경
    temperature=0.7
)
```

## 📊 상태 관리

시스템은 다음 상태를 관리합니다:

- `user_request`: 사용자 요청
- `epic_list`: 전체 Epic 목록
- `current_epic_index`: 현재 처리 중인 Epic 인덱스
- `current_task_list`: 현재 Task 목록
- `current_code_result`: 현재 코드 생성 결과
- `current_verification`: 현재 검증 결과
- `completed_epics`: 완료된 Epic 목록
- `retry_count`: 재시도 횟수
- `all_generated_files`: 전체 생성된 파일 목록

## 🐛 문제 해결

### API 키 오류
```
❌ 오류: GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다.
```
→ `.env` 파일에 `GOOGLE_API_KEY`를 설정하세요.

### 빌드 실패
검증 단계에서 빌드가 실패하면 자동으로 재시도합니다. 최대 3회 재시도 후 다음 Epic으로 진행합니다.

## 📄 라이센스

이 프로젝트는 MIT 라이센스를 따릅니다.

