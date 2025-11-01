"""
4개의 에이전트 구현:
1. Analyst Agent - 사용자 요청을 Epic List로 분해
2. Planner Agent - Epic을 Task List로 분해
3. Coder Agent - Task List를 파일로 생성
4. Verifier Agent - 도메인 단위 검증
"""

import os
import uuid
import shutil
import zipfile
from time import sleep
from pathlib import Path
from typing import Dict, Any, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from .schemas import (
    MultiAgentState,
    EpicList,
    TaskList,
    GeneratedFile,
    CodeGenerationResult,
    VerificationResult,
    TokenUsage,
    ProjectSetup,
)

def create_file(task_id: str, file_path: Path, file_name: str, content: str) -> GeneratedFile:
    """
    파일을 생성하고 GeneratedFile 객체를 반환합니다.
    
    Args:
        task_id: 작업 ID
        file_path: 전체 파일 경로 (Path 객체)
        file_name: 파일명
        content: 파일 내용
    
    Returns:
        GeneratedFile 객체
    """
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 이스케이프 시퀀스를 실제 문자로 변환 (LLM이 \n을 문자열로 반환하는 경우 대비)
        # encode().decode('unicode_escape')를 사용하여 \n, \t 등을 실제 문자로 변환
        if '\\n' in content or '\\t' in content:
            content = content.encode().decode('unicode_escape')
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"  ✅ 성공: {file_path}")
        
        return GeneratedFile(
            task_id=task_id,
            file_name=file_name,
            file_path=str(file_path),
            code_content=content,
            status="success"
        )
            
    except Exception as e:
        print(f"  ❌ 실패: {file_path} - {str(e)}")
        return GeneratedFile(
            task_id=task_id,
            file_name=file_name,
            file_path=str(file_path),
            code_content="",
            status="failed",
            error_message=str(e)
        )

def get_llm(model: str = "gemini-2.5-pro"):
    # return ChatGoogleGenerativeAI(model=model)
    return ChatOpenAI(model="gpt-4o-mini")


def extract_token_usage(result, step_name: str) -> TokenUsage:
    """
    LLM 응답에서 직접 토큰 정보를 추출합니다.
    
    Args:
        result: LLM 응답 객체 (AIMessage 등)
        step_name: 단계 이름
    
    Returns:
        TokenUsage 객체
    """
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    
    # AIMessage의 response_metadata에서 추출
    if hasattr(result, 'usage_metadata'):
        response_metadata = result.usage_metadata
        input_tokens = response_metadata.get('input_tokens', 0)
        output_tokens = response_metadata.get('output_tokens', 0)
        total_tokens = response_metadata.get('total_tokens', 0)
        
    
    token_usage = TokenUsage(
        step_name=step_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens
    )
    
    print(f"📊 토큰 사용량 - {step_name}: 입력={input_tokens:,}, 출력={output_tokens:,}, 총={total_tokens:,}")
    
    return token_usage

# ============================================
# 0. Analyze User Request (사용자 요청 분석)
# ============================================

def analyze_user_request(state: MultiAgentState) -> str:
    """
    사용자 요청을 분석하여 Epic List를 생성합니다.
    """

    print("\n" + "="*80)
    print("🔍 [Analyze User Request] 사용자 요청 분석 시작")
    print("="*80)

    user_request = state["user_request"]

    llm = get_llm()

    system_prompt = """
당신은 **요구사항 분석 전문가**입니다.

사용자의 간략한 요청을 **실제 구현 가능한 구체적인 요구사항**으로 확장하는 것이 당신의 임무입니다.

### 핵심 원칙:

1. **명시된 기능을 구체화**하세요
   - 세부 사항, 입력/출력, 제약사항을 명확히 정의

2. **암시된 기능을 발굴**하세요
   - 명시되지 않았지만 필요한 기능들을 식별
   - 예: 게시판 → CRUD, 페이징, 검색 / 사용자 → 회원가입, 로그인, 권한 관리

3. **필요한 공통 기능을 판단**하세요
   - 인증/인가, 예외 처리, 데이터 검증 등
   - 프로젝트 규모에 맞게 선택적으로 포함

### 출력 형식:
추가적인 의견이나 설명은 작성하지 말고, 분석된 요구사항만 작성하라.
또한 사람이 아닌 LLM이 읽을 수 있도록 작성하라.
토큰 사용량을 최소화 하기 위해 필요없는 내용은 제거하며, 필요한 내용을 압축하여 작성하라.
`
자유롭게 작성하되, 다음 항목을 포함하세요:
- 프로젝트 개요 및 주요 기능
- 기능 요구사항 (도메인별)
- 필요시 간단한 API 엔드포인트 및 공통 기능 예시

**중요**: 프로젝트 규모와 복잡도에 맞게 적절히 판단하여 작성하세요.
"""
    prompt = ChatPromptTemplate([
        ("system", system_prompt),
        ("human", "사용자 요청: {request}")
    ])
    chain = prompt | llm
    result = chain.invoke({"request": user_request})

    print(result.content)

    token_usage = extract_token_usage(result, f"Analyze User Request Agent")
    token_usage_list = state.get("token_usage_list", [])
    token_usage_list.append(token_usage)

    return {
        "analyzed_user_request": result.content,
        "token_usage_list": token_usage_list
    }

# ============================================
# 1. Setup Project (프로젝트 설정)
# ============================================

def save_initial_setup_files(
    project_name: str,
    dest_dir: Path,
    build_gradle_content: str,
    application_yml_content: str
):
    """
    초기 설정 파일 4개를 생성하고 저장합니다.
    
    Args:
        project_name: 프로젝트 이름 (예: TodoList)
        dest_dir: 프로젝트 루트 디렉토리
        build_gradle_content: build.gradle.kts 파일 내용
        application_yml_content: application.yml 파일 내용
    
    Returns:
        생성된 파일 목록
    """
    print("\n[파일 생성] 초기 설정 파일 생성 중...")
    
    lower_project_name = project_name.lower()
    generated_files = []
    
    settings_gradle_content = f'rootProject.name = "{lower_project_name}"\n'
    application_java_content = f"""package com.example.{lower_project_name};

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class {project_name}Application {{

    public static void main(String[] args) {{
        SpringApplication.run({project_name}Application.class, args);
    }}
}}
"""
        
        # 2. 파일 설정
    file_configs = [
        {
            "id": "setup-1",
            "name": "build.gradle.kts",
            "path": dest_dir,
            "content": build_gradle_content
        },
        {
            "id": "setup-2",
            "name": "settings.gradle.kts",
            "path": dest_dir,
            "content": settings_gradle_content
        },
        {
            "id": "setup-3",
            "name": "application.yml",
            "path": dest_dir / "src" / "main" / "resources",
            "content": application_yml_content
        },
        {
            "id": "setup-4",
            "name": f"{project_name}Application.java",
            "path": dest_dir / "src" / "main" / "java" / "com" / "example" / lower_project_name,
            "content": application_java_content
        }
    ]
    
    for config in file_configs:
        generated_file = create_file(config["id"], config["path"] / config["name"], config["name"], config["content"])
        generated_files.append(generated_file)
        
    
    return generated_files

def setup_project(state: MultiAgentState) -> Dict[str, Any]:
    """
    프로젝트 설정 및 초기 파일 생성
    - 프로젝트 디렉토리 생성
    - 프로젝트 이름 및 의존성 결정
    - 초기 설정 파일 생성 (build.gradle.kts, settings.gradle.kts, application.yml, Application.java)
    """
    print("\n" + "="*80)
    print("🔧 [Setup Project] 프로젝트 설정 및 초기 파일 생성 시작")
    print("="*80)
    
    # 프로젝트 디렉토리 생성
    project_uuid = str(uuid.uuid4())
    zip_src = Path(__file__).parent.parent / "springTemplate" / "demo.zip"    
    project_dir = Path(__file__).parent.parent.parent
    dest_dir = project_dir / "generated" / project_uuid
    dest_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_src, "r") as zip_ref:
        zip_ref.extractall(dest_dir)
    
    llm = get_llm()
    token_usage_list = state.get("token_usage_list", [])
    generated_files = []
    
    # 1단계: 프로젝트 이름과 의존성 결정
    print("\n[1단계] 프로젝트 이름 및 의존성 결정...")
    system_prompt = """
    당신은 소프트웨어 프로젝트의 **프로젝트 설정 전문가**입니다.
    사용자의 요청을 바탕으로 프로젝트의 이름을 결정하고, 필요한 설정 파일을 생성해야 합니다.
    
    프로젝트 이름 작성 규칙:
    - 길고 복잡한 이름 대신, 간단명료하게 단어 위주로 작성합니다.
    - "SimpleTodolist" 처럼 형용사 등을 붙이기보단, 핵심 도메인만 사용 (예: Todolist, Blog 등).
    - 불필요한 접두어/접미어/형용사는 제외합니다.
    - 영어 단어만 사용하고, 공백 없이 카멜표기법으로 작성합니다.
    - 너무 축약/생략하지 마시고, 사용자의 요구가 드러나는 명사를 충실하게 씁니다.

    ### 파일 1: build.gradle.kts
    - Kotlin DSL 문법 사용
    - Spring Boot 3.x 버전 사용
    - Java 17 사용
    - 기본 의존성: Spring Web, Spring Data JPA, H2 Database, Lombok, Spring Boot Starter Test
    - 추가로 필요한 의존성도 모두 포함

    ### 파일 2: application.yml
    - 서버 포트 8080
    - H2 데이터베이스 설정 (콘솔 활성화, 인메모리 DB)
    - JPA 설정 (hibernate ddl-auto: create-drop, show-sql: true)
    - 로깅 레벨 설정
    - 이외 필요한 설정도 모두 포함하라.

    ### 중요:
    - 각 파일의 **순수한 코드만** 출력하세요 (설명, 마크다운 코드 블록 사용 금지)
    - 주석은 필요한 경우에만 최소한으로 작성
    - 줄바꿈은 실제 줄바꿈을 사용하세요 (\\n 문자열이 아닌 실제 개행)

    ### 출력 형식:
    출력 형식은 주어진 Pydantic 모델 형식으로 출력합니다.
    파일 내용은 실제 줄바꿈이 포함된 멀티라인 문자열로 작성하세요.
    """

    prompt = ChatPromptTemplate([
        ("system", system_prompt),
        ("human", "사용자 요청: {user_request}")
    ])
    chain = prompt | llm.with_structured_output(ProjectSetup, include_raw=True) 

    response = chain.invoke({
        "user_request": state["analyzed_user_request"]
    })

    project_setup = response["parsed"]
    raw_message = response["raw"]

    token_usage = extract_token_usage(raw_message, "Setup Project - 기본 설정")
    token_usage_list.append(token_usage)
    
    project_name = project_setup.project_name
    
    # 파일 생성
    generated_files = save_initial_setup_files(
        project_name=project_name,
        dest_dir=dest_dir,
        build_gradle_content=project_setup.build_gradle_kts,
        application_yml_content=project_setup.application_yml
    )
    
    return {
        "project_uuid": project_uuid,
        "project_dir": str(dest_dir),
        "project_name": project_name.lower(),
        "project_setup_files": generated_files,
        "token_usage_list": token_usage_list
    }


def verify_project_setup(state: MultiAgentState) -> Dict[str, Any]:
    """
    프로젝트 설정을 검증합니다.
    """
    print("\n" + "="*80)
    print("🔍 [Verify Project Setup] 프로젝트 설정 검증 시작")
    print("="*80)
    
    setup_files = state["project_setup_files"]

    for setup_file in setup_files:
        if setup_file.status == "failed":
            print(f"❌ 프로젝트 설정 생성 실패: {setup_file.file_name} - {setup_file.error_message}")
            return {
                "project_setup_status": "failed"
            }

    return {
        "project_setup_status": "success"
    }

# ============================================
# 2. Analyst Agent (분석 에이전트)
# ============================================

def analyst_agent(state: MultiAgentState) -> Dict[str, Any]:
    """
    역할: 전략가
    입력: 1차 분석된 사용자 요청
    출력: Epic List (간결한 에픽 목록)
    """
    print("\n" + "="*80)
    print("🎯 [Analyst Agent] 분석 시작")
    print("="*80)
    
    user_request = state["analyzed_user_request"]
    print(f"사용자 요청: {user_request}\n")
    
    llm = get_llm()
    
    system_prompt = """
당신은 소프트웨어 프로젝트의 **전략가**입니다.

사용자의 1차 분석된 요청을 받아, **간결한 '에픽(Epic) 목록'**으로 분해하는 것이 당신의 임무입니다.

### 에픽이란?
- 큰 기능 단위 (도메인 단위)
- 예: "User Domain (Auth)", "Post Domain (Core)", "Comment Domain (Sub)"

### 분석 원칙:
1. 사용자 요청을 도메인별로 분해합니다
2. 각 에픽은 **독립적으로 구현 가능**해야 합니다
3. 우선순위를 명확히 정합니다 (낮을수록 먼저 구현)
4. 에픽간의 중복이 존재해서는 안됩니다. 연관성이 있는 에픽은 하나의 에픽으로 처리합니다.
   올바른 예시: "User Domain (Auth)", "Post Domain (Core)", "Comment Domain (Sub)"
   잘못된 예시: "User Domain (Post)", "User Domain (Delete)", "User Domain (Update)"


### 출력 형식:
- 주어진 Pydantic 모델 형식으로 출력합니다
- Epic은 id, title, description, priority를 포함합니다
"""
    
    prompt = ChatPromptTemplate([
        ("system", system_prompt),
        ("human", "사용자 요청: {request}\n\n위 요청을 분석하여 Epic List를 생성해주세요.")
    ])
    
    # structured output with raw response (토큰 정보 포함)
    chain = prompt | llm.with_structured_output(EpicList, include_raw=True)
    response = chain.invoke({"request": user_request})
    
    # parsed: EpicList 객체, raw: AIMessage (토큰 정보 포함)
    result = response["parsed"]
    raw_message = response["raw"]
    
    # 토큰 사용량 저장
    token_usage_list = state.get("token_usage_list", [])
    token_usage = extract_token_usage(raw_message, "Analyst Agent")
    token_usage_list.append(token_usage)
    
    print(f"✅ 생성된 Epic 목록 ({len(result.epics)}개):")
    for epic in result.epics:
        print(f"  - [{epic.id}] {epic.title} (우선순위: {epic.priority})")
    
    return {
        "epic_list": result,
        "current_epic_index": 0,
        "current_status": "planning",
        "completed_epics": [],
        "retry_count": 0,
        "max_retries": 3,
        "all_generated_files": [],
        "token_usage_list": token_usage_list
    }

def feedback_epic_list(state: MultiAgentState):
    """
    에픽 목록을 피드백 받습니다.
    """
    print("\n" + "="*80)
    print("🔍 [Feedback Epic List] 에픽 목록 피드백 시작")
    print("="*80)
    
    epic_list = state["epic_list"]
    token_usage_list = state.get("token_usage_list", [])
    user_request = state["analyzed_user_request"]

    llm = get_llm()
    system_prompt = """
당신은 소프트웨어 프로젝트의 **품질 관리자(QA Specialist)**입니다.

Analyst가 생성한 Epic List를 **검토하고 개선**하는 것이 당신의 임무입니다.

## 주요 임무

### 1. Project Setup Epic 검증 (최우선)
- 첫 번째 Epic이 "Project Setup"인지 확인
- Epic ID는 "epic-1", title은 정확히 "Project Setup"
- priority는 1 (가장 높은 우선순위)
- description에 다음 내용이 **구체적으로** 포함되어 있는지 확인:
  
  ✅ 필수 체크리스트:
  - 생성할 파일 4개 명시 (build.gradle.kts, settings.gradle.kts, application.yml, Application.java)
  - 각 파일의 정확한 경로 명시
  - build.gradle.kts에 필요한 모든 의존성 나열 (Spring Boot, JPA, H2, Lombok, Security 등)
  - application.yml 설정 항목 명시 (H2, JPA, port, logging)
  - 패키지명 규칙 명시
  
  ⚠️ 누락 시: description을 보강하여 위 내용 모두 포함시킬 것

### 2. Epic 중복 검사
- 같은 도메인이 여러 Epic으로 분리되어 있는지 확인
- 예: "User 회원가입", "User 로그인" → "User Domain (Auth)" 하나로 통합
- 중복 발견 시: Epic을 병합하고 description을 통합

### 3. Epic 누락 검사
사용자 요청을 다시 확인하여 빠진 기능이 없는지 체크:
- 명시된 기능이 Epic으로 변환되었는가?
- 암시된 기능이 포함되었는가? (예: 게시판 → CRUD, 페이징, 검색)
- 공통 기능이 필요한가? (예: 예외 처리, 공통 응답 포맷 등)

누락 발견 시: 새로운 Epic 추가

### 4. Epic 설명 품질 검사
각 Epic의 description이 다음을 포함하는지 확인:
- 구체적인 기능 목록
- 입력/출력 데이터 형식
- 필요한 엔티티 목록
- API 엔드포인트 예시 (있는 경우)
- 비즈니스 규칙 (있는 경우)

품질 부족 시: description을 구체적으로 보강 

### 5. Epic 간 의존성 확인
- 선행 Epic 없이 구현 불가능한 Epic이 있는지 확인
- 예: Comment Epic은 Post Epic 이후에 와야 함
- 의존성 순서대로 priority 재배치

## 출력 규칙

### 수정이 필요한 경우:
- 개선된 Epic List를 반환
- project_name은 원본 유지

### 수정이 불필요한 경우:
- 원본 Epic List를 그대로 반환

## 중요 원칙

1. **보수적으로 판단**: 확실하지 않으면 원본 유지
2. **Project Setup 최우선**: 이 Epic이 완벽하지 않으면 반드시 수정
3. **일관성 유지**: Epic 스타일과 형식 통일
4. **구체성 강화**: 모호한 설명은 구체적으로 개선

당신의 검토로 프로젝트의 품질이 결정됩니다. 신중하고 철저하게!
    """
    human_prompt = """
    사용자 요청: {user_request}
    주어진 에픽 목록: {epic_list}
    """

    prompt = ChatPromptTemplate([
        ("system", system_prompt),
        ("human", human_prompt)
    ])
    
    chain = prompt | llm.with_structured_output(EpicList, include_raw=True)

    response = chain.invoke({
        "user_request": user_request,
        "epic_list": epic_list
    })

    result = response["parsed"]
    raw_message = response["raw"]

    token_usage = extract_token_usage(raw_message, "Feedback Epic List")
    token_usage_list.append(token_usage)

    print(f"✅ 수정된 Epic 목록 ({len(result.epics)}개):")
    for epic in result.epics:
        print(f"  - [{epic.id}] {epic.title} (우선순위: {epic.priority})")

    return {
        "epic_list": result,
        "token_usage_list": token_usage_list
    }

# ============================================
# 3. Planner Agent (계획 에이전트)
# ============================================

def planner_agent(state: MultiAgentState) -> Dict[str, Any]:
    """
    역할: 설계자
    입력: Epic 1개
    출력: Task List (파일 목록)
    """
    print("\n" + "="*80)
    print("📋 [Planner Agent] 계획 수립 시작")
    print("="*80)
    
    epic_list = state["epic_list"]
    current_index = state["current_epic_index"]
    
    if current_index >= len(epic_list.epics):
        print("⚠️ 모든 에픽이 완료되었습니다.")
        return {"current_status": "completed"}
    
    current_epic = epic_list.epics[current_index]
    print(f"현재 Epic: [{current_epic.id}] {current_epic.title}")
    print(f"설명: {current_epic.description}\n")
    
    llm = get_llm()
    
    system_prompt = """
당신은 소프트웨어 프로젝트의 **설계자**입니다.

주어진 에픽(Epic) 1개를 받아, 이를 구현하는 데 필요한 **상세 작업 목록(Task List)**을 생성하는 것이 당신의 임무입니다.

### Task란?
- **파일 1개 = Task 1개**
- 예: User.java (Entity), UserRepository.java, UserService.java 등

### 계획 원칙:
1. 에픽을 완성하는 데 필요한 **모든 파일**을 나열합니다
2. Spring Boot 베스트 프랙티스를 따릅니다
3. 파일 간 의존성을 명확히 합니다
4. 구현 순서를 고려합니다 (Entity → Repository → DTO → Service → Controller)

### 규칙:
1. Gradle-kotlin을 사용합니다
2. 설정파일(application.yml, build.gradle.kts, *.Application.java)을 제외한다.

### 파일 구조
1. 설정 파일을 제외한 모든 파일의 기본 경로는 src/main/java/com/example/{project_name} 이다.
2. 보안, 설정, 유틸리티 파일등의 공통 파일은 common/(폴더명) 폴더에 위치한다.
    - 폴더명은 다음으로 제한됩니다. config, exception, utils
3. 도메인별 파일은 domain 폴더에 위치한다.
4. 비슷한 종류의 파일(Dto, Service 등)이 2개 이상 존재할 경우 /도메인명/분류명 폴더에 위치한다.

파일 구조 예시:
src/main/java/com/example/{project_name}/common/config/SecurityConfig.java
src/main/java/com/example/{project_name}/domain/user/User.java
src/main/java/com/example/{project_name}/domain/user/dto/UserDto.java


출력 예시:
주어진 Pydantic 모델 형식으로 출력합니다
파일 경로에서 파일명은 제외하고 경로만 출력합니다.
- id: task-1-1
- file_path: src/main/java/com/example/{project_name}/domain/user
- file_name: User.java
- description: User 엔티티 클래스
- dependencies: []
"""
    
    prompt = ChatPromptTemplate([
        ("system", system_prompt),
        ("human", """
Epic ID: {epic_id}
Epic 제목: {epic_title}
Epic 설명: {epic_description}

위 에픽을 구현하기 위한 Task List를 생성해주세요.
""")
    ])
    
    chain = prompt | llm.with_structured_output(TaskList, include_raw=True)
    response = chain.invoke({
        "project_name": state["project_name"],
        "epic_id": current_epic.id,
        "epic_title": current_epic.title,
        "epic_description": current_epic.description
    })
    
    result = response["parsed"]
    raw_message = response["raw"]
    
    token_usage_list = state.get("token_usage_list", [])
    token_usage = extract_token_usage(raw_message, f"Planner Agent (Epic: {current_epic.id})")
    token_usage_list.append(token_usage)
    
    print(f"✅ 생성된 Task 목록 ({len(result.tasks)}개):")
    for task in result.tasks:
        deps = f" (의존: {', '.join(task.dependencies)})" if task.dependencies else ""
        print(f"  - [{task.id}] {task.file_name}{deps}")
    
    return {
        "current_task_list": result,
        "current_status": "coding",
        "token_usage_list": token_usage_list
    }

# ============================================
# 4. Coder Agent (코더 에이전트)
# ============================================

def coder_agent(state: MultiAgentState) -> Dict[str, Any]:
    """
    역할: 작업자
    입력: Task List (파일 목록)
    출력: 생성된 파일들
    """
    print("\n" + "="*80)
    print("💻 [Coder Agent] 코드 생성 시작")
    print("="*80)
    
    task_list = state["current_task_list"]
    epic_list = state["epic_list"]
    current_index = state["current_epic_index"]
    current_epic = epic_list.epics[current_index]
    project_name = state["project_name"]
    
    print(f"Epic: [{current_epic.id}] {current_epic.title}")
    print(f"생성할 파일 수: {len(task_list.tasks)}\n")
    
    llm = get_llm()
    
    generated_files = []
    token_usage_list = state.get("token_usage_list", [])
    
    for i, task in enumerate(task_list.tasks, 1):
        print(f"[{i}/{len(task_list.tasks)}] 파일 생성 중: {task.file_name}")
        
        system_prompt = """
당신은 **Spring Boot 전문 개발자**입니다.

주어진 Task 설명을 바탕으로 **완전하고 실행 가능한 Java 코드**를 생성하는 것이 당신의 임무입니다.

### 코드 작성 원칙:
1. **코드만 출력**합니다 (설명이나 마크다운 문법 제외)
2. Spring Boot 베스트 프랙티스를 따릅니다
3. 필요한 import 문을 모두 포함합니다
4. Lombok 어노테이션을 적극 활용합니다
5. JPA, Spring Security 등 필요한 어노테이션을 사용합니다

### 주의사항:
- 코드 블록(```)을 사용하지 마세요
- 주석은 필요한 경우에만 간단히 작성하세요
- 패키지명은 com.example.{project_name}을 기본으로 사용하세요
"""
        
        context = ""
        if task.dependencies:
            dep_files = [f for f in generated_files if f.task_id in task.dependencies]
            if dep_files:
                context = "\n\n### 참고: 의존 파일 정보\n"
                for dep in dep_files:
                    context += f"\n// {dep.file_name}\n{dep.code_content[:500]}...\n"
        


        prompt = ChatPromptTemplate([
            ("system", system_prompt),
            ("human", """
Task ID: {task_id}
파일명: {file_name}
파일 경로: {file_path}
설명: {description}
{context}

위 정보를 바탕으로 {file_name} 파일의 완전한 Java 코드를 생성해주세요.
""")
        ])
        
        chain = prompt | llm
        sleep(5)
        
        result = chain.invoke({
            "project_name": project_name,
            "task_id": task.id,
            "file_name": task.file_name,
            "file_path": task.file_path,
            "description": task.description,
            "context": context,
        })
        
        token_usage = extract_token_usage(result, f"Coder Agent - {task.file_name}")
        token_usage_list.append(token_usage)
        
        code_content = result.content.strip()
        
        if code_content.startswith("```"):
            lines = code_content.split("\n")
            code_content = "\n".join(lines[1:-1])
        
        project_dir = Path(state["project_dir"])
        full_path = project_dir / task.file_path / task.file_name
        
        generated_file = create_file(task.id, full_path, task.file_name, code_content)
        generated_files.append(generated_file)
    
    code_result = CodeGenerationResult(
        epic_id=current_epic.id,
        generated_files=generated_files
    )
    
    success_count = len([f for f in generated_files if f.status == "success"])
    print(f"\n✅ 코드 생성 완료: {success_count}/{len(generated_files)} 성공")
    
    all_generated = state.get("all_generated_files", [])
    all_generated.extend(generated_files)
    
    return {
        "current_code_result": code_result,
        "current_status": "verifying",
        "all_generated_files": all_generated,
        "token_usage_list": token_usage_list
    }

# ============================================
# 5. Verifier Agent (검증 에이전트)
# ============================================

def verifier_agent(state: MultiAgentState) -> Dict[str, Any]:
    """
    역할: 품질 검증자
    입력: 도메인 단위 코드
    출력: 검증 결과 (SUCCESS or FAILED)
    """
    print("\n" + "="*80)
    print("🔍 [Verifier Agent] 검증 시작")
    print("="*80)
    
    code_result = state["current_code_result"]
    epic_list = state["epic_list"]
    current_index = state["current_epic_index"]
    current_epic = epic_list.epics[current_index]
    
    print(f"Epic: [{current_epic.id}] {current_epic.title}")
    print(f"검증할 파일 수: {len(code_result.generated_files)}\n")
    
    # 실제 빌드 실행 (mvn clean install 또는 gradle build)
    # 여기서는 간단히 시뮬레이션
    
    print("🔨 빌드 실행 중...")
    
    # TODO: 실제 빌드 명령어 실행
    # import subprocess
    # result = subprocess.run(
    #     ["mvn", "clean", "install"],
    #     cwd="generated",
    #     capture_output=True,
    #     text=True
    # )
    
    # 시뮬레이션: 모든 파일이 성공적으로 생성되었는지 확인
    failed_files = [f for f in code_result.generated_files if f.status == "failed"]
    
    if failed_files:
        # 실패한 경우
        verification = VerificationResult(
            epic_id=current_epic.id,
            status="FAILED",
            build_log="파일 생성 실패",
            error_files=[f.file_name for f in failed_files],
            error_message=f"{len(failed_files)}개 파일 생성 실패"
        )
        
        print(f"❌ 검증 실패: {len(failed_files)}개 파일 생성 실패")
        for f in failed_files:
            print(f"  - {f.file_name}: {f.error_message}")
        
        # 재시도 로직
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 3)
        
        if retry_count < max_retries:
            print(f"\n🔄 재시도 {retry_count + 1}/{max_retries}")
            return {
                "current_verification": verification,
                "current_status": "coding",  # 다시 코딩으로 돌아감
                "retry_count": retry_count + 1
            }
        else:
            print(f"\n⚠️ 최대 재시도 횟수 초과. 다음 에픽으로 진행합니다.")
            return {
                "current_verification": verification,
                "current_status": "planning",
                "current_epic_index": current_index + 1,
                "retry_count": 0
            }
    
    # 성공한 경우
    verification = VerificationResult(
        epic_id=current_epic.id,
        status="SUCCESS",
        build_log="빌드 성공"
    )
    
    print("✅ 검증 성공!")
    
    # 완료된 에픽 목록에 추가
    completed_epics = state.get("completed_epics", [])
    completed_epics.append(current_epic.id)
    
    # 다음 에픽으로 이동
    next_index = current_index + 1
    
    if next_index >= len(epic_list.epics):
        print("\n🎉 모든 에픽이 완료되었습니다!")
        return {
            "current_verification": verification,
            "current_status": "completed",
            "completed_epics": completed_epics,
            "final_message": f"프로젝트 완료! 총 {len(completed_epics)}개 에픽, {len(state['all_generated_files'])}개 파일 생성"
        }
    else:
        print(f"\n➡️ 다음 에픽으로 진행: {epic_list.epics[next_index].title}")
        return {
            "current_verification": verification,
            "current_status": "planning",
            "current_epic_index": next_index,
            "completed_epics": completed_epics,
            "retry_count": 0
        }

