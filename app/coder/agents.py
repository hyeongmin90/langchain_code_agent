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
from pathlib import Path
from typing import Dict, Any, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from .schemas import (
    MultiAgentState,
    EpicList,
    TaskList,
    GeneratedFile,
    CodeGenerationResult,
    VerificationResult,
    TokenUsage,
)


def get_llm(model: str = "gemini-2.5-pro"):
    return ChatGoogleGenerativeAI(model=model)


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

4. **주요 엔티티와 관계를 정의**하세요
   - 핵심 엔티티의 속성과 관계(1:N, N:M)를 명시

### 기술 환경:
- Spring Boot, Gradle-Kotlin, H2 Database, JPA

### 출력 형식:
자유롭게 작성하되, 다음 항목을 포함하세요:
- 프로젝트 개요 및 주요 기능
- 도메인 모델 (엔티티 및 관계)
- 기능 요구사항 (도메인별)
- 필요시 API 엔드포인트 및 공통 기능

**중요**: 프로젝트 규모와 복잡도에 맞게 적절히 판단하여 작성하세요.
"""
    prompt = ChatPromptTemplate([
        ("system", system_prompt),
        ("human", "사용자 요청: {request}")
    ])
    chain = prompt | llm
    result = chain.invoke({"request": user_request})

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

def setup_project(state: MultiAgentState) -> Dict[str, Any]:
    """
    프로젝트 설정
    """
    print("\n" + "="*80)
    print("🔧 [Setup Project] 프로젝트 설정 시작")
    print("="*80)

    
    project_uuid = str(uuid.uuid4())
    zip_src = Path(__file__).parent.parent / "springTemplate" / "demo.zip"    
    project_dir = Path(__file__).parent.parent.parent
    dest_dir = project_dir / "generated" / project_uuid
    dest_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_src, "r") as zip_ref:
        zip_ref.extractall(dest_dir)

    return {
        "project_uuid": project_uuid,
        "project_dir": str(dest_dir),
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
또한 프로젝트의 이름을 정하는 것도 당신의 임무입니다. 프로젝트 이름은 영어로 정합니다.

### 에픽이란?
- 큰 기능 단위 (도메인 단위)
- 예: "User Domain (Auth)", "Post Domain (Core)", "Comment Domain (Sub)"

### 분석 원칙:
1. 사용자 요청을 도메인별로 분해합니다
2. 각 에픽은 **독립적으로 구현 가능**해야 합니다
3. 우선순위를 명확히 정합니다 (낮을수록 먼저 구현)
4. 첫 번째 에픽은 항상 "Project Setup"이어야 하며, 설명에는 다음이 포함되어야 합니다.
- Project Setup은 build.gradle.kts, settings.gradle.kts, application.yml, *.Application.java 파일만 생성합니다.
- 필요한 의존성을 모두 적어야하며, 누락되서는 안된다.
- build.gradle.kts, settings.gradle.kts 파일은 Root 경로에 위치합니다.

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
5. 에픽간의 중복이 존재해서는 안됩니다.

### 규칙:
1. DB는 H2 Database를 사용합니다
2. Gradle-kotlin을 사용합니다
3. 파일경로를 지정하지 않으면 프로젝트 root 경로에 위치합니다.
4. epic명이 Project Setup이 아닐 경우엔 설정파일(application.yml, build.gradle.kts, *.Application.java)을 제외한다.

### 파일 구조
설정 파일을 제외한 모든 파일은 src/main/java/com/example/{project_name} 폴더에 위치합니다.
보안, 설정, 유틸리티 파일등의 공통 파일은 common/(폴더명) 폴더에 위치합니다.
    - 폴더명은 다음으로 제한됩니다. config, exception, utils
도메인별 파일은 domain 폴더에 위치합니다.
Dto 파일의 경우 domain/(도메인명)/dto 폴더에 위치합니다.

ex) src/main/java/com/example/{project_name}/common/config/SecurityConfig.java
ex) src/main/java/com/example/{project_name}/domain/user/User.java
ex) src/main/java/com/example/{project_name}/domain/user/dto/UserDto.java
ex) build.gradle.kts

### 출력 형식:
- 주어진 Pydantic 모델 형식으로 출력합니다
- 각 Task는 id, file_name, file_path, description, dependencies를 포함합니다
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
    
    # structured output with raw response (토큰 정보 포함)
    chain = prompt | llm.with_structured_output(TaskList, include_raw=True)
    response = chain.invoke({
        "project_name": epic_list.project_name,
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
    project_name = epic_list.project_name
    
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
        
        try:
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
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(code_content)
            
            generated_file = GeneratedFile(
                task_id=task.id,
                file_name=task.file_name,
                file_path=str(full_path),
                code_content=code_content,
                status="success"
            )
            
            generated_files.append(generated_file)
            print(f"  ✅ 성공: {full_path}")
            
        except Exception as e:
            print(f"  ❌ 실패: {task.file_name} - {str(e)}")
            generated_file = GeneratedFile(
                task_id=task.id,
                file_name=task.file_name,
                file_path=task.file_path,
                code_content="",
                status="failed",
                error_message=str(e)
            )
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

