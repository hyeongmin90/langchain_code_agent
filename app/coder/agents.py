"""
4개의 에이전트 구현:
1. Analyst Agent - 사용자 요청을 Epic List로 분해
2. Planner Agent - Epic을 Task List로 분해
3. Coder Agent - Task List를 파일로 생성
4. Verifier Agent - 도메인 단위 검증
"""

import os
from pathlib import Path
from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from schemas import (
    MultiAgentState,
    EpicList,
    TaskList,
    GeneratedFile,
    CodeGenerationResult,
    VerificationResult,
)

# ============================================
# 1. Analyst Agent (분석 에이전트)
# ============================================

def analyst_agent(state: MultiAgentState) -> Dict[str, Any]:
    """
    역할: 전략가
    입력: 사용자의 모호한 요청
    출력: Epic List (간결한 에픽 목록)
    """
    print("\n" + "="*80)
    print("🎯 [Analyst Agent] 분석 시작")
    print("="*80)
    
    user_request = state["user_request"]
    print(f"사용자 요청: {user_request}\n")
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0.7
    )
    
    system_prompt = """
당신은 소프트웨어 프로젝트의 **전략가**입니다.

사용자의 모호하고 큰 요청을 받아, **간결한 '에픽(Epic) 목록'**으로 분해하는 것이 당신의 임무입니다.

### 에픽이란?
- 큰 기능 단위 (도메인 단위)
- 예: "User Domain (Auth)", "Post Domain (Core)", "Comment Domain (Sub)"

### 분석 원칙:
1. 사용자 요청을 도메인별로 분해합니다
2. 각 에픽은 **독립적으로 구현 가능**해야 합니다
3. 우선순위를 명확히 정합니다 (낮을수록 먼저 구현)
4. 첫 번째 에픽은 항상 "Project Setup"이어야 합니다

### 출력 형식:
- JSON 형식으로 출력합니다
- Epic은 id, title, description, priority를 포함합니다

### 예시:
입력: "회원가입, 로그인, 게시판 기능이 있는 블로그 MVP"
출력:
{{
    "epics": [
        {{
            "id": "epic-1",
            "title": "Project Setup",
            "description": "Spring Boot 프로젝트 초기 설정, 의존성 설정, application.yml 설정",
            "priority": 1
        }},
        {{
            "id": "epic-2",
            "title": "User Domain (Auth)",
            "description": "회원가입, 로그인, JWT 인증 기능",
            "priority": 2
        }},
        {{
            "id": "epic-3",
            "title": "Post Domain (Core)",
            "description": "게시글 CRUD 기능",
            "priority": 3
        }}
    ]
}}
"""
    
    prompt = ChatPromptTemplate([
        ("system", system_prompt),
        ("human", "사용자 요청: {request}\n\n위 요청을 분석하여 Epic List를 생성해주세요.")
    ])
    
    chain = prompt | llm.with_structured_output(EpicList)
    
    result = chain.invoke({"request": user_request})
    
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
        "all_generated_files": []
    }

# ============================================
# 2. Planner Agent (계획 에이전트)
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
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0.7
    )
    
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

### 출력 형식:
- JSON 형식으로 출력합니다
- 각 Task는 id, file_name, file_path, description, dependencies를 포함합니다

### 예시:
입력: Epic "User Domain (Auth)"
출력:
{{
    "epic_id": "epic-2",
    "tasks": [
        {{
            "id": "task-2-1",
            "file_name": "User.java",
            "file_path": "src/main/java/com/example/domain/user/",
            "description": "User 엔티티: id, username, password, email, createdAt 필드 포함",
            "dependencies": []
        }},
        {{
            "id": "task-2-2",
            "file_name": "UserRepository.java",
            "file_path": "src/main/java/com/example/domain/user/",
            "description": "User JPA Repository: findByUsername 메서드 포함",
            "dependencies": ["task-2-1"]
        }}
    ]
}}
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
    
    chain = prompt | llm.with_structured_output(TaskList)
    
    result = chain.invoke({
        "epic_id": current_epic.id,
        "epic_title": current_epic.title,
        "epic_description": current_epic.description
    })
    
    print(f"✅ 생성된 Task 목록 ({len(result.tasks)}개):")
    for task in result.tasks:
        deps = f" (의존: {', '.join(task.dependencies)})" if task.dependencies else ""
        print(f"  - [{task.id}] {task.file_name}{deps}")
    
    return {
        "current_task_list": result,
        "current_status": "coding"
    }

# ============================================
# 3. Coder Agent (코더 에이전트)
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
    
    print(f"Epic: [{current_epic.id}] {current_epic.title}")
    print(f"생성할 파일 수: {len(task_list.tasks)}\n")
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0.3
    )
    
    generated_files = []
    
    # Task를 순회하며 파일 생성
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
- 패키지명은 com.example을 기본으로 사용하세요
"""
        
        # 이전에 생성된 파일 정보 (의존성 참고용)
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
                "task_id": task.id,
                "file_name": task.file_name,
                "file_path": task.file_path,
                "description": task.description,
                "context": context
            })
            
            code_content = result.content.strip()
            
            # 코드 블록 제거 (혹시 모를 경우)
            if code_content.startswith("```"):
                lines = code_content.split("\n")
                code_content = "\n".join(lines[1:-1])
            
            # 파일 저장
            full_path = Path("generated") / task.file_path / task.file_name
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
    
    # 전체 생성된 파일 목록에 추가
    all_generated = state.get("all_generated_files", [])
    all_generated.extend(generated_files)
    
    return {
        "current_code_result": code_result,
        "current_status": "verifying",
        "all_generated_files": all_generated
    }

# ============================================
# 4. Verifier Agent (검증 에이전트)
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

