"""
Agentic Coder의 에이전트 구현

3개의 에이전트:
1. Specification Writer Agent - 요청사항 → 명세서 작성 (API 시그니처 추출 및 주입)
2. Code Generator Agent - 명세서 → 코드 작성
3. Static Reviewer Agent - 작성된 코드 → 정적 리뷰
"""

import os
import asyncio
import uuid
import zipfile
from pathlib import Path
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from .schemas import (
    AgenticCoderState,
    TokenUsage,
    RequirementAnalysisResult,
    SkeletonCodeList,
    BuildGradleKts,
)

def create_file(file_path: Path, content: str):
    """
    파일을 생성합니다.
    
    Args:
        file_path: 전체 파일 경로 (Path 객체)
        content: 파일 내용
    
    Returns:
        None
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if '\\n' in content or '\\t' in content:
        content = content.encode().decode('unicode_escape')
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

def get_llm(model: str = "gpt-5-mini", is_openai: bool = True):
    """LLM 인스턴스 생성"""
    if is_openai:
        return ChatOpenAI(model=model)
    else:
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
# 1. requirement_analyst_agent
# ============================================

def requirement_analyst_agent(state: AgenticCoderState) -> Dict[str, Any]:
    """
    역할: 요구사항 분석자
    입력: 사용자의 요구사항
    출력: 요구사항 분석 결과
    
    주요 작업:
    - 사용자의 요구사항을 분석하여 요구사항 분석 결과를 생성
    """
    print("\n" + "="*80)
    print("📝 [Requirement Analyst Agent] 요구사항 분석 시작")
    print("="*80)
    
    orchestrator_request = state["orchestrator_request"]
    print(f"에이전트 요청: {orchestrator_request}")
    
    llm = get_llm()
    
    system_prompt = """
당신은 **요구사항 분석 전문가**입니다.
사용자의 간단한 요청을 받아 프로젝트 이름, 요구사항 분석 결과를 생성하는 것이 당신의 임무입니다.

### 프로젝트 이름 작성 규칙
- 길고 복잡한 이름 대신, 간단명료하게 단어 위주로 작성합니다.
- "SimpleTodolist" 처럼 형용사 등을 붙이기보단, 핵심 도메인만 사용 (예: Todolist, Blog 등).
- 불필요한 접두어/접미어/형용사는 제외합니다.
- 영어 단어만 사용하고, 공백 없이 카멜표기법으로 작성합니다.
- 너무 축약/생략하지 마시고, 사용자의 요구가 드러나는 명사를 충실하게 씁니다.

### 요구사항 분석
- 명시된 기능과 암시된 기능을 모두 파악
- 필요한 도메인 모델 식별
- 비즈니스 규칙 정의
- 자바 백엔드 기준으로 분석

## 출력 형식
- 추가적인 설명이나 의견은 작성하지 말고, 분석된 요구사항만 작성하라.

## 중요 원칙
1. **구체성**: 모호한 표현 금지, 구현 가능한 수준으로 작성
2. **완전성**: 모든 필요한 요구사항을 빠짐없이 포함
"""
    
    prompt = ChatPromptTemplate([
        ("system", system_prompt),
        ("human", "사용자 요청: {user_request}\n\n위 요청을 분석하여 요구사항 분석 결과를 생성해주세요.")
    ])
    
    chain = prompt | llm.with_structured_output(RequirementAnalysisResult, include_raw=True)
    
    response = chain.invoke({"user_request": orchestrator_request})

    result = response["parsed"]
    raw_message = response["raw"]
    
    token_usage_list = state.get("token_usage_list", [])
    token_usage = extract_token_usage(raw_message, "Requirement Analyst Agent")
    token_usage_list.append(token_usage)

    return {
        "requirement_analysis_result": result.model_dump(),
        "token_usage_list": token_usage_list
    }



# ============================================
# 1. Setup Project (프로젝트 설정)
# ============================================

def setup_project(state: AgenticCoderState) -> Dict[str, Any]:
    """
    프로젝트 설정 및 초기 파일 생성
    - 프로젝트 디렉토리 생성
    - 프로젝트 이름 및 의존성 결정
    - 초기 설정 파일 생성 (build.gradle.kts, settings.gradle.kts, application.yml)
    """
    print("\n" + "="*80)
    print("🔧 [Setup Project] 프로젝트 설정 및 초기 파일 생성 시작")
    print("="*80)
    
    
    llm = get_llm()
    
    system_prompt = """
    당신은 소프트웨어 프로젝트의 **프로젝트 설정 전문가**입니다.
    사용자의 요청을 바탕으로 필요한 설정 파일을 생성해야 합니다.

    ### 파일 1: build.gradle.kts
    - gradle-kotlin
    - Spring Boot 3.x 버전 사용
    - Java 17 사용
    - 기본 의존성: Spring Web, Spring Data JPA, H2 Database, Lombok
    - 요구사항을 분석하여 필요한 의존성을 모두 포함

    ### 파일 2: application.yml
    - 서버 포트 8080
    - 만약 db가 필요하다면 H2 데이터베이스, 콘솔 활성화, 인메모리 DB를 사용하라.
    - JPA 설정 (hibernate ddl-auto: create-drop, show-sql: true)
    - 환경 변수 설정
    - 요구사항을 분석하여 필요한 설정을 모두 포함

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
    chain = prompt | llm.with_structured_output(BuildGradleKts, include_raw=True) 

    response = chain.invoke({
        "user_request": state["requirement_analysis_result"]
    })

    project_setup = response["parsed"]
    raw_message = response["raw"]

    token_usage_list = state.get("token_usage_list", [])
    token_usage = extract_token_usage(raw_message, "Setup Project - 기본 설정")
    token_usage_list.append(token_usage)
    
    project_name = state.get("requirement_analysis_result", {}).get("project_name", "")
    lower_project_name = project_name.lower()
    
    settings_gradle_content = f'rootProject.name = "{lower_project_name}"\n'
    
    setup_files = {
        "build.gradle.kts": {
            "file_name": "build.gradle.kts",
            "file_path": "build.gradle.kts",  # 프로젝트 루트
            "code_content": project_setup.build_gradle_kts
        },
        "settings.gradle.kts": {
            "file_name": "settings.gradle.kts",
            "file_path": "settings.gradle.kts",  # 프로젝트 루트
            "code_content": settings_gradle_content
        },
        "src/main/resources/application.yml": {
            "file_name": "application.yml",
            "file_path": "src/main/resources/application.yml",  # 상대 경로
            "code_content": project_setup.application_yml
        }
    }
    
    return {
        "generated_files": setup_files,
        "token_usage_list": token_usage_list
    }

# ============================================
# 2. Skeleton Code Generator Agent
# ============================================

def skeleton_code_generator_agent(state: AgenticCoderState) -> Dict[str, Any]:
    """
    역할: 스켈레톤 코드 생성자
    입력: 파일 계획
    출력: 스켈레톤 코드
    """
    print("\n" + "="*80)
    print("💻 [Skeleton Code Generator Agent] 스켈레톤 코드 생성")
    print("="*80)
    
    llm = get_llm("gemini-2.5-pro", is_openai=False)
    
    system_prompt = """
    당신은 **Spring Boot 전문 개발자**입니다.
    주어진 요구사항을 바탕으로 **스켈레톤 코드**를 생성하는 것이 당신의 임무입니다.
    
    ### 주의사항
    - 자바 스프링 부트 3.x 기반으로 스켈레톤 코드를 생성하라.
    - 자바 17 사용
    - 주어진 기술 스택을 준수하라.

    ### 핵심 작업

    ### 1. 요구사항 분석
    - 요구사항을 분석하여 스켈레톤 코드를 생성해야 하는 파일을 결정

    ### 2. 스켈레톤 코드 생성
    - 결정된 파일을 바탕으로 스켈레톤 코드를 생성
    - 기능(메서드) 구현 대신 주석으로 필요한 기능을 명시
        - 주석에는 필요한 정보를 자세히 명시하여 다른 파일을 보지 않아도 코드 구현이 가능하도록 하라.
        - 의존성있는 타입, 메서드, 변수명, 클래스명 등을 명시하라.
    - 또한 문제 없이 컴파일 되는 코드를 생성하라.
    - Entity, DTO, Repository의 경우 완성된 코드를 생성하라.
    - .java 확장자 파일만 생성하라.
    - import, 어노테이션, 메서드, 의존성 주입만 구현된 스켈레톤 코드를 생성하라.


    ### 3. 패키지 구조 설계
    - 프로젝트의 규모에 따라 적절한 패키지 구조를 설계하라.
    - 프로젝트의 이름을 정하고 패키지 구조를 설계하라. 기본 패키지 구조는 com.example.(프로젝트 이름) 이다.
    - 파일 경로는 파일 이름까지 포함된 전체 경로를 출력하라.

    ## 출력 형식
    - SkeletonCode 모델로 출력
    - file_name, file_path, code_content, need_to_generate
    - import, 어노테이션, 메서드, 의존성 주입만 구현된 스켈레톤 코드를 생성하라.
    - 프로젝트의 실행점인 *.Application.java 파일을 가장 처음에 생성하라.
    - need_to_generate 설정:
      - Entity, DTO, Repository, Exception(Custom Exception 제외), Application.java: False (이미 완전한 코드로 생성되므로 추가 구현 불필요)
      - Service, Controller, Security, Handler, Filter 등 비즈니스 로직이 필요한 파일: True (메서드 구현 필요)
    
    ## 중요 원칙
    - 일관성: 패키지 구조, 파일 명, 클래스 명, 메서드 명, 변수 명 등 일관성 유지
    """

    prompt = ChatPromptTemplate([
        ("system", system_prompt),
        ("human", "요구사항: {user_request}\n\n application.yml 파일 내용: {application_yml}\n\n위 요구사항을 바탕으로 스켈레톤 코드를 생성해주세요.")
    ])
    
    chain = prompt | llm.with_structured_output(SkeletonCodeList, include_raw=True)

    requirement_analysis_result = state.get("requirement_analysis_result", {})

    generated_files = state.get("generated_files", {})
    application_yml = generated_files.get("src/main/resources/application.yml", {}).get("code_content", "")

    response = chain.invoke({
        "user_request": requirement_analysis_result,
        "application_yml": application_yml
    })
    
    result = response["parsed"]
    raw_message = response["raw"]
    
    all_skeleton_list = result.skeleton_code_list

    result_list = [sc.file_path for sc in all_skeleton_list]
    result_list = "\n".join(result_list)

    completed_skeleton_list = [sc for sc in all_skeleton_list if sc.need_to_generate is False]
    skeleton_code_list = [sc for sc in all_skeleton_list if sc.need_to_generate is True]
    
    completed_code = {
        sc.file_path: {
            "file_name": sc.file_name,
            "file_path": sc.file_path,
            "code_content": sc.skeleton_code
        }
        for sc in completed_skeleton_list
    }

    generated_files = state.get("generated_files", {})
    generated_files = {**generated_files, **completed_code}

    # need_to_generate가 True인 것들만 skeleton에 포함
    skeleton = {
        sc.file_path: {
            "file_name": sc.file_name,
            "file_path": sc.file_path,
            "skeleton_code": sc.skeleton_code
        }
        for sc in skeleton_code_list
    }

    all_skeleton = {
        sc.file_path: {
            "file_name": sc.file_name,
            "file_path": sc.file_path,
            "skeleton_code": sc.skeleton_code
        }
        for sc in all_skeleton_list
    }

    print(f"✅ 스켈레톤 코드 생성 완료")
    print(result_list)
    print("="*80)
    
    token_usage_list = state.get("token_usage_list", [])
    token_usage = extract_token_usage(raw_message, "Skeleton Code Generator Agent")
    token_usage_list.append(token_usage)
    
    return {
        "generated_files": generated_files,
        "skeleton_code_list": skeleton,
        "token_usage_list": token_usage_list,
        "all_skeleton": all_skeleton
    }

# ============================================
# 3. Code File Generator Agent
# ============================================

async def _generate_single_file_async(file_info: dict, semaphore: asyncio.Semaphore) -> dict:
    async with semaphore:  
        print(f"🔄 생성 시작: {file_info['file_name']}")
        
        llm = get_llm()
        
        system_prompt = """
            당신은 **Spring Boot 전문 개발자**입니다.

            주어진 스켈레톤 코드를 바탕으로 **완전한 코드**를 생성하는 것이 당신의 임무입니다.

            ## 핵심 작업

            ### 1. 하나의 파일만 생성
            - 주어진 스켈레톤 코드를 바탕으로 하나의 파일만 생성
            - 주어진 스켈레톤 코드의 파일 명, 경로, 설명을 준수하라.
            - 주어진 스켈레톤 코드의 기능(메서드)을 구현하라.

            ### 2. 코드 품질
            - Spring Boot 베스트 프랙티스
            - 적절한 예외 처리
            - Validation 어노테이션
            - 깔끔하고 자명한 코드 구현

            ## 출력 형식
            - 완전한 Java 코드 (import부터 끝까지)
            - 주석은 필요한 경우에만 간단히 작성하세요
            - 코드 블록(```)을 사용하지 마세요
            - 추가적인 설명이나 의견은 작성하지 말고, 코드만 작성하라.

            ## 중요 원칙
            1. **완전성**: 주어진 스켈레톤 코드의 기능(메서드)을 모두 구현.
            2. **일관성**: 주어진 스켈레톤 코드의 파일 명, 메서드, 변수명, 경로 일치.
            3. **실행 가능성**: 컴파일 가능한 코드 구현.
        """
            
        prompt = ChatPromptTemplate([
            ("system", system_prompt),
            ("human", """
                다음 스켈레톤 코드를 바탕으로 하나의 파일을 생성하라.

                생성할 파일:
                - 파일명: {file_name}
                - 경로: {file_path}
                - 스켈레톤 코드
                {skeleton_code}
                """)
        ])
        
        chain = prompt | llm
        
        # 동기 LLM 호출을 비동기 환경에서 실행
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: chain.invoke({
                "file_name": file_info["file_name"],
                "file_path": file_info["file_path"],
                "skeleton_code": file_info["skeleton_code"]
            })
        )
        
        code_content = response.content
        code_content = code_content.strip()
        code_content = code_content.replace("```java", "").replace("```", "")
        
        print(f"✅ 생성 완료: {file_info['file_name']} ({len(code_content)} 자)")
        
        return {
            "file": {
                file_info["file_path"]: {
                    "file_name": file_info["file_name"],
                    "file_path": file_info["file_path"],
                    "code_content": code_content,
                },
            },
            "token_usage": extract_token_usage(response, f"Code Generator - {file_info['file_name']}")
        }

def code_file_generator_agent(state: AgenticCoderState) -> Dict[str, Any]:
    """
    역할: 다중 파일 코드 생성자 
    입력: 생성할 파일 목록 (files_to_generate)
    출력: 생성된 파일들
    """
    
    print("\n" + "="*80)
    print("💻 [Code File Generator Agent] 다중 파일 생성")
    print("="*80)

    skeleton = state.get("skeleton_code_list", {})
    file_tasks = []
    for _, content in skeleton.items():
        file_tasks.append({
            "file_name": content["file_name"],
            "file_path": content["file_path"],
            "skeleton_code": content["skeleton_code"]
        })

    async def _run_parallel_generation():
        semaphore = asyncio.Semaphore(10)
        tasks = [
            _generate_single_file_async(task, semaphore) for task in file_tasks
        ]
        return await asyncio.gather(*tasks)
    
    # 이벤트 루프 실행
    results = asyncio.run(_run_parallel_generation())
    
    # 결과 수집
    new_files = {
        file_info["file_path"]: file_info
        for r in results
        for file_info in r["file"].values()
    }

    generated_files = state.get("generated_files", {})  # setup_project에서 생성한 설정 파일들
    merged_files = {**generated_files, **new_files}
    
    token_usages = [r["token_usage"] for r in results]
    
    token_usage_list = state.get("token_usage_list", [])
    token_usage_list.extend(token_usages)
    
    print(f"\n✅ 전체 파일 생성 완료: {len(merged_files)}개")
    for _, f in merged_files.items():
        print(f"   - {f['file_name']} : {f['file_path']}")
    
    return {
        "generated_files": merged_files,
        "token_usage_list": token_usage_list
    }

# ============================================
# 4. File Writer Node (파일 생성 노드)
# ============================================

def file_writer_node(state: AgenticCoderState) -> Dict[str, Any]:
    """
    역할: 생성된 코드를 파일 시스템에 저장
    입력: generated_files (메모리상의 코드)
    출력: 파일 시스템에 저장된 파일들
    
    주요 작업:
    - generated_files의 모든 파일을 실제 파일 시스템에 생성
    """
    print("\n" + "="*80)
    print("📁 [File Writer Agent] 파일 시스템에 파일 생성")
    print("="*80)

    # 프로젝트 디렉토리 생성
    project_uuid = str(uuid.uuid4())
    zip_src = Path(__file__).parent.parent / "springTemplate" / "demo.zip"    
    project_dir = Path(__file__).parent.parent.parent
    dest_dir = project_dir / "generated" / project_uuid
    dest_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_src, "r") as zip_ref:
        zip_ref.extractall(dest_dir)
    
    generated_files = state.get("generated_files", {})

    if not generated_files:
        print("⚠️ 생성할 파일이 없습니다.")
        return {}
    
    project_dir_path = dest_dir
    print(f"📂 프로젝트 디렉토리: {project_dir_path}\n")
    
    created_count = 0
    for _, file_info in generated_files.items():
        if file_info["file_path"]:
            full_file_path = project_dir_path / file_info["file_path"]
        
            create_file(full_file_path, file_info["code_content"])
        
        # 출력용 경로 표시
        display_path = file_info["file_path"] if file_info["file_path"] else "."
        print(f"   ✓ 생성 완료: {display_path}/{file_info['file_name']}")
        created_count += 1
    
    print(f"\n✅ 총 {created_count}개 파일 생성 완료!\n")

    all_skeleton = state.get("all_skeleton", {})
    for _, content in all_skeleton.items():
        if content["file_path"]:
            full_file_path = project_dir_path / "skeleton" / content["file_path"]
            create_file(full_file_path, content["skeleton_code"])
        print(f"   - {content['file_path']}")
    
    return {}