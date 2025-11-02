"""
Agentic Coder의 에이전트 구현

3개의 에이전트:
1. Specification Writer Agent - 요청사항 → 명세서 작성 (API 시그니처 추출 및 주입)
2. Code Generator Agent - 명세서 → 코드 작성
3. Static Reviewer Agent - 작성된 코드 → 정적 리뷰
"""

import os
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from .schemas import (
    AgenticCoderState,
    Specification,
    SingleFileGeneration,
    StaticReviewResult,
    CodeIssue,
    OrchestratorDecision,
    TokenUsage
)


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
# 오케스트라 에이전트 (Orchestrator Agent)
# ============================================

def orchestrator_agent(state: AgenticCoderState) -> Dict[str, Any]:
    """
    에이전틱 오케스트라 에이전트: LLM을 사용하여 워크플로우를 지능적으로 조율
    
    역할:
    - 현재 상태를 분석하고 평가
    - 각 에이전트의 결과를 검토
    - 다음 단계를 동적으로 결정
    - 재시도 필요 여부 판단
    - 워크플로우 완료 조건 평가
    
    특징:
    - 단순 조건문이 아닌 LLM 기반 의사결정
    - 상황에 맞는 적응적 판단
    - 명확한 이유와 함께 결정 제시
    """
    print("\n" + "="*80)
    print("🎯 [Orchestrator Agent] 워크플로우 분석 및 다음 단계 결정")
    print("="*80)
    
    llm = ChatOpenAI(model="gpt-5-mini")
    
    # 현재 상태 분석을 위한 컨텍스트 구성
    current_status = state.get("current_status", "spec")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)
        

    # 파일 생성 상태
    files_plan = state.get("files_plan", [])
    generated_files = state.get("generated_files", [])
    current_file_index = state.get("current_file_index", 0)
    
    if files_plan:
        code_status = f"진행 중 ({len(generated_files)}/{len(files_plan)} 파일)"
        if len(generated_files) == len(files_plan):
            code_status = "✅ 완료"
    else:
        code_status = "❌ 계획 전"
    
    review_status = "✅ 통과" if state.get("review_passed") else ("❌ 실패" if state.get("review_result") else "⏳ 대기중")
    
    print(f"📊 현재 진행 상태:")
    print(f"  - 파일 계획: {'✅ 완료' if files_plan else '❌ 미완료'} ({len(files_plan) if files_plan else 0}개 파일)")
    print(f"  - 코드 생성: {code_status}")
    if generated_files:
        print(f"    최근 생성 파일:")
        for gf in generated_files[-2:]:
            print(f"      - {gf['file_name']}")
    print(f"  - 정적 리뷰: {review_status}")
    print(f"  - 현재 상태: {current_status}\n")
    
    # 상황 정보 수집
    context_info = f"""
현재 워크플로우 상태:
- 단계: {current_status}
- 검증 횟수: {retry_count}/{max_retries}

이전 단계 결과:
{state.get("pre_result", "없음")}

이전 단계 제안사항:
{state.get("previous_suggestions", "없음")}

각 단계 완료 상태:
- 파일 계획: {'완료' if files_plan else '미완료'} (총 {len(files_plan) if files_plan else 0}개 파일)
- 코드 생성: {code_status}
  - 생성 완료: {len(generated_files) if generated_files else 0}개
  - 남은 파일: {len(files_plan) - len(generated_files) if files_plan and generated_files else 0}개
- 정적 리뷰: {review_status}
"""
    # 파일 계획 상세 정보 추가
    if files_plan:
        context_info += f"\n파일 계획 상세 ({len(files_plan)}개):\n"
        context_info += f"{files_plan}"
    
    # 최근 생성된 파일 정보
    if generated_files:
        context_info += f"\n모든 생성된 파일:\n"
        for gf in generated_files:
            context_info += f"- {gf['file_name']}\n"
    
    # 리뷰 결과가 있다면 추가
    if state.get("review_result"):
        import json
        review_data = json.loads(state["review_result"])
        issues_summary = f"발견된 이슈: {len(review_data['issues'])}개"
        if review_data['issues']:
            critical_count = sum(1 for issue in review_data['issues'] if issue['severity'] == 'CRITICAL')
            major_count = sum(1 for issue in review_data['issues'] if issue['severity'] == 'MAJOR')
            issues_summary += f" (CRITICAL: {critical_count}, MAJOR: {major_count})"
        
        context_info += f"\n정적 리뷰 결과:\n- {issues_summary}\n- 요약: {review_data['summary']}\n"
    
    system_prompt = """
당신은 **코드 생성 워크플로우의 총괄 매니저이자 파일 계획자**입니다.
당신의 임무는 현재 상황을 분석하고, **다음 행동과 생성할 파일을 지능적으로 결정**하는 것입니다.

당신은 최종적으로 사용자의 요구사항에 맞는 프로젝트를 완성하는 것이 목적입니다.

## 워크플로우 단계
1. **specification_writer**: 명세서 작성 (files_plan 수립)
2. **code_generator**: 코드 생성 (파일 하나씩)
3. **static_reviewer**: 정적 리뷰
4. **completed**: 모든 단계 완료

## 의사결정 원칙

### 1.파일 계획 수립
files_plan이 없다면:
- next_action: "specification_writer"

### 2. 파일 생성 진행 중
files_plan이 있고 아직 모든 파일이 생성되지 않았다면:
- **다음 파일 결정 (next_file)**
  - files_plan에 있는 파일 중 우선순위가 가장 높은 파일을 next_file로 설정
  - 의존하는 파일이 모두 생성되었는지 확인
- **의존성 확인 (dependent_files)**
  - 의존성이 있다면 의존하는 파일 계획을 dependent_files에 추가.
- next_action: "code_generator"

### 3. 모든 파일 생성 완료
모든 파일이 생성되었다면:
- next_action: "static_reviewer"

### 4. 리뷰 결과 분석 및 최종 결정
리뷰가 완료되었다면:
- **통과 (passed=True)**: 
  - next_action: "completed"
  - final_message 생성
  
- **실패 (passed=False)**:
  - 이슈 심각도 분석:
    - CRITICAL 이슈 있음 → 재시도 필요
    - MAJOR 이슈 많음 (3개 이상) → 재시도 고려
    - MINOR만 있음 → completed 가능

- 검증 횟수 도달시 실패 처리:
  - next_action: "failed"
  - final_message 생성

## 출력 형식
- OrchestratorDecision 모델로 출력
- **next_file**: code_generator로 갈 때마다 다음 파일 (FilePlan 하나)
- **dependent_files**: next_file이 의존하는 파일들 (선택사항)
- next_action, reasoning, suggestions

## 중요 원칙
1. **계획은 한번**: specification_writer 는 한번만 수행
2. **한번에 하나**: next_file은 매번 한 파일씩만 지정
3. **의존성 고려**: 의존하는 파일이 먼저 생성되도록
4. **명확한 순서**: Entity → Repository → DTO → Service → Controller
"""
    
    prompt = ChatPromptTemplate([
        ("system", system_prompt),
        ("human", """
현재 상황:
{context}

위 상황을 분석하여 다음 행동을 결정해주세요.
""")
    ])
    
    chain = prompt | llm.with_structured_output(OrchestratorDecision, include_raw=True)
    
    response = chain.invoke({"context": context_info})
    
    decision = response["parsed"]
    raw_message = response["raw"]
    
    print(f"✅ 오케스트라 결정 완료")
    print(f"  - 다음 행동: {decision.next_action}")
    print(f"\n📝 결정 이유:")
    print(f"  {decision.reasoning}\n")
    
    # 다음 파일이 있으면 출력
    if decision.next_file:
        print(f"🎯 다음 생성 파일: {decision.next_file.file_name}")
        print(f"   경로: {decision.next_file.file_path}")
        print(f"   설명: {decision.next_file.description}\n")
    
    if decision.suggestions:
        print(f"💡 제안사항:")
        print(f"  {decision.suggestions}")
        print()
    
    token_usage_list = state.get("token_usage_list", [])
    token_usage = extract_token_usage(raw_message, "Orchestrator Agent")
    token_usage_list.append(token_usage)

    # State 업데이트
    result = {
        "current_status": decision.next_action,
        "orchestrator_reasoning": decision.reasoning,
        "token_usage_list": token_usage_list,
        "previous_suggestions": decision.suggestions
    }
    
    # next_file이 있으면 저장
    if decision.next_file:
        result["next_file_to_generate"] = decision.next_file.model_dump()
    
    # final_message가 있으면 저장 (completed일 때)
    if decision.final_message:
        result["final_message"] = decision.final_message
        print(f"\n📢 최종 메시지: {decision.final_message}")
    
    return result


# ============================================
# 1. Specification Writer Agent
# ============================================

def specification_writer_agent(state: AgenticCoderState) -> Dict[str, Any]:
    """
    역할: 명세서 작성자
    입력: 사용자 요청
    출력: 파일과 시그니처 목록
    
    주요 작업:
    - 사용자 요청 분석
    - 기능 요구사항 도출
    - API 시그니처 추출
    - 기술 스택 결정
    - 아키텍처 설계
    """
    print("\n" + "="*80)
    print("📝 [Specification Writer Agent] 명세서 작성 시작")
    print("="*80)
    
    user_request = state["user_request"]
    print(f"사용자 요청: {user_request}\n")
    
    llm = get_llm("gemini-2.5-pro", is_openai=False)
    
    system_prompt = """
당신은 **API 시그니처 작성 전문가**입니다.
사용자의 간단한 요청을 받아 **API 시그니처**를 작성하는 것이 당신의 임무입니다.
프로젝트 생성에 필요한 모든 파일과 시그니처를 작성하라. 

## 핵심 작업

### 1. 요구사항 분석
- 명시된 기능과 암시된 기능을 모두 파악
- 필요한 도메인 모델 식별
- 비즈니스 규칙 정의

### 2. API 시그니처 작성
- 파일명과 API 시그니처를 구조화된 형태로 작성
- 의존하는 파일명도 명시
api_signatures 필드의 signature 필드는 반드시 다음과 같은 형태로 작성하라.
완성된 코드가 아닌 인터페이스의 형태처럼 작성하라.
Class Todo 
    id: Long,
    title: String,
    description: String,
    priority: Int

Class TodoService
    getTodo(Long id): Todo, 
    createTodo(Todo todo): Todo

### 3. 기술 스택 결정
- Spring Boot 기반 (Java 17, Spring Boot 3.x)
- 필요한 의존성 명시 (JPA, Security, Validation 등)
- 데이터베이스(H2 고정) 및 기타 인프라

## 출력 형식
- 주어진 Pydantic 모델(Specification) 형식으로 출력
- API 시그니처는 APISignature 모델 리스트로 구조화
- 명확하고 구현 가능한 수준으로 작성

## 중요 원칙
1. **구체성**: 모호한 표현 금지, 구현 가능한 수준으로 작성
2. **완전성**: 모든 필요한 API와 기능을 빠짐없이 포함
3. **일관성**: 명명 규칙, 응답 형식 등 일관성 유지
4. **실용성**: 과도한 설계 지양, MVP 수준의 실용적 설계
"""
    
    prompt = ChatPromptTemplate([
        ("system", system_prompt),
        ("human", "사용자 요청: {user_request}\n\n위 요청을 분석하여 완전한 API 시그니처를 작성해주세요.")
    ])
    
    chain = prompt | llm.with_structured_output(Specification, include_raw=True)
    
    response = chain.invoke({"user_request": user_request})
    
    specification = response["parsed"]
    raw_message = response["raw"]
    
    print(f"✅ API 시그니처 작성 완료")
    print(f"  - 프로젝트: {specification.title}")
    print(f"  - API 개수: {len(specification.api_signatures)}개")
    print(f"  - 기술 스택: {specification.technical_stack}\n")

    print(f"📋 파일 계획 목록:")
    for i, file_plan in enumerate(specification.api_signatures):
        print(f"  - {i+1}. {file_plan.file_name}: {file_plan.description}")

    token_usage_list = state.get("token_usage_list", [])
    token_usage = extract_token_usage(raw_message, "Specification Writer Agent")
    token_usage_list.append(token_usage)

    sum_result = [f"{content.file_name}: {content.description}" for content in specification.api_signatures]
    sum_result = "\n".join(sum_result)

    return {
        "files_plan": specification.api_signatures,
        "pre_result": sum_result,
        "current_status": "orchestrator",  # 오케스트라가 판단하도록
        "token_usage_list": token_usage_list
    }

# ============================================
# 2. Code Generator Agent (단일 파일 생성)
# ============================================

def code_generator_agent(state: AgenticCoderState) -> Dict[str, Any]:
    """
    역할: 단일 파일 코드 생성자
    입력: 오케스트라가 지정한 파일 정보 (next_file_to_generate)
    출력: 단일 파일 코드
    
    주요 작업:
    - 오케스트라가 지정한 파일 하나만 생성
    - 이미 생성된 파일들(generated_files)을 참조
    - 명세서와 일관성 유지
    """
    print("\n" + "="*80)
    print("💻 [Code Generator Agent] 단일 파일 생성")
    print("="*80)
    
    import json
    
    next_file = state.get("next_file_to_generate")
    if not next_file:
        print("⚠️ 생성할 파일 정보가 없습니다.")
        return {"current_status": "orchestrator"}
    
    generated_files = state.get("generated_files", [])
    
    print(f"생성할 파일: {next_file['file_name']}")
    print(f"경로: {next_file['file_path']}")
    print(f"설명: {next_file['description']}")
    print(f"이미 생성된 파일: {len(generated_files)}개\n")
    
    llm = get_llm()
    
    system_prompt = """
당신은 **Spring Boot 전문 개발자**입니다.

주어진 명세서와 파일 정보를 바탕으로 **단일 파일의 완전한 코드**를 생성하는 것이 당신의 임무입니다.

## 핵심 작업

### 1. 현재 파일만 생성
- 주어진 파일 하나만 집중
- 파일 타입에 맞는 구현:
  - **Entity**: JPA 엔티티 (@Entity, @Id, Lombok)
  - **Repository**: Spring Data JPA 인터페이스
  - **DTO**: Request/Response 객체 (Validation 포함)
  - **Service**: 비즈니스 로직 (@Service, @Transactional)
  - **Controller**: REST API 엔드포인트
  - **Exception**: 커스텀 예외, 글로벌 핸들러

### 2. 이미 생성된 파일 활용
- 의존하는 파일들의 코드 참조
- 클래스명, 패키지명, 스타일 일관성 유지
- 타입 호환성 보장

### 3. 명세서 준수
- API 시그니처 정확히 구현
- HTTP 메서드, 경로, 요청/응답 형식 일치

### 4. 코드 품질
- Spring Boot 베스트 프랙티스
- 적절한 예외 처리
- Validation 어노테이션
- 깔끔하고 자명한 코드

## 출력 형식
- SingleFileGeneration 모델로 출력
- file_name, file_path, code_content
- 완전한 Java 코드 (import부터 끝까지)

## 중요 원칙
1. **완전성**: 모든 import, 어노테이션, 메서드 포함
2. **일관성**: 기존 파일들과 스타일 일치
3. **실행 가능성**: 컴파일 가능한 코드
4. **집중**: 현재 파일만 생성
"""
    
    # 의존 파일 컨텍스트
    dependency_context = ""
    if next_file.get("dependencies") and generated_files:
        dependency_context = "\n\n### 참고: 의존하는 파일들\n"
        for dep_name in next_file["dependencies"]:
            for gen_file in generated_files:
                if gen_file["file_name"] == dep_name:
                    dependency_context += f"\n// {gen_file['file_name']}\n"
                    dependency_context += f"{gen_file['code_content'][:800]}...\n"
                    break
    
    # 생성된 파일 요약
    generated_summary = ""
    if generated_files:
        generated_summary = f"\n\n### 이미 생성된 파일 ({len(generated_files)}개):\n"
        for gf in generated_files:
            generated_summary += f"- {gf['file_name']} ({gf['file_path']})\n"
    
    prompt = ChatPromptTemplate([
        ("system", system_prompt),
        ("human", """
다음 파일을 생성해주세요.

생성할 파일:
- 파일명: {file_name}
- 경로: {file_path}
- 설명: {description}
반드시 다음 시그니처를 준수하라.
{signature}
{generated_summary}
{dependency_context}

위 정보를 바탕으로 {file_name} 파일의 완전한 코드를 생성해주세요.
""")
    ])
    
    chain = prompt | llm.with_structured_output(SingleFileGeneration, include_raw=True)
    
    response = chain.invoke({
        "file_name": next_file["file_name"],
        "file_path": next_file["file_path"],
        "signature": next_file["signature"],
        "description": next_file["description"],
        "generated_summary": generated_summary,
        "dependency_context": dependency_context
    })
    
    file_gen = response["parsed"]
    raw_message = response["raw"]
    
    print(f"✅ 파일 생성 완료")
    print(f"  - 파일: {file_gen.file_name}")
    print(f"  - 코드 길이: {len(file_gen.code_content)} 자\n")

    print(f"📋 생성된 코드:")
    print(f"  - {file_gen.code_content}")
    print("="*80)
    
    # 생성된 파일 추가
    new_file = {
        "file_name": file_gen.file_name,
        "file_path": file_gen.file_path,
        "code_content": file_gen.code_content,
        "description": next_file["description"]
    }
    
    updated_generated_files = generated_files + [new_file]
    current_index = state.get("current_file_index", 0)

    token_usage_list = state.get("token_usage_list", [])
    token_usage = extract_token_usage(raw_message, "Code Generator Agent")
    token_usage_list.append(token_usage)
    
    return {
        "pre_result": "generated: " + file_gen.file_name + ": " + next_file["description"],
        "current_file_code": file_gen.code_content,
        "generated_files": updated_generated_files,
        "current_file_index": current_index + 1,
        "current_status": "orchestrator",  # 오케스트라가 다음 판단
        "token_usage_list": token_usage_list
    }


# ============================================
# 3. Static Reviewer Agent
# ============================================

def static_reviewer_agent(state: AgenticCoderState) -> Dict[str, Any]:
    """
    역할: 정적 분석 및 리뷰어
    입력: 생성된 코드
    출력: 리뷰 결과 (이슈 목록, 통과 여부)
    
    주요 작업:
    - 코드 정적 분석
    - 잠재적 버그 탐지 (Null Pointer, Resource Leak 등)
    - 보안 취약점 검사
    - 코드 스멜 탐지
    - 베스트 프랙티스 준수 여부
    - 개선 제안
    """
    print("\n" + "="*80)
    print("🔍 [Static Reviewer Agent] 정적 리뷰 시작")
    print("="*80)
    
    generated_files = state.get("generated_files", [])
    
    if not generated_files:
        print("⚠️ 리뷰할 파일이 없습니다.")
        return {"current_status": "completed"}
    
    print(f"리뷰할 파일 수: {len(generated_files)}개\n")
    
    # 모든 코드를 하나로 합치기
    all_code = ""
    for file in generated_files:
        all_code += f"\n\n{'='*80}\n"
        all_code += f"파일: {file['file_path']}/{file['file_name']}\n"
        all_code += f"{'='*80}\n"
        all_code += file['code_content']
    
    llm = get_llm()
    
    system_prompt = """
당신은 **시니어 Java/Spring Boot 개발자이자 코드 리뷰 전문가**입니다.

생성된 코드를 **정적으로 분석하고 리뷰**하는 것이 당신의 임무입니다.

## 검사 항목

### 1. 잠재적 버그 (CRITICAL/MAJOR)
- **NullPointerException 위험**
  - Optional 처리 누락
  - Null 체크 없는 메서드 호출
  - findById(), orElseThrow() 등 적절한 처리 여부
  
- **Resource Leak**
  - Stream, Connection 등 리소스 정리
  
- **동시성 이슈**
  - Thread-safety 문제
  - 공유 상태 관리

### 2. 보안 취약점 (CRITICAL/MAJOR)
- **SQL Injection**: JPQL, Native Query 검사
- **인증/인가 누락**: 보안이 필요한 API에 @PreAuthorize 등 누락
- **민감 정보 노출**: 비밀번호 평문 저장, 로그에 민감정보 출력
- **CSRF, XSS 대응**

### 3. Spring Boot 베스트 프랙티스 (MAJOR/MINOR)
- **의존성 주입**: 생성자 주입 사용 (필드 주입 지양)
- **트랜잭션**: @Transactional 적절한 위치 및 옵션
- **예외 처리**: Custom Exception, @ControllerAdvice 활용
- **Validation**: @Valid, @NotNull 등 적절한 사용
- **Layered Architecture**: 레이어 간 책임 분리

### 4. 코드 스멜 (MINOR/INFO)
- **긴 메서드**: 메서드가 너무 긴 경우
- **중복 코드**: 반복되는 로직
- **매직 넘버/문자열**: 상수화 필요
- **과도한 결합도**: 클래스 간 의존성 과다

### 5. 명명 및 컨벤션 (MINOR)
- **명명 규칙**: 클래스, 메서드, 변수명 적절성
- **Java 컨벤션**: Camel Case, Pascal Case 등

## 심각도 분류
- **CRITICAL**: 즉시 수정 필요 (보안, 치명적 버그)
- **MAJOR**: 반드시 수정 권장 (잠재적 버그, 중요 베스트 프랙티스)
- **MINOR**: 개선 권장 (코드 품질, 가독성)
- **INFO**: 참고사항 (최적화 제안 등)

## 리뷰 통과 기준
- CRITICAL 이슈: 0개
- MAJOR 이슈: 3개 이하
- 위 기준을 만족하면 passed=True, 아니면 passed=False

## 출력 형식
- 주어진 Pydantic 모델(StaticReviewResult) 형식으로 출력
- 각 이슈는 CodeIssue 형식으로 구조화
- 구체적인 파일명, 줄 번호(가능한 경우), 이슈 설명, 개선 제안 포함

## 중요 원칙
1. **구체성**: "문제가 있습니다" X, "UserService.java의 findById() 호출 시 null 체크 누락" O
2. **실용성**: 사소한 이슈보다 중요한 이슈에 집중
3. **건설적**: 비판보다는 개선 제안 중심
4. **정확성**: 실제 문제만 지적, 오탐 최소화
"""
    
    prompt = ChatPromptTemplate([
        ("system", system_prompt),
        ("human", """
다음 생성된 코드를 정적으로 분석하고 리뷰해주세요.

생성된 코드:
{generated_code}

모든 잠재적 이슈를 찾아내고, 심각도를 평가하며, 구체적인 개선 제안을 제공해주세요.
""")
    ])
    
    chain = prompt | llm.with_structured_output(StaticReviewResult, include_raw=True)
    
    response = chain.invoke({"generated_code": all_code})
    
    review_result = response["parsed"]
    raw_message = response["raw"]
    
    print(f"✅ 정적 리뷰 완료")
    print(f"  - 통과 여부: {'✅ PASS' if review_result.passed else '❌ FAIL'}")
    print(f"  - 발견된 이슈: {len(review_result.issues)}개")
    print(f"  - 요약: {review_result.summary}\n")
    
    if review_result.issues:
        print("🔍 발견된 이슈:")
        for issue in review_result.issues:
            severity_emoji = {
                "CRITICAL": "🔴",
                "MAJOR": "🟠",
                "MINOR": "🟡",
                "INFO": "🔵"
            }.get(issue.severity, "⚪")
            
            location = f"{issue.file_name}"
            if issue.line_number:
                location += f":{issue.line_number}"
            
            print(f"  {severity_emoji} [{issue.severity}] {location}")
            print(f"     {issue.issue_type}: {issue.description}")
            if issue.suggestion:
                print(f"     💡 제안: {issue.suggestion}")
            print()
    
    if review_result.recommendations:
        print("📌 전반적인 개선 권장사항:")
        for rec in review_result.recommendations:
            print(f"  - {rec}")
    
    # 리뷰 결과 반환 (다음 행동은 오케스트라가 결정)
    print(f"\n📋 리뷰 완료. 오케스트라에게 결과 전달...")
    
    token_usage_list = state.get("token_usage_list", [])
    token_usage = extract_token_usage(raw_message, "Static Reviewer Agent")
    token_usage_list.append(token_usage)
    
    return {
        "review_result": review_result.model_dump_json(indent=2),
        "review_passed": review_result.passed,
        "issues_found": [f"[{issue.severity}] {issue.file_name}: {issue.description}" 
                        for issue in review_result.issues],
        "current_status": "orchestrator",  # 오케스트라가 다음 결정
        "code_files": generated_files,  # 최종 결과 저장
        "retry_count": state.get("retry_count", 0) + 1,
        "token_usage_list": token_usage_list
    }

