import os
import json
from dotenv import load_dotenv
import operator
from typing import TypedDict, Annotated, List, Optional, Literal
from pydantic import BaseModel, Field
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END
from schemas import RefinedUserStoriesDraft, CrossCuttingConcern, NonFunctionalRequirements, ScopeAndConstraints

class Module(BaseModel):
    name: str = Field(description="모듈 이름 (예: auth, todo)")
    epic: str = Field(description="연결된 에픽 이름")
    user_story_ids: List[str] = Field(description="이 모듈이 구현하는 유저스토리 ID 목록")
    module_type: Literal["core", "supporting", "generic"] = Field(
        description="모듈 타입 - core: 핵심 비즈니스, supporting: 지원, generic: 공통"
    )
    package: str = Field(description="Java 패키지 경로 (예: com.example.todo.domain.auth)")
    responsibility: str = Field(description="모듈의 책임 (1-2문장)")
    entities: List[str] = Field(description="이 모듈이 관리하는 Entity 목록")
    dependencies: List[str] = Field(description="의존하는 다른 모듈 이름 목록")
    public_interfaces: List[str] = Field(
        description="다른 모듈에 제공하는 주요 메서드 시그니처 목록"
    )

class GlobalConfigItem(BaseModel):
    name: str = Field(description="설정 이름 (예: security, exception_handling)")
    based_on: str = Field(description="기반이 되는 횡단 관심사 이름")
    implementation: str = Field(description="구현 방식")
    config_class: str = Field(description="Spring Config 클래스명 (예: SecurityConfig)")
    affected_modules: List[str] = Field(
        description="영향받는 모듈 목록 (전체면 ['all'])"
    )

class ModuleDesignOutput(BaseModel):
    modules: List[Module] = Field(description="Spring 모듈 목록")
    global_config: List[GlobalConfigItem] = Field(description="전역 설정 (횡단 관심사 기반)")

class FirstInput(BaseModel):
    epic: str = Field(description="프로젝트의 최상위 목표를 정의하는 에픽")
    refined_user_stories_grouped: List[GroupedEpic] = Field(description="기능별 분류된 유저 스토리 목록")
    cross_cutting_concerns: List[CrossCuttingConcern] = Field(description="그룹화된 유저 스토리의 횡단 관심사 목록")
    non_functional_requirements: NonFunctionalRequirements = Field(description="비기능 요구사항 목록")
    scope_and_constraints: ScopeAndConstraints = Field(description="프로젝트 범위 및 제약")

class ArchitectureDesignState(TypedDict):
    first_input: FirstInput
    architecture: ModuleDesignOutput

def design_architecture(state: ArchitectureDesignState):
    print("--- 📝 전체 아키텍처 설계 시작 ---")

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
    system_prompt = """
    당신은 Spring Boot 아키텍트입니다. 기능별로 그룹화된 에픽을 분석하여 전체적인 Spring Boot 모듈 구조를 설계하라.
    주어진 입력 데이터를 바탕으로 아키텍처를 설계하라.

    [작업]

    1. 에픽 → Spring 모듈 매핑

    각 에픽을 하나의 Spring 도메인 모듈로 매핑합니다.

    [기본 원칙]
    - 1 에픽 = 1 모듈 (원칙)
    - 에픽이 너무 복잡하면 하위 모듈로 분할 고려
    - 에픽이 너무 단순하면 다른 모듈에 통합 고려

    [모듈 정보]
    - name: 모듈 이름 (소문자, 단수형, 예: auth, todo, share)
    - epic: 연결된 에픽 이름
    - user_story_ids: 에픽의 group_user_stories_id 그대로 사용
    - type: 모듈 타입 분류
    - core: 핵심 비즈니스 가치 제공 (매출, 핵심 기능)
    - supporting: 핵심을 지원 (인증, 알림)
    - generic: 공통 기능 (로깅, 설정)
    - package: com.example.(프로젝트명).domain.(모듈명)
    - responsibility: 이 모듈의 책임 (2-3문장)
    - entities: key_entities에서 이 모듈이 관리할 엔티티만 선택
    - dependencies: group_dependencies의 "to"를 모듈명으로 변환
    - public_interfaces: 다른 모듈이 사용할 주요 메서드 시그니처

    [public_interfaces 작성 가이드]
    - 메서드명(파라미터): 리턴타입 형식
    - 예: createTodo(userId, content, dueDate): Todo
    - 예: login(email, password): Token
    - 각 모듈당 3-7개 정도의 주요 인터페이스

    2. 모듈 타입 분류

    [core (핵심 도메인)]
    - 비즈니스 가치를 직접 제공
    - 프로젝트의 존재 이유
    - 예: 주문, 결제, 할 일 관리

    [supporting (지원 도메인)]
    - 핵심 도메인을 지원
    - 그 자체로는 비즈니스 가치 없음
    - 예: 인증, 알림, 파일 업로드

    [generic (일반 도메인)]
    - 어떤 프로젝트에서나 필요
    - 재사용 가능한 공통 기능
    - 예: 로깅, 모니터링, 설정 관리

    3. 의존성 검증

    [체크 사항]
    - 순환 의존 없는지 확인 (A→B→A)
    - core 모듈이 supporting 모듈에 의존하는 것은 자연스러움
    - supporting이 core에 의존하면 문제 (재설계 필요)

    [순환 의존 발견 시]
    - 모듈 경계 재조정
    - 또는 이벤트 기반으로 변경 (triggers)

    4. Global Config 설계

    횡단 관심사를 Spring Global Config로 매핑합니다.

    다음 3개는 기본적으로 포함할 것
    [security]
    - cross_cutting_concerns에서 "인증" 관련 찾기
    - affected_epics를 protected_modules로 변환

    [exception_handling]
    - "예외 처리", "에러" 관련 찾기
    - 모든 모듈에 영향

    [logging]
    - "로깅" 관련 찾기
    - 모든 모듈에 영향

    5. Dependency Graph 생성

    각 모듈의 dependencies를 그래프 형태로 정리합니다.
    {
    "모듈A": [],
    "모듈B": ["모듈A"],
    "모듈C": ["모듈A", "모듈B"]
    }

    [출력 형식]
    반드시 Pydantic 스키마에 맞춰 JSON을 출력하라.

    [주의사항]
    - 모듈 이름은 소문자, 단수형으로 작성하라
    - 순환 의존이 발견되면 모듈 경계를 재조정하라
    - public_interfaces는 실제 구현 가능한 메서드 시그니처로 작성하라
    - entities는 각 모듈이 직접 관리하는 엔티티만 포함하라
    - cross_cutting_concerns가 없어도 global_config의 기본 구조는 유지하라
    """

    human_prompt = """
    프로젝트 전체 에픽
    {epic}
    기능별로 그룹화된 에픽 목록
    {refined_user_stories_grouped}
    횡단 관심사
    {cross_cutting_concerns}
    비기능 요구사항 목록
    {non_functional_requirements}
    프로젝트 범위 및 제약
    {scope_and_constraints}
    """
    prompt = ChatPromptTemplate([
        ("system", system_prompt),
        ("human", human_prompt)
    ])
    
    chain = prompt | llm.with_structured_output(ModuleDesignOutput)
    
    result = chain.invoke({
        "epic": state["first_input"].epic,
        "refined_user_stories_grouped": state["first_input"].refined_user_stories_grouped,
        "cross_cutting_concerns": state["first_input"].cross_cutting_concerns,
        "non_functional_requirements": state["first_input"].non_functional_requirements,
        "scope_and_constraints": state["first_input"].scope_and_constraints
    })

    return { "architecture": result }

def design_modules(state: ArchitectureDesignState):
    print("--- 📝 모듈 설계 시작 ---")

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
    system_prompt = """
    당신은 Spring Boot 모듈 설계자입니다. 기능별로 그룹화된 에픽을 분석하여 전체적인 Spring Boot 모듈 구조를 설계하라.
    주어진 입력 데이터를 바탕으로 모듈 구조를 설계하라.
    """
    
    human_prompt = """
    프로젝트 전체 에픽
    {epic}
    기능별로 그룹화된 에픽 목록
    {refined_user_stories_grouped}
    횡단 관심사
    {cross_cutting_concerns}
    """
    prompt = ChatPromptTemplate([

    chain = prompt | llm.with_structured_output(ModuleDesignOutput)
    
    result = chain.invoke({
        "epic": state["first_input"].epic,
        "refined_user_stories_grouped": state["first_input"].refined_user_stories_grouped,
        "cross_cutting_concerns": state["first_input"].cross_cutting_concerns
    })

    


def main(first_input: FirstInput):
    load_dotenv()

    workflow = StateGraph(ArchitectureDesignState)

    workflow.add_node("design_architecture", design_architecture)

    workflow.add_edge(START, "design_architecture")
    workflow.add_edge("design_architecture", END)

    app = workflow.compile()

    initial_state = {
        "first_input": first_input,
        "architecture": None
    }

    final_state = app.invoke(initial_state)

    print("아키텍처 설계 결과 -----------------")
    print(final_state["architecture"].model_dump_json(indent=2))
    print("--------------------------------")

    return final_state["architecture"]