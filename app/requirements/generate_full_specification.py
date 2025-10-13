import os
import json
from dotenv import load_dotenv
import operator
from typing import TypedDict, Annotated, List, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, START, END
from schemas import DecomposeAgentState, UserStoriesResult, RefinedUserStoriesResult, FinalUserStoriesResult, RefinedUserStoriesDraft, NonFunctionalRequirements, ScopeAndConstraints
import asyncio
import pprint

def decompose_into_user_stories(state: DecomposeAgentState):
    print("--- 📝 유저 스토리 초안 생성 중... ---")

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
    system_prompt = """
    [역할]
    당신은 새로운 프로젝트의 브레인스토밍 세션을 이끄는, 창의적이고 아이디어가 넘치는 애자일 프로덕트 오너(Agile Product Owner)입니다. 
    당신의 주된 임무는 완벽한 계획이 아니라, 고객의 요청 사항에서 나온 가능성을 빠짐없이 포착하여 사용자 스토리의 "초안" 목록을 만드는 것입니다.

    [지시]
    사용자의 요청사항을 바탕으로, 프로젝트의 전체적인 방향을 나타내는 에픽(Epic) 1개와, 
    그 목표를 달성하기 위해 필요한 기능 아이디어를 담은 유저 스토리 초안 목록(User Story Drafts)을 생성하라.


    [출력 형식]
    - 반드시 유효한 pydantic 모델의 형식으로 출력하라.
    - 최상위에는 epic과 user_stories_draft 두 개의 키가 존재해야 합니다.
    - 각 유저 스토리는 고유한 id와 함께, 표준 형식인 "As a..., I want to..., so that..." 구조를 따라 as_a, i_want_to, so_that 키로 나누어 작성하십시오.
    """
    
    prompt = ChatPromptTemplate([
        ("system", system_prompt),
        ("human", "사용자 요청사항: {user_request}")
    ])
    
    chain = prompt | llm.with_structured_output(UserStoriesResult)
    
    result = chain.invoke({
        "user_request": state["user_request"]
    })
    print("작성된 유저스토리 -----------------")
    print(result)
    print("--------------------------------")

    return {
        "raw_user_stories": result
    }

def generate_non_functional_requirements(state: DecomposeAgentState):
    print("--- 📝 비기능적 요구사항 생성 중... ---")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
    system_prompt = """
    [역할]
    당신은 수많은 프로젝트를 성공으로 이끈 20년 경력의 시니어 애자일 프로덕트 오너입니다. 
    당신의 임무는 주어진 유저 스토리 목록을 분석하여, 프로젝트의 비기능적 요구사항을 명확히 하는 것입니다.

    [지시]
    주어지는 유저 스토리 목록을 검토하고 다음 [non_functional_requirements 필드]와 [scope_and_constraints 필드]에 따라 수정 및 개선하여 비기능적 요구사항을 생성하라.

    [non_functional_requirements 필드]
    프로젝트 전체에 적용되어야 하는 비기능적 요구사항을 문자열 목록으로 여기에 작성하십시오.
    보안(비밀번호 암호화), 성능(응답 시간 목표), 안정성, 로깅 등 일반적으로 누락되기 쉬운 항목들을 반드시 점검하고 포함시키십시오.
    
    [scope_and_constraints 필드]
    사용자 스토리와 전체 대화 내용을 바탕으로 프로젝트의 범위(Scope), 가정(Assumptions), 제약(Constraints)을 명확히 식별하십시오.
    식별된 내용을 각각의 목록 필드에 나누어 기술하십시오.

    [출력 형식]
    - 반드시 유효한 pydantic 모델의 형식으로 출력하라.
    """
    prompt = ChatPromptTemplate([
        ("system", system_prompt),
        ("human", "에픽:\n {epic}\n\n유저 스토리 목록:\n {refined_user_stories}\n\n사용자의 최초 요청 사항:\n {user_request}")
    ])
    chain = prompt | llm.with_structured_output(NonFunctionalRequirements)
    result = chain.invoke({
        "epic": state["epic"],
        "refined_user_stories": state["refined_user_stories"],
        "user_request": state["user_request"]
    })

    print("작성된 비기능적 요구사항 -----------------")
    print(result)
    print("--------------------------------")

    return {
        "non_functional_requirements": result
    }


def refine_user_stories(state: DecomposeAgentState):
    print("--- 📝 유저 스토리 정제 중... ---")

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
    system_prompt = """
    [역할]
    당신은 수많은 프로젝트를 성공으로 이끈 20년 경력의 시니어 애자일 프로덕트 오너입니다. 
    당신의 특기는 주니어 팀원이 만든 사용자 스토리 초안을 날카롭게 분석하여, 명확하고, 가치 있으며, 실행 가능한 스토리로 재탄생시키는 것입니다.

    [지시]  
    주어지는 [사용자 스토리 초안 목록]을 검토하고, 다음 [정제 규칙]에 따라 수정 및 개선하여 정제된 사용자 스토리 목록(Refined User Stories)을 생성하라.

    [정제 규칙]
    - 병합 (Merge): 서로 중복되거나 지나치게 유사한 스토리가 있다면, 하나의 명확한 스토리로 병합하라.
    - 분해 (Decompose): 하나의 스토리가 너무 크거나 여러 기능을 포함하고 있다면(Epic에 가깝다면), 독립적으로 개발 가능한 더 작은 스토리 여러 개로 분해하라.
    - 구체화 (Specify): "쉽게", "빠르게", "잘"과 같은 모호한 표현을 사용자가 체감할 수 있는 구체적인 행동이나 결과로 바꾸라. (예: "상품을 쉽게 찾는다" -> "카테고리별로 상품을 필터링한다")
    - 가치 부여 (Add Value): 모든 스토리가 최종 사용자에게 명확한 가치(so_that)를 제공하는지 확인하고, 불분명하다면 가치를 명확히 하라.

    [출력 형식]
    - 반드시 유효한 pydantic 모델의 형식으로 출력하라.
    - ID는 새로운 고유한 값으로 다시 부여하라.
    """

    human_prompt = """
    사용자의 최초 요청 사항:
    {user_request}
    에픽:
    {epic}
    사용자 스토리 초안 목록:
    {user_stories_draft}
    확정된 비기능적 요구사항:
    {non_functional_requirements}
    """

    prompt = ChatPromptTemplate([
        ("system", system_prompt),
        ("human", human_prompt)
    ])
    
    chain = prompt | llm.with_structured_output(UserStoriesResult)
    
    result = chain.invoke({
        "user_request": state["user_request"],
        "epic": state["epic"],
        "user_stories_draft": state["raw_user_stories"],
        "non_functional_requirements": state["non_functional_requirements"]
    })

    print("정제된 유저스토리 -----------------")
    print(result)
    print("--------------------------------")

    return {
        "epic": result.epic,
        "refined_user_stories": result.user_stories_draft
    }


def split_by_group(state: DecomposeAgentState):
    print("--- 📝 유저 스토리 그룹화 중... ---")  
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
    system_prompt = """
    [역할]
    당신은 수많은 프로젝트를 성공으로 이끈 20년 경력의 시니어 애자일 프로덕트 오너입니다. 
    
    [지시]  
    주어지는 유저 스토리 목록을 검토하고 우선순위를 부여하고, 다음 [우선순위 부여 규칙]과 [그룹화 규칙]에 따라 수정 및 개선하여 그룹화된 유저 스토리 목록을 생성하라.

    [우선 순위 부여 규칙]
    - 우선순위 부여 (Prioritize): 각 스토리가 프로젝트의 핵심 성공(MVP)에 얼마나 중요한지를 판단하여 priority (High, Medium, Low)를 부여하라.

    [그룹화 규칙]
    - 유사하거나 연관된 스토리는 그룹화하여 그룹화된 스토리 목록으로 생성하라.
    - 그룹은 1-3개의 스토리로 구성되어야 한다.
    - 1개의 스토리로 이뤄진 그룹의 생성을 최대한 피하라.

    [출력 형식]
    - 반드시 유효한 pydantic 모델의 형식으로 출력하라.
    - 그룹화된 스토리 목록은 그룹화된 스토리 목록으로 생성하라.
    """

    prompt = ChatPromptTemplate([
        ("system", system_prompt),
        ("human", "에픽:\n {epic}\n\n정제된 유저 스토리 목록:\n {refined_user_stories}")
    ])
    
    chain = prompt | llm.with_structured_output(RefinedUserStoriesResult)
    
    result = chain.invoke({
        "epic": state["epic"],
        "refined_user_stories": state["refined_user_stories"]
    })
    
    return {
        "refined_user_stories_grouped": result.refined_user_stories
    }


async def generate_final_specifications(state: DecomposeAgentState):
    print("--- 📝 유저 스토리 상세화 중... ---")
    
    LIMIT = 5
    semaphore = asyncio.Semaphore(LIMIT)
    project_brief = f"에픽: {state['epic']}\n 비기능적 요구사항: {state['non_functional_requirements'].non_functional_requirements}\n 범위, 가정, 제약: {state['non_functional_requirements'].scope_and_constraints}"
    tasks = [llm_call(i, semaphore, user_story_group, project_brief) for i, user_story_group in enumerate(state["refined_user_stories_grouped"])]

    
    results = await asyncio.gather(*tasks)

    flattened_stories = [item for sublist in results for item in sublist]

    final_result = {
        "final_user_stories": results
    }

    print("최종 결과물 -----------------")
    for story in flattened_stories:
        print(story.model_dump_json(indent=2))
    print("--------------------------------")

    return {
        "final_specifications": final_result
    }

async def llm_call(
    task_id: int, 
    semaphore: asyncio.Semaphore, 
    user_story_group: List[RefinedUserStoriesDraft], 
    project_brief: str
    ):
    async with semaphore:
        print(f"Task {task_id} started")

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
        system_prompt = """
        [역할]
        당신은 20년 경력의 시니어 소프트웨어 아키텍트이자 개발 리더입니다. 
        당신의 임무는 프로덕트 오너가 작성한 사용자 스토리를 보고, 개발자가 즉시 코딩을 시작하고 QA 엔지니어가 테스트 케이스를 작성할 수 있는, 
        완벽하고 실행 가능한 기술 명세서를 작성하는 것입니다.

        [핵심 지침]
        이 명세서는 한 번 확정되면 다시 수정되지 않는다는 전제 하에 작성됩니다. 
        따라서 추후에 해석의 여지가 생기지 않도록, 모든 항목을 최대한 정확하고 세세하게 기술해야 하며 확정되어야 합니다.
        당신의 유일한 목표는 제공된 Pydantic 모델의 모든 필드를 아래 규칙에 맞게 정확하게 채우는 것입니다.

        [지시 및 작성 규칙]
        주어진 **[입력 데이터]**를 바탕으로, 각 사용자 스토리에 대해 아래의 detailed_specification과 acceptance_criteria 필드를 채워라.

        1. detailed_specification 필드 작성 규칙:
        목표: 이 스토리를 구현하는 데 필요한 모든 기능적 요구사항을 상세히 기술하라.

        포함할 내용:
        데이터 모델: 이 기능을 위해 필요한 데이터베이스 테이블 스키마 또는 객체 모델을 정의하라. (필드명, 타입, 제약조건 등)
        UI/UX 동작: 사용자가 보게 될 화면의 구성 요소와 구체적인 상호작용 방식을 설명하라.
        유효성 검사 규칙: 서버와 클라이언트 양쪽에서 수행되어야 할 모든 데이터 유효성 검사 규칙을 명시하라. (필수 여부, 길이 제한, 형식 등)
        기능 로직: 기능이 어떤 순서로, 어떤 조건에 따라 동작해야 하는지 논리적인 흐름을 설명하십시오.

        2. acceptance_criteria 필드 작성 규칙:
        목표: 이 스토리가 '완료'되었음을 객관적으로 증명할 수 있는 여러 개의 검증 시나리오를 List[AcceptanceCriteria] 형태로 작성하라.
        포함할 내용:
        다양한 시나리오: 각 스토리에 대해 **최소 1개의 성공 시나리오("Happy Path")**와 **1개 이상의 실패 또는 엣지 케이스 시나리오("Sad Path")**를 반드시 포함해야 합니다.
        Gherkin 형식: 모든 시나리오는 다음 형식을 엄격히 준수해야 합니다.
        scenario: 시나리오의 명확한 제목 (예: "성공적인 프로젝트 등록")
        given: 시나리오가 시작되기 전의 전제 조건
        when: 사용자가 취하는 특정 행동
        then: 그 행동으로 인해 발생해야 하는 기대 결과
        구체성: "사용자 정보가 보인다"와 같은 모호한 표현 대신, "화면 상단에 사용자의 이름과 이메일 주소가 표시된다"처럼 구체적이고 검증 가능하게 작성하라.

        [역할 경계]
        이 명세서는 '무엇을(What)' 만들지에만 집중합니다. '어떻게(How)' 만들지에 해당하는 특정 기술 스택(예: React, Django)이나 라이브러리 이름은 절대로 명시하지 마라.

        [산출물 기준]
        최종 산출물은 입력받은 user_story_group의 각 스토리 객체에 detailed_specification과 acceptance_criteria 필드가 완벽하게 채워진 Pydantic 모델 형식의 JSON이어야 한다.
        """
        
        prompt = ChatPromptTemplate([
            ("system", system_prompt),  
            ("human", "프로젝트 브리프: {project_brief}\n정제된 사용자 스토리 그룹:\n {user_story_group}")
        ])

        chain = prompt | llm.with_structured_output(FinalUserStoriesResult)

        result = await chain.ainvoke({
            "user_story_group": user_story_group,
            "project_brief": project_brief
        })
        print(f"Task {task_id} completed-------")
        return result.final_user_stories

async def main(user_request: str):
    load_dotenv()

    workflow = StateGraph(DecomposeAgentState)

    workflow.add_node("decompose_into_user_stories", decompose_into_user_stories)
    workflow.add_node("generate_non_functional_requirements", generate_non_functional_requirements)
    workflow.add_node("refine_user_stories", refine_user_stories)
    workflow.add_node("split_by_group", split_by_group)
    workflow.add_node("generate_final_specifications", generate_final_specifications)

    workflow.add_edge(START, "decompose_into_user_stories")
    workflow.add_edge("decompose_into_user_stories", "generate_non_functional_requirements")
    workflow.add_edge("generate_non_functional_requirements", "refine_user_stories")
    workflow.add_edge("refine_user_stories", "split_by_group")
    workflow.add_edge("split_by_group", "generate_final_specifications")
    workflow.add_edge("generate_final_specifications", END)

    app = workflow.compile()

    initial_state = {
        "user_request": user_request,
        "epic": None,
        "raw_user_stories": None,
        "refined_user_stories": None,
        "refined_user_stories_grouped": None,
        "non_functional_requirements": None,
        "final_specifications": None
    }

    final_state = await app.ainvoke(initial_state)

    return final_state["final_specifications"]

