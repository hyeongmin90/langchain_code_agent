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
from schemas import DecomposeAgentState, UserStoriesResult, RefinedUserStoriesResult, FinalUserStoriesResult, RefinedUserStoriesDraft
import asyncio
import pprint

def decompose_into_user_stories(state: DecomposeAgentState):
    print("--- 📝 1단계 결과물 생성 중... ---")

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
    system_prompt = """
    [역할]
    당신은 새로운 프로젝트의 브레인스토밍 세션을 이끄는, 창의적이고 아이디어가 넘치는 애자일 프로덕트 오너(Agile Product Owner)입니다. 
    당신의 주된 임무는 완벽한 계획이 아니라, 고객의 요청 사항에서 나온 가능성을 빠짐없이 포착하여 사용자 스토리의 "초안" 목록을 만드는 것입니다.

    [지시]
    사용자의 요청사항을 바탕으로, 프로젝트의 전체적인 방향을 나타내는 에픽(Epic) 1개와, 
    그 목표를 달성하기 위해 필요한 기능 아이디어를 담은 사용자 스토리 초안 목록(User Story Drafts)을 생성하라.

    [출력 형식]
    - 반드시 유효한 pydantic 모델의 형식으로 출력하라.
    - 최상위에는 epic과 user_stories_draft 두 개의 키가 존재해야 합니다.
    - 각 사용자 스토리는 고유한 id와 함께, 표준 형식인 "As a..., I want to..., so that..." 구조를 따라 as_a, i_want_to, so_that 키로 나누어 작성하십시오.
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


def refine_user_stories(state: DecomposeAgentState):
    print("--- 📝 2단계 결과물 생성 중... ---")

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
    system_prompt = """
    [역할]
    당신은 수많은 프로젝트를 성공으로 이끈 20년 경력의 시니어 애자일 프로덕트 오너(Senior Agile Product Owner)입니다. 
    당신의 특기는 주니어 팀원이 만든 사용자 스토리 초안을 날카롭게 분석하여, 업계 표준인 INVEST 원칙에 따라 명확하고, 가치 있으며, 실행 가능한 스토리로 재탄생시키는 것입니다.

    [지시]  
    주어지는 [사용자 스토리 초안 목록]을 검토하고, 다음 [정제 규칙]에 따라 수정 및 개선하여 정제된 사용자 스토리 목록(Refined User Stories)을 생성하라.

    [정제 규칙 (INVEST 원칙 기반)]
    - 병합 (Merge): 서로 중복되거나 지나치게 유사한 스토리가 있다면, 하나의 명확한 스토리로 병합하라.
    - 분해 (Decompose): 하나의 스토리가 너무 크거나 여러 기능을 포함하고 있다면(Epic에 가깝다면), 독립적으로 개발 가능한 더 작은 스토리 여러 개로 분해하라.
    - 구체화 (Specify): "쉽게", "빠르게", "잘"과 같은 모호한 표현을 사용자가 체감할 수 있는 구체적인 행동이나 결과로 바꾸라. (예: "상품을 쉽게 찾는다" -> "카테고리별로 상품을 필터링한다")
    - 가치 부여 (Add Value): 모든 스토리가 최종 사용자에게 명확한 가치(so_that)를 제공하는지 확인하고, 불분명하다면 가치를 명확히 하라.
    - 우선순위 부여 (Prioritize): 각 스토리가 프로젝트의 핵심 성공(MVP)에 얼마나 중요한지를 판단하여 priority (High, Medium, Low)를 부여하라.

    [출력 형식]
    - 반드시 유효한 pydantic 모델의 형식으로 출력하라.
    - 각 사용자 스토리 객체에는 priority 필드가 반드시 포함되어야 한다.
    - ID는 새로운 고유한 값으로 다시 부여하라.
    - 유사하거나 연관된 스토리는 그룹화하여 그룹화된 스토리 목록으로 생성하라.
    - 그룹은 1-3개의 스토리로 구성되어야 한다.
    """

    prompt = ChatPromptTemplate([
        ("system", system_prompt),
        ("human", "사용자의 최초 요청 사항: {user_request}\n사용자 스토리 초안 목록:\n {user_stories_draft}")
    ])
    
    chain = prompt | llm.with_structured_output(RefinedUserStoriesResult)
    
    result = chain.invoke({
        "user_request": state["user_request"],
        "user_stories_draft": state["raw_user_stories"]
    })

    print("정제된 유저스토리 -----------------")
    for story_group in result.refined_user_stories:
        for story in story_group:
            print(story)
        print("\n--------------------------------")


    return {
        "epic": result.epic,
        "refined_user_stories": result.refined_user_stories
    }


async def generate_final_specifications(state: DecomposeAgentState):
    print("--- 📝 3단계 결과물 생성 중... ---")
    
    LIMIT = 5
    semaphore = asyncio.Semaphore(LIMIT)
    tasks = [llm_call(i, semaphore, user_story_group, state["epic"]) for i, user_story_group in enumerate(state["refined_user_stories"])]

    results = await asyncio.gather(*tasks)

    flattened_stories = [item for sublist in results for item in sublist]

    final_result = {
        "epic": state["epic"],
        "final_user_stories": results
    }

    print("최종 결과물 -----------------")
    for story in flattened_stories:
        print(story.model_dump_json(indent=2))
    print("--------------------------------")

    return {
        "final_specifications": final_result
    }

async def llm_call(task_id: int, semaphore: asyncio.Semaphore, user_story_group: List[RefinedUserStoriesDraft], epic: Epic):
    async with semaphore:
        print(f"Task {task_id} started")

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
        system_prompt = """
        [역할]
        당신은 디테일에 강하고 논리적인 사고를 하는 QA(품질 보증) 엔지니어입니다. 당신의 임무는 프로덕트 오너가 작성한 사용자 스토리를 보고, 개발자가 무엇을 만들어야 하고 테스터가 무엇을 검증해야 하는지 명확히 알 수 있도록, 구체적인 **수용 기준(Acceptance Criteria)**을 시나리오 기반으로 작성하는 것입니다.

        [지시]
        주어진 정제된 사용자 스토리 그룹의 각각에 대해, 개발 완료 여부를 판단할 수 있는 수용 기준을 2개 이상 생성하십시오.

        [수용 기준 작성 규칙]
        - Gherkin 형식 사용: 모든 수용 기준은 "Given-When-Then" 시나리오 형식으로 작성하라.
        - Given: 시나리오가 시작되기 전의 전제 조건
        - When: 사용자가 취하는 특정 행동
        - Then: 그 행동으로 인해 발생해야 하는 기대 결과
        - 다양한 시나리오: 각 스토리에 대해 최소 1개의 성공 시나리오("Happy Path")와 1개 이상의 실패 또는 엣지 케이스 시나리오("Sad Path")를 포함해야 한다.
        - 구체성: "사용자 정보가 보인다"와 같은 모호한 표현 대신, "화면 상단에 사용자의 이름과 이메일 주소가 표시된다"처럼 구체적이고 검증 가능하게 작성하라.

        [출력 형식]
        - 반드시 유효한 pydantic 모델의 형식으로 출력하라.
        - 입력받은 스토리 정보는 그대로 유지하고, acceptance_criteria 라는 키를 추가하라.
        - acceptance_criteria는 각 시나리오를 담은 객체들의 리스트여야 한다.
        - 각 시나리오에는 scenario 필드가 반드시 포함되어야 한다.
        """
        
        prompt = ChatPromptTemplate([
            ("system", system_prompt),  
            ("human", "에픽: {epic}\n정제된 사용자 스토리 그룹:\n {user_story_group}")
        ])

        chain = prompt | llm.with_structured_output(FinalUserStoriesResult)

        result = await chain.ainvoke({
            "user_story_group": user_story_group,
            "epic": epic
        })
        print(f"Task {task_id} completed-------")
        return result.final_user_stories


async def main(user_request: str):
    load_dotenv()

    workflow = StateGraph(DecomposeAgentState)

    workflow.add_node("decompose_into_user_stories", decompose_into_user_stories)
    workflow.add_node("refine_user_stories", refine_user_stories)
    workflow.add_node("generate_final_specifications", generate_final_specifications)

    workflow.add_edge(START, "decompose_into_user_stories")
    workflow.add_edge("decompose_into_user_stories", "refine_user_stories")
    workflow.add_edge("refine_user_stories", "generate_final_specifications")
    workflow.add_edge("generate_final_specifications", END)

    app = workflow.compile()

    initial_state = {
        "user_request": user_request,
        "epic": None,
        "raw_user_stories": None,
        "refined_user_stories": None,
        "final_specifications": None
    }

    final_state = await app.ainvoke(initial_state)

    return final_state["final_specifications"]

