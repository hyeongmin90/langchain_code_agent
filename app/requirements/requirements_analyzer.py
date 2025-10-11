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
from schemas import FinalUserStoriesResult, AnalysisResult, FeedbackAnalysisResult, AnalysisAgentState, ProfessionalSpecificationDocument



def request_analysis(state: AnalysisAgentState):
    """분석된 유저 스토리를 분석하는 노드"""
    print("--- 📝 요구사항 분석 중... ---")

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
    system_prompt = """
    [역할]
    당신은 20년 경력의 시니어 요구사항 분석가이자 소프트웨어 아키텍트입니다. 
    당신의 임무는 주어진 사용자 스토리 목록을 분석하여, 개발팀이 즉시 구현 작업을 시작할 수 있을 정도로 명확하고 실행 가능한 상세 기능 명세서를 작성하는 것입니다.
    제공된 Pydantic JSON 스키마에 따라 완벽하게 구조화된 상세 요구사항 명세서를 작성하는 것입니다.

    [핵심 지침]
    이 명세서는 한 번 확정되면 다시 수정되지 않는다는 전제 하에 작성됩니다. 
    따라서 추후에 기능 추가나 해석의 여지가 생기지 않도록, 모든 항목을 최대한 정확하고 세세하게 기술해야 합니다. 
    모든 기능은 이 문서에서 확정되어야 합니다.


    [목표]
    주어진 사용자 스토리 목록을 구체적이고 검증 가능한 기능 명세서로 정제합니다.
    기능적 요구사항(Functional Requirements)과 비기능적 요구사항(Non-Functional Requirements)을 명확히 분리하고 모호함을 제거합니다.

    작성 규칙 (필드별 지침)
    user_stories 필드 (기능 요구사항):

    입력받은 각 사용자 스토리에 대해 FullySpecifiedUserStory 객체를 하나씩 생성합니다.

    [detailed_specification 필드]
    각 스토리를 구현하는 데 필요한 모든 기능적 요구사항을 여기에 상세히 기술하십시오. (데이터 모델, 유효성 검사 규칙, UI 동작 등). 이 부분이 과거의 "1. 기능 요구사항" 섹션을 대체합니다.
    [acceptance_criteria 필드]
    각 스토리가 완료되었음을 증명할 수 있는 여러 개의 시나리오(성공, 실패 등)를 AcceptanceCriteria 객체 목록으로 작성하십시오.
    [non_functional_requirements 필드]
    프로젝트 전체에 적용되어야 하는 비기능적 요구사항을 문자열 목록으로 여기에 작성하십시오.
    보안(비밀번호 암호화), 성능(응답 시간 목표), 안정성, 로깅 등 일반적으로 누락되기 쉬운 항목들을 반드시 점검하고 포함시키십시오.

    [주의]
    특정 스토리에만 해당하는 비기능적 요구사항은 해당 스토리의 detailed_specification 안에 기술하십시오.
    기존의 유저 스토리는 수정되서는 안된다. 유저 스토리는 그대로 유지하라.
    [scope_and_constraints]
    사용자 스토리와 전체 대화 내용을 바탕으로 프로젝트의 범위(Scope), 가정(Assumptions), 제약(Constraints)을 명확히 식별하십시오.
    식별된 내용을 각각의 목록 필드에 나누어 기술하십시오.
    [역할 경계]
    이 명세서는 '무엇을(What)' 만들지에만 집중합니다. '어떻게(How)' 만들지에 해당하는 특정 기술 스택이나 라이브러리 이름은 절대로 명시하지 마십시오.

    [이전 피드백 (있을 경우)]
    {feedback}

    [산출물 기준]
    - 모든 출력은 한국어로 작성합니다.
    - 출력 형식은 제공된 Pydantic 모델의 스키마를 엄격히 준수해야 합니다. 필드 이름을 임의로 변경하지 마십시오.
    """

    human_prompt = """
    [지시]
    아래 [사용자 스토리 초안 목록]을 검토하고, 당신의 역할에 맞게 정제된 결과물을 생성하십시오.

    [입력: 사용자 스토리 초안 목록]
    {user_stories}
    """

    prompt = ChatPromptTemplate([
        ("system", system_prompt),
        ("human", human_prompt)
    ])

    chain = prompt | llm.with_structured_output(ProfessionalSpecificationDocument)

    result = chain.invoke({
        "feedback": state.get("feedback", ""),
        "user_stories": state["user_stories"]
    })

    return {
        "functional_requirements": result.functional_requirements, 
    }

def feedback_analysis(state: AnalysisAgentState):
    """작성된 요구사항에 대해 평가하는 노드"""

    print("--- 📝 작성된 요구사항에 대해 평가 중... ---")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
   
    system_prompt = """
    당신은 시니어 요구사항 분석가이자 소프트웨어 아키텍트다. 유저스토리에 대해 작성된 요구사항을 평가하라.

    [평가 규칙]
    - 작성된 기능 명세서가 유저스토리를 충분히 반영했는지 평가하라.
    - 충분히 반영했다면 is_complete를 true로 설정한다.
    - 충분히 반영하지 못했다면 is_complete를 false로 설정하고 feedback에 평가를 작성한다.
    - feedback에는 기능 명세서 재성성시 고려해야할 점을 작성한다.

    [산출물 기준]
    - 모든 출력은 한국어로 작성한다.
    - Pydantic 모델의 필드 이름을 임의로 변경하지 않는다.

    """

    human_prompt = """
    [지시]
    아래 [작성된 기능 명세서]가 [유저스토리]를 충분히 반영했는지 평가하라.

    [작성된 기능 명세서]
    {final_specifications}
    """

    prompt = ChatPromptTemplate([
        ("system", system_prompt),
        ("human", human_prompt),
    ])

    chain = prompt | llm.with_structured_output(FeedbackAnalysisResult)
    result = chain.invoke({
        "final_specifications": state["final_specifications"]
    })

    return {
        "feedback": result.feedback, 
        "is_complete": result.is_complete
    }
    

def is_complete(state: AnalysisAgentState):
    return state["is_complete"]

def main(user_stories: FinalUserStoriesResult):
    load_dotenv()

    workflow = StateGraph(AnalysisAgentState)

    workflow.add_node("request_analysis", request_analysis)
    workflow.add_node("feedback_analysis", feedback_analysis)
    
    workflow.add_edge(START, "request_analysis") 
    workflow.add_edge("request_analysis", "feedback_analysis")
    workflow.add_conditional_edges("feedback_analysis", is_complete,{True: END, False: "request_analysis"})    

    app = workflow.compile()

    initial_state = {
        "final_specifications": final_specifications,
        "is_complete": False,
        "feedback": None,
    }
    
    final_state = app.invoke(initial_state)
    
    return final_state
    
