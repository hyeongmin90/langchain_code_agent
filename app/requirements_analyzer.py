import os
import json
from dotenv import load_dotenv
import operator
from typing import TypedDict, Annotated, List, Optional

# Google Cloud 관련 경고 메시지 억제
os.environ['GRPC_VERBOSITY'] = 'ERROR'

from pydantic import BaseModel, Field
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, START, END

class AnalysisResult(BaseModel):
    goal: str = Field(description="분석된 고객의 요청사항에 대한 목표")
    functional_requirements: str = Field(description="기능적 요구사항")
    non_functional_requirements: str = Field(description="비 기능적 요구사항")
    is_complete: bool = Field(description="요구사항 분석을 완료하기에 정보가 충분하면 true(확정), 부족하면 false(추가 정보 요청)")
    question_to_user: Optional[str] = Field(description="정보가 부족할 경우, 사용자에게 물어볼 질문")

class UserQuestion(BaseModel):
    question_to_user: str = Field(description="사용자에게 물어볼 질문")
    is_complete: bool = Field(description="요구사항 분석을 완료하기에 정보가 충분하면 true, 부족하면 false")

class AgentState(TypedDict):
    request: str
    functional_requirements: str
    non_functional_requirements: str
    is_complete: bool
    question_to_user: Optional[str]
    messages: Annotated[list, operator.add]

def request_analysis(state: AgentState):
    """사용자의 요청사항을 분석하고, 추가 정보가 필요한지 판단하는 노드"""
    print("--- 📝 요구사항 분석 중... ---")


    parser = JsonOutputParser(pydantic_object=AnalysisResult)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
    system_prompt = """
    당신은 시니어 요구사항 분석가이자 소프트웨어 아키텍트다. 아래 규칙에 따라 대화를 분석하고 요구사항을 명확하고 실행 가능하게 정리하라.
    한번 확정된 요구사항은 다시 수정하지 않고 구현하기 때문에 추후에 기능이 추가되지 않도록 최대한 정확하고 세세하게 작성해야한다.
    먼저 임의로 필요한 모든 기능에 대해 요구사항을 작성하고, 사용자의 응답을 받아 요구사항을 수정한다.
    
    [목표]
    - 고객의 추상적 요구를 구체적이고 검증 가능한 요구사항으로 정제한다.
    - 기능적/비기능적 요구를 명확히 분리하고, 모호성을 제거한다.
    - 정보가 부족하면 핵심 공백을 식별하고 추가 질문을 통해 보완한다.

    [작성 규칙]
    - 기능/비기능 요구는 번호 목록 형태로 작성하되, 각 항목은 측정 가능하거나 검증 가능한 수치/조건을 포함한다. 예: "응답 시간 p95 ≤ 300ms", "동시 접속 5,000 사용자 지원".
    - 모호한 표현(빠르게, 크게, 안정적 등)은 금지하고, 구체적 수치/조건/사례로 대체한다.
    - 범위(Scope), 가정(Assumptions), 제약(Constraints)이 암시되어 있으면 명시적으로 드러내고 해당 항목에 통합한다.
    - 리스크와 불확실성이 보이면 해당 항목을 질문으로 전환하여 고객 확인이 필요함을 표시한다.
    - 보안/개인정보, 로깅/모니터링, 배포/롤백, 국제화/현지화, 접근성 등 일반적으로 누락되기 쉬운 비기능 항목을 습관적으로 점검한다.
    - 대화 이력의 핵심 요구와 비핵심 잡음을 구분하여, 핵심만 반영한다.

    [대화 활용 방법]
    - 먼저 대화(conversation_history)의 목적/배경/제약을 1-2문장으로 함축 요약한 뒤, 요구사항을 정리한다.
    - 불충분한 부분이 있으면 question_to_user에 최소 3개 이상 구체적이고 답하기 쉬운 질문을 제시한다. 단, 정보가 충분하면 질문 개수를 줄일 수 있다.

    [산출물 기준]
    - 모든 출력은 한국어로 작성한다.
    - 출력 형식은 반드시 제공된 JSON 스키마 지침을 엄격히 따른다. 필드 이름을 임의로 변경하지 않는다.


    현재 계획된 기능: {functional_requirements}
    현재 계획된 비기능적 요구: {non_functional_requirements}


    {format_instructions}
    """

    user_request = """
    요청된 요구사항: {question_to_user}
    그에 따른 사용자 응답: {request}
    """

    prompt = ChatPromptTemplate([
        ("system", system_prompt),
        ("human", user_request)
    ])

    chain = prompt | llm | parser

    result = chain.invoke({
        "format_instructions": parser.get_format_instructions(),
        "request": state["request"],    
        "question_to_user": state["question_to_user"],
        "functional_requirements": state.get("functional_requirements", "아직 정의되지 않음"),
        "non_functional_requirements": state.get("non_functional_requirements", "아직 정의되지 않음"),
    })

    return {
        "request": result["goal"], 
        "functional_requirements": result["functional_requirements"], 
        "non_functional_requirements": result["non_functional_requirements"], 
        "is_complete": result["is_complete"], 
        "question_to_user": result["question_to_user"]
    }

def ask_to_user(state: AgentState):
    """사용자에게 추가 정보를 요청하는 노드"""

    print("--- 📝 사용자에게 추가 정보를 요청 중... ---")
    parser = JsonOutputParser(pydantic_object=UserQuestion)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
   

    system_prompt = """
    당신은 시니어 요구사항 분석가이자 소프트웨어 아키텍트다. 개발팀이 요청한 정보와 대화 이력을 바탕으로, 사용자에게 추가로 확인해야 할 질문을 생성한다.
    
    
    [지침]
    - 대화 이력(conversation_history)의 맥락을 반영하되, 이미 물어본 질문은 반복하지 않는다.
    - 질문은 구체적이고 답하기 쉽게 작성하며, 선택지나 예시를 덧붙여 응답 품질을 높인다.
    - 한 번에 너무 많은 질문을 주지 말고 최대 3-5개로 우선순위를 반영한다.
    - 각 질문은 단일 이슈만 다루도록 분리한다.

    [산출물 기준]
    - 모든 출력은 한국어로 작성한다.
    - 출력 형식은 반드시 제공된 JSON 스키마 지침을 엄격히 따른다. 필드 이름을 임의로 변경하지 않는다.
      - question_to_user: 사용자에게 보낼 질문(번호 목록). 중복/모호 금지.
      - is_complete: 정보가 충분해 구현 계획 수립이 가능하면 true, 아니면 false.

  
    {format_instructions}
    """

    chat_history = state["messages"]

    prompt = ChatPromptTemplate([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "추가 정보가 필요합니다. 사용자에게 질문을 생성해주세요."),
    ])

    chain = prompt | llm | parser
    result = chain.invoke({
        "format_instructions": parser.get_format_instructions(),
        "chat_history": chat_history
    })

    return {
        "messages": [AIMessage(content=result["question_to_user"])], 
        "question_to_user": result["question_to_user"], 
        "is_complete": result["is_complete"]
    }
    

def user_response(state: AgentState):
    """사용자의 응답을 받는 노드"""

    print("--- 📝 사용자의 응답을 받는 중... ---")
    print(f"\n❓ {state['question_to_user']}")
    response = input("👤 답변: ")

    return {"request": response, "messages": [HumanMessage(content=response)]}

def is_complete(state: AgentState):
    return state["is_complete"]

def main(first_request: str):
   
    load_dotenv()

    # 그래프 생성
    workflow = StateGraph(AgentState)

    # 노드 추가
    workflow.add_node("request_analysis", request_analysis)
    workflow.add_node("ask_to_user", ask_to_user)
    workflow.add_node("user_response", user_response)

    workflow.add_edge(START, "request_analysis") 
    workflow.add_conditional_edges("request_analysis", is_complete,{True: END, False: "ask_to_user"})    
    workflow.add_edge("ask_to_user", "user_response")
    workflow.add_edge("user_response", "request_analysis")

    app = workflow.compile()

    initial_state = {
        "request": first_request,
        "functional_requirements": "",
        "non_functional_requirements": "",
        "is_complete": False,
        "question_to_user": None,
        "messages": [HumanMessage(content=first_request)]
    }
    
    final_state = app.invoke(initial_state)

    print("\n" + "="*60)
    print("✅ 요구사항 분석 완료!")
    print("="*60)
    print(f"\n📋 목표: {final_state['request']}")
    print(f"\n📌 기능적 요구사항:\n{final_state['functional_requirements']}")
    print(f"\n⚙️ 비기능적 요구사항:\n{final_state['non_functional_requirements']}")
    
    return final_state
    

if __name__ == "__main__":
    print("💡 요구사항을 입력해주세요:")
    first_request = input("👉 ")
    main(first_request)