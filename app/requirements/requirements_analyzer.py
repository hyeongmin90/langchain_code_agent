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

class AnalysisResult(BaseModel):
    goal: str = Field(description="분석된 고객의 요청사항에 대한 목표")
    functional_requirements: str = Field(description="기능적 요구사항")
    non_functional_requirements: str = Field(description="비기능적 요구사항")

class FeedbackAnalysisResult(BaseModel):
    feedback: str = Field(description="작성된 요구사항에 대한 평가")
    is_complete: bool = Field(description="작성된 요구사항이 사용자의 요청사항을 충분히 반영했는지 평가하라. 충분히 반영했다면 true, 아니면 false")


class AgentState(TypedDict):
    request: str
    functional_requirements: str
    non_functional_requirements: str
    is_complete: bool
    feedback: Optional[str]
    messages: Annotated[list, operator.add]
    clarifying_questions: List[str]


def request_analysis(state: AgentState):
    """사용자의 요청사항을 분석하고, 추가 정보가 필요한지 판단하는 노드"""
    print("--- 📝 요구사항 분석 중... ---")

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
    system_prompt = """
    당신은 시니어 요구사항 분석가이자 소프트웨어 아키텍트다. 아래 규칙에 따라 대화를 분석하고 요구사항을 명확하고 실행 가능하게 정리하라.
    한번 확정된 요구사항은 다시 수정하지 않고 구현하기 때문에 추후에 기능이 추가되지 않도록 최대한 정확하고 세세하게 작성해야한다.
    모든 기능은 확정되어야 한다.

    [목표]
    - 고객의 추상적 요구를 구체적이고 검증 가능한 요구사항으로 정제한다.
    - 기능적/비기능적 요구를 명확히 분리하고, 모호성을 제거한다.
    - 정말 필요한 정보에 대해서만 추가 질문을 통해 보완한다.

    [작성 규칙]
    - 기능/비기능 요구는 번호 목록 형태로 작성한다.
    - 기능 요구는 최대한 자세하게 작성하고, 비기능적 요구는 최소한의 기준을 가지고 작성한다.
    - 모호한 표현(빠르게, 크게, 안정적 등)은 금지하고, 구체적 수치/조건/사례로 대체한다.
    - 범위(Scope), 가정(Assumptions), 제약(Constraints)이 암시되어 있으면 명시적으로 드러내고 해당 항목에 통합한다.
    - 사용자의 요청이 없을 경우에는 가장 일반적인 방법을 우선 채택하여 작성하며, 리스크를 최소화한다.
    - 보안/개인정보, 로깅/모니터링, 배포/롤백, 국제화/현지화, 접근성 등 일반적으로 누락되기 쉬운 비기능 항목을 습관적으로 점검한다.
    - 피드백이 있다면 이를 반영하여 요구사항을 수정하라.

    [산출물 기준]
    - 모든 출력은 한국어로 작성한다.
    - Pydantic 모델의 필드 이름을 임의로 변경하지 않는다.


    현재 계획된 기능: {functional_requirements}

    피드백: {feedback}
    """

    prompt = ChatPromptTemplate([
        ("system", system_prompt),
        ("human", "사용자 응답: {request}")
    ])

    chain = prompt | llm.with_structured_output(AnalysisResult)

    result = chain.invoke({
        "request": state["request"],    
        "feedback": state.get("feedback", ""), 
        "functional_requirements": state.get("functional_requirements", "아직 정의되지 않음"),
        "non_functional_requirements": state.get("non_functional_requirements", "아직 정의되지 않음"),
    })

    return {
        "goal": result.goal,
        "functional_requirements": result.functional_requirements, 
        "non_functional_requirements": result.non_functional_requirements
    }

def feedback_analysis(state: AgentState):
    """작성된 요구사항에 대해 평가하는 노드"""

    print("--- 📝 작성된 요구사항에 대해 평가 중... ---")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
   
    system_prompt = """
    당신은 시니어 요구사항 분석가이자 소프트웨어 아키텍트다. 사용자의 요청사항에 대해 작성된 요구사항을 평가하라.

    [평가 규칙]
    - 작성된 요구사항이 사용자의 요청사항을 충분히 반영했는지 평가하라.
    - 충분히 반영했다면 is_complete를 true로 설정한다.
    - 충분히 반영하지 못했다면 is_complete를 false로 설정하고 feedback에 평가를 작성한다.
    
    - 또한 사용자가 요구사항을 추후에 수정할 수 있으므로 이를 감안하여 평가하라.

    [산출물 기준]
    - 모든 출력은 한국어로 작성한다.
    - Pydantic 모델의 필드 이름을 임의로 변경하지 않는다.

    작성된 요구사항: {functional_requirements}
    작성된 비기능적 요구사항: {non_functional_requirements}
    """

    chat_history = state["messages"]

    prompt = ChatPromptTemplate([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "작성된 요구사항에 대해 평가하라."),
    ])

    chain = prompt | llm.with_structured_output(FeedbackAnalysisResult)
    result = chain.invoke({
        "chat_history": chat_history,
        "functional_requirements": state["functional_requirements"],
        "non_functional_requirements": state["non_functional_requirements"],
    })

    return {
        "feedback": result.feedback, 
        "is_complete": result.is_complete
    }
    

def is_complete(state: AgentState):
    return state["is_complete"]

def main(first_request: str):
    load_dotenv()

    workflow = StateGraph(AgentState)

    workflow.add_node("request_analysis", request_analysis)
    workflow.add_node("feedback_analysis", feedback_analysis)
    
    workflow.add_edge(START, "request_analysis") 
    workflow.add_edge("request_analysis", "feedback_analysis")
    workflow.add_conditional_edges("feedback_analysis", is_complete,{True: END, False: "request_analysis"})    

    app = workflow.compile()

    initial_state = {
        "request": first_request,
        "functional_requirements": "",
        "non_functional_requirements": "",
        "is_complete": False,
        "feedback": None,
        "messages": [HumanMessage(content=first_request)]
    }
    
    final_state = app.invoke(initial_state)

    print("\n" + "="*60)
    print("✅ 요구사항 분석 완료!")
    print("="*60)
    print(f"\n📋 목표: {final_state['request']}")
    print(f"\n📌 기능적 요구사항:\n{final_state['functional_requirements']}")
    print(f"\n📌 비기능적 요구사항:\n{final_state['non_functional_requirements']}")
    
    return final_state
    

if __name__ == "__main__":
    print("💡 요구사항을 입력해주세요:")
    first_request = input("👉 ")
    main(first_request)