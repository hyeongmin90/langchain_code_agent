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

class ClarifyingQuestionsResult(BaseModel):
    clarifying_questions: List[str] = Field(description="추가 질문 목록")

class UserResponseResult(BaseModel):
    request: str = Field(description="사용자에게 보여줄 메시지 텍스트")

class AgentState(TypedDict):
    request: str
    functional_requirements: str
    non_functional_requirements: str
    is_complete: bool
    feedback: Optional[str]
    messages: Annotated[list, operator.add]
    clarifying_questions: List[str]

def generate_clarifying_questions(state: AgentState):
    """사용자의 요청사항에 대해 추가 질문을 생성하는 노드"""
    print("--- 📝 추가 질문 생성 중... ---")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
    system_prompt = """
    당신은 시니어 요구사항 분석가이자 소프트웨어 아키텍트다. 명세서 작성에 앞서 반드시 확인해야 할 핵심 질문 3가지를 생성하세요. 
    질문은 프로젝트의 범위, 핵심 기능, 제약 조건을 명확히 하는 데 초점을 맞춰야 합니다
    
    [추가 질문 생성 규칙]
    - 사용자의 요청사항에 대해 추가 질문을 생성하라.
    - 추가 질문은 사용자의 요청사항에 대해 추가 정보가 필요한지 판단하는 질문이다.
    - 추가 질문은 최대 3가지까지 생성하라.
    - 추가 질문은 모호하지 않고 명확하게 작성하라.

    [산출물 기준]
    - 모든 출력은 한국어로 작성한다.
    - 출력 형식은 반드시 제공된 JSON 스키마 지침을 엄겹히 따른다. 필드 이름을 임의로 변경하지 않는다.
      - clarifying_questions: 추가 질문 목록.
    """

    prompt = ChatPromptTemplate([
        ("system", system_prompt),
        ("human", "사용자 응답: {request}")
    ])
    chain = prompt | llm.with_structured_output(ClarifyingQuestionsResult)
    result = chain.invoke({
        "request": state["request"]
    })

    return {
        "clarifying_questions": result.clarifying_questions
    }

def user_response(state: AgentState):
    """사용자의 응답을 받는 노드"""
    print("--- 📝 사용자의 응답을 받는 중... ---")
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    system_prompt = """
    당신은 사용자의 프로젝트 아이디어를 현실로 만들어주는 친절하고 유능한 AI 프로젝트 어시스턴트입니다. 
    당신의 임무는 딱딱한 질문 목록을 따뜻하고 자연스러운 대화로 바꾸어, 사용자가 편안하게 자신의 생각을 이야기하도록 돕는 것입니다.

    [지시]
    아래 [핵심 질문 목록]을 자연스럽게 소개하면서, 사용자가 답변을 입력하도록 유도하는 환영 메시지를 작성하십시오. 
    이 메시지는 사용자가 보게 될 첫인상이므로, 협력적인 분위기를 조성하는 것이 매우 중요합니다.

    [핵심 질문 목록]
    {clarifying_questions}

    [출력 형식]
    - 다른 설명 없이, 사용자에게 보여줄 최종 환영 메시지 텍스트만 출력하십시오.

    [규칙]
    - 왜 이 질문들이 필요한지 간략히 설명하여 사용자를 안심시키십시오. (예: "더 정확한 기능 명세서를 만들기 위해...")
    - 딱딱한 질문 어조가 아닌, 부드럽고 대화하는 듯한 문체를 사용하십시오.
    - 메시지 마지막에는 사용자의 답변을 기다린다는 뉘앙스를 풍겨 대화를 유도하십시오.

    [산출물 기준]
    - 모든 출력은 한국어로 작성한다.
    - 출력 형식은 반드시 제공된 JSON 스키마 지침을 엄격히 따른다. 필드 이름을 임의로 변경하지 않는다.
      - request: 사용자에게 보여줄 메시지 텍스트.
    """
    prompt = ChatPromptTemplate([
        ("system", system_prompt),
        ("human", "사용자 응답: {request}"),
    ])
    chain = prompt | llm.with_structured_output(UserResponseResult)

    result = chain.invoke({
        "request": state["request"],
        "clarifying_questions": state["clarifying_questions"]
    })

    return {"request": result.request, "messages": state["messages"]}

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
    - 기능/비기능 요구는 번호 목록 형태로 작성하되, 각 항목은 측정 가능하거나 검증 가능한 수치/조건을 포함한다. 예: "응답 시간 p95 ≤ 300ms", "동시 접속 5,000 사용자 지원".
    - 모호한 표현(빠르게, 크게, 안정적 등)은 금지하고, 구체적 수치/조건/사례로 대체한다.
    - 범위(Scope), 가정(Assumptions), 제약(Constraints)이 암시되어 있으면 명시적으로 드러내고 해당 항목에 통합한다.
    - 사용자의 요청이 없을 경우에는 가장 일반적인 방법을 우선 채택하여 작성하며, 리스크를 최소화한다.
    - 보안/개인정보, 로깅/모니터링, 배포/롤백, 국제화/현지화, 접근성 등 일반적으로 누락되기 쉬운 비기능 항목을 습관적으로 점검한다.
    - 대화 이력의 핵심 요구와 비핵심 잡음을 구분하여, 핵심만 반영한다.
    - 피드백이 있다면 이를 반영하여 요구사항을 수정하라.
    - 기능에 대한 요구사항은 최대한 자세하게 작성하고, 비기능적 요구사항은 최대한 간결하게 작성한다.

    [산출물 기준]
    - 모든 출력은 한국어로 작성한다.
    - 출력 형식은 반드시 제공된 JSON 스키마 지침을 엄격히 따른다. 필드 이름을 임의로 변경하지 않는다.


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
    - 출력 형식은 반드시 제공된 JSON 스키마 지침을 엄격히 따른다. 필드 이름을 임의로 변경하지 않는다.
      - feedback: 작성된 요구사항에 대한 평가.
      - is_complete: 정보가 충분해 구현 계획 수립이 가능하면 true, 아니면 false.

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
    
    return final_state
    

if __name__ == "__main__":
    print("💡 요구사항을 입력해주세요:")
    first_request = input("👉 ")
    main(first_request)