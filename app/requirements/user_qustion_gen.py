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

class ClarifyingQuestionsResult(BaseModel):
    clarifying_questions: List[str] = Field(description="추가 질문 목록")
    
class UserResponseResult(BaseModel):
    ask_question: str = Field(description="사용자에게 보여줄 메시지 텍스트")
    result: str = Field(description="질문에 대한 사용자 응답 결과를 정리한 텍스트")
    is_complete: bool = Field(description="사용자의 응답이 충분한지 판단하라. 충분하면 true, 아니면 false")

class AgentState(TypedDict):
    request: str
    clarifying_questions: List[str]
    ask_question : str
    result: str
    messages: Annotated[list, operator.add]
    is_complete: bool

def generate_clarifying_questions(state: AgentState):
    """사용자의 요청사항에 대해 추가 질문을 생성하는 노드"""
    print("--- 📝 추가 질문 생성 중... ---")
    parser = JsonOutputParser(pydantic_object=ClarifyingQuestionsResult)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
    system_prompt = """
    당신은 시니어 요구사항 분석가이자 소프트웨어 아키텍트다. 명세서 작성에 앞서 반드시 확인해야 할 핵심 질문 2~4가지를 생성하세요. 
    질문은 프로젝트의 범위, 핵심 기능, 제약 조건을 명확히 하는 데 초점을 맞춰야 합니다
    
    [추가 질문 생성 규칙]
    - 사용자의 요청사항에 대해 추가 질문을 생성하라.
    - 추가 질문은 사용자의 요청사항에 대해 추가 정보가 필요한지 판단하는 질문이다.
    - 추가 질문은 최대 4가지까지 생성하라.
    - 추가 질문은 모호하지 않고 명확하게 작성하라.

    [산출물 기준]
    - 모든 출력은 한국어로 작성한다.
    - 출력 형식은 반드시 제공된 JSON 스키마 지침을 엄겹히 따른다. 필드 이름을 임의로 변경하지 않는다.
      - clarifying_questions: 추가 질문 목록.

    {format_instructions}
    """

    prompt = ChatPromptTemplate([
        ("system", system_prompt),
        ("human", "사용자 응답: {request}")
    ])
    chain = prompt | llm | parser
    result = chain.invoke({
        "format_instructions": parser.get_format_instructions(),
        "request": state["request"]
    })
    print("--------------------------------")
    print(result["clarifying_questions"])
    print("--------------------------------")

    return {
        "clarifying_questions": result["clarifying_questions"]
    }

def generate_user_request(state: AgentState):
    """사용자의 요청을 생성하는 노드"""
    print("--- 📝 사용자의 요청을 생성하는 중... ---")

    parser = JsonOutputParser(pydantic_object=UserResponseResult)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    system_prompt = """
    당신은 사용자의 프로젝트 아이디어를 현실로 만들어주는 친절하고 유능한 AI 프로젝트 어시스턴트입니다. 
    당신의 임무는 딱딱한 질문 목록을 따뜻하고 자연스러운 대화로 바꾸어, 사용자가 편안하게 자신의 생각을 이야기하도록 돕는 것입니다.
    또한 사용자의 응답이 부족하면 추가 질문을 하여 사용자의 응답을 유도하는 것이 중요합니다.

    [지시]
    [핵심 질문 목록]을 자연스럽게 소개하면서, 사용자가 답변을 입력하도록 유도하는 환영 메시지를 작성하십시오. 
    이 메시지는 사용자가 보게 될 첫인상이므로, 협력적인 분위기를 조성하는 것이 매우 중요합니다.
    대화 이력에 따라 추가 질문을 하여 사용자의 응답을 유도하는 것이 중요합니다.

    [출력 형식]
    - 다른 설명 없이, 사용자에게 보여줄 최종 환영 메시지 텍스트만 출력하십시오.

    [규칙]
    - 왜 이 질문들이 필요한지 간략히 설명하여 사용자를 안심시키십시오. (예: "더 정확한 기능 명세서를 만들기 위해...")
    - 딱딱한 질문 어조가 아닌, 부드럽고 대화하는 듯한 문체를 사용하십시오.
    - 메시지 마지막에는 사용자의 답변을 기다린다는 뉘앙스를 풍겨 대화를 유도하십시오.
    - 사용자의 응답이 부족하면 is_complete를 false로 설정하고 ask_question을 추가 질문으로 설정하십시오.
    - 사용자의 응답이 충분하면 is_complete를 true로 설정하고 result에 사용자의 응답 결과를 정리한 텍스트를 작성하십시오.
    - 핵심 질문의 내용을 반드시 사용하되 질문의 핵심 내용을 지키면서 자연스럽게 소개하십시오, 질문의 핵심이 바뀌지 않는한 수정해도 된다.
    
    [산출물 기준]
    - 모든 출력은 한국어로 작성한다.
    - 출력 형식은 반드시 제공된 JSON 스키마 지침을 엄격히 따른다. 필드 이름을 임의로 변경하지 않는다.
      - ask_question: 사용자에게 보여줄 메시지 텍스트.
      - is_complete: 사용자의 응답이 충분한지 판단하라. 충분하면 true, 아니면 false.
      - result: 질문에 대한 사용자 응답 결과를 정리한 텍스트.

    !!!출력 형식이 다르면 에러가 발생할 수 있으므로 반드시 다음의 JSON 스키마 지침을 엄격히 따라야 한다.!!!
    {format_instructions}
    """
    prompt = ChatPromptTemplate([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "핵심 질문 목록: {clarifying_questions}")
    ])

    chain = prompt | llm | parser

    result = chain.invoke({
        "format_instructions": parser.get_format_instructions(),
        "clarifying_questions": state["clarifying_questions"],
        "messages": state["messages"]
    })

    updated_messages = state["messages"] + [AIMessage(content=result["ask_question"])]
    return {"ask_question": result["ask_question"], "result": result.get("result", ""), "is_complete": result.get("is_complete", False), "messages": updated_messages}

def user_response(state: AgentState):
    """사용자의 응답을 받는 노드"""
    print("--- 📝 사용자의 응답을 받는 중... ---")
    print(state.get("ask_question", "질문이 준비되지 않았습니다. 요구사항을 설명해 주세요."))
    response = input("답변: ")
    updated_messages = state["messages"] + [HumanMessage(content=response)]
    return {"messages": updated_messages, "request": response}

def is_complete(state: AgentState):
    return state["is_complete"] == True

def main(first_request: str):
    load_dotenv()

    workflow = StateGraph(AgentState)

    workflow.add_node("generate_clarifying_questions", generate_clarifying_questions)
    workflow.add_node("generate_user_request", generate_user_request)
    workflow.add_node("user_response", user_response)

    workflow.add_edge(START, "generate_clarifying_questions")
    workflow.add_edge("generate_clarifying_questions", "generate_user_request")
    workflow.add_conditional_edges("generate_user_request", is_complete, {True: END, False: "user_response"})
    workflow.add_edge("user_response", "generate_user_request")

    app = workflow.compile()

    initial_state = {
        "request": first_request,
        "clarifying_questions": [],
        "messages": [HumanMessage(content=first_request)],
        "is_complete": False,
        "ask_question": "",
        "result": ""
    }

    final_state = app.invoke(initial_state)
    print("--------------------------------")
    print(final_state["result"])
    print("--------------------------------")

    return final_state

if __name__ == "__main__":
    first_request = input()
    main(first_request)