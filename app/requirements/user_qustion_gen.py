import os
import json
from dotenv import load_dotenv
import operator
from typing import TypedDict, Annotated, List, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END
from langchain_core.output_parsers import JsonOutputParser

class ClarifyingQuestionsResult(BaseModel):
    clarifying_questions: List[str] = Field(description="추가 질문 목록")
    
class UserResponseResult(BaseModel):
    ask_question: str = Field(description="사용자에게 보여줄 메시지 텍스트. 더 이상 질문이 없다면, 대화가 곧 완료될 것임을 알리는 메시지를 담아주세요.")
    is_complete: bool = Field(description="사용자의 응답이 충분한지 판단하라. 충분하면 true, 아니면 false")

class AgentState(TypedDict):
    request: str
    interaction_mode: str
    question_count: int
    ask_question : str
    result: str
    messages: Annotated[list, operator.add]
    is_complete: bool

def generate_next_question_or_complete(state: AgentState):
    """대화 기록을 바탕으로 다음 질문을 생성하거나, 대화를 완료할지 결정하는 노드"""
    print("--- 🤔 다음 행동 결정 중... ---")

    state['question_count'] += 1
    max_questions = 3 if state['interaction_mode'] == '빠른' else 7

    if state['question_count'] > max_questions:
        print(f"--- 💬 최대 질문 개수({max_questions}개)에 도달하여 대화를 종료합니다. ---")
        return {
            "is_complete": True,
            "ask_question": "충분한 정보가 수집된 것 같습니다. 이제 최종 요구사항 명세서를 정리하겠습니다."
        }

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
    
    system_prompt = """
    당신은 사용자의 프로젝트 아이디어를 구체화하는 데 도움을 주는 뛰어난 시니어 요구사항 분석가입니다.
    당신의 목표는 사용자와의 대화를 통해 프로젝트의 요구사항을 명확히 하고, 이를 바탕으로 개발팀이 이해할 수 있는 상세 명세서를 작성하는 것입니다.

    [상황]
    지금까지 사용자와 나눈 대화 기록이 주어집니다. 이 대화 기록을 면밀히 분석하여 다음 행동을 결정해야 합니다.
    사용자는 '{interaction_mode}' 모드를 선택했습니다. 이 모드에 맞춰 당신의 질문 스타일과 깊이를 조절해야 합니다.
    현재 질문은 {question_count}번째 질문이며, 최대 {max_questions}개까지 질문할 수 있습니다.

    [작업]
    다음 두 가지 행동 중 하나를 선택하고, 그에 맞는 결과(JSON 형식)를 출력해야 합니다.

    1. 추가 질문하기 (is_complete: false):
       - 아직 요구사항이 불분명하거나 더 구체화할 필요가 있다고 판단될 때 선택합니다.
       - 지금까지의 대화 내용에서 가장 중요하고 궁금한 **단 하나의 질문**을 생성합니다.
       - **(중요)** 사용자의 답변 부담을 줄이기 위해, 가능한 경우 **적절한 기본값을 포함한 제안 형태**로 질문을 구성하세요.
         - 예시 (나쁜 질문): "어떤 데이터베이스를 사용하시겠어요?"
         - 예시 (좋은 질문): "데이터베이스는 표준적인 PostgreSQL로 구성하는 것을 제안하는데, 괜찮으신가요? 다른 선호하는 데이터베이스가 있다면 알려주세요."
       - '{interaction_mode}' 모드에 따라 질문의 깊이를 조절합니다.
         - '빠른' 모드: 핵심 기능과 범위에 집중된 최소한의 질문을 합니다.
         - '상세' 모드: 기술 스택, 비기능적 요구사항, 엣지 케이스 등 더 깊이 있는 질문을 할 수 있습니다.

    2. 대화 완료하기 (is_complete: true):
       - 사용자가 "그만", "완료", "충분해요" 등 대화 종료를 의미하는 발언을 했을 경우, 즉시 이 행동을 선택해야 합니다.
       - 사용자의 요구사항이 충분히 명확해져서 더 이상 질문할 필요가 없다고 판단될 때 선택합니다.
       - '빠른' 모드에서는 몇 가지 핵심 사항만 확인되면 빠르게 완료할 수 있습니다.
       - '상세' 모드에서는 모든 측면이 충분히 다루어졌는지 신중하게 판단 후 완료합니다.
       - is_complete를 true로 설정하고 대화를 종료합니다.

    [규칙]
    - 반드시 지정된 Pydantic 모델 형식으로만 출력해야 합니다.
    - 모든 출력은 한국어로 작성합니다.
    - 대화 기록을 바탕으로 맥락에 맞는 질문을 생성해야 합니다.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ])

    chain = prompt | llm.with_structured_output(UserResponseResult)

    result = chain.invoke({
        "messages": state["messages"],
        "interaction_mode": state["interaction_mode"],
        "question_count": state["question_count"],
        "max_questions": max_questions
    })

    print(f"---\n 🗣️ 질문 {state['question_count']}: {result.ask_question} (완료: {result.is_complete}) \n---\n")

    return {
        "ask_question": result.ask_question, 
        "is_complete": result.is_complete or False, 
        "question_count": state["question_count"],
        "messages": [AIMessage(content=result.ask_question)]
    }


def user_response(state: AgentState):
    """사용자의 응답을 받는 노드"""
    print(state.get("ask_question", "질문이 준비되지 않았습니다. 요구사항을 설명해 주세요."))
    response = input("답변: ")
    return {"messages": [HumanMessage(content=response)]}  

def final_result_generation(state: AgentState):
    """최종 결과물을 생성하는 노드"""
    print("--- 📝 최종 결과물을 생성하는 중... ---")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
    system_prompt = """
    당신은 지금까지의 대화내역을 바탕으로 최종 결과물을 생성하는 시니어 요구사항 분석가이자 소프트웨어 아키텍트다. 
    최종 결과물은 프로젝트의 범위, 핵심 기능, 제약 조건을 명확히 하는 데 초점을 맞춰야 합니다

    [최종 결과물 생성 규칙]
    - 최종 결과물은 프로젝트의 범위, 핵심 기능, 제약 조건을 명확히 하는 데 초점을 맞춰야 합니다
    - 최종 결과물은 1. 주제 2. 기능 3. 범위, 가정, 제약 4. 기타 정보로 구성되어야 합니다.
    - 최종 결과물은 사용자와의 대화 이력을 바탕으로 언급되거나 암시된 모든 기능에 대해 작성하라.
    - 최대한 자세하게 작성하라.
    - 최종 결과물은 모든 출력은 한국어로 작성한다.
    - 최종 결과물 외의 너의 생각이나 대화 이력등의 모든 기타 정보는 포함하지 않는다.
    """
    prompt = ChatPromptTemplate([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "대화 내역을 바탕으로 최종 결과물을 생성하라.")
    ])
    chain = prompt | llm
    result = chain.invoke({
        "messages": state["messages"]
    })

    print("최종 결과물 -----------------")
    print(result.content)
    print("--------------------------------")

    return {"result": result.content}

def is_complete(state: AgentState):
    return state["is_complete"] == True

def main(first_request: str, interaction_mode: str):
    load_dotenv()

    workflow = StateGraph(AgentState)

    workflow.add_node("generate_next_question", generate_next_question_or_complete)
    workflow.add_node("user_response", user_response)
    workflow.add_node("final_result_generation", final_result_generation)

    workflow.add_edge(START, "generate_next_question")
    workflow.add_conditional_edges("generate_next_question", is_complete, {True: "final_result_generation", False: "user_response"})
    workflow.add_edge("user_response", "generate_next_question")
    workflow.add_edge("final_result_generation", END)

    app = workflow.compile()

    initial_state = {
        "request": first_request,
        "interaction_mode": interaction_mode,
        "messages": [HumanMessage(content=first_request)],
        "is_complete": False,
        "ask_question": "",
        "result": "",
        "question_count": 0
    }

    final_state = app.invoke(initial_state)

    return final_state["result"]

if __name__ == "__main__":
    print("안녕하세요! 어떤 프로젝트를 만들고 싶으신가요? 자세히 알려주실수록 좋습니다.")
    first_request = input("요구사항: ")
    
    mode = ""
    while mode not in ["1", "2"]:
        print("\n어떤 모드로 진행할까요?")
        print("1. 빠른 모드 (핵심 질문 위주로 빠르게 진행)")
        print("2. 상세 모드 (기술 스택, 제약 조건 등 상세하게 진행)")
        mode = input("선택 (1 또는 2): ")

    interaction_mode = "빠른" if mode == "1" else "상세"
    
    print(main(first_request, interaction_mode))