import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
import uuid
from pathlib import Path

class CodeResult(BaseModel):
    code_content: List[str] = Field(description="코드 내용 한줄 단위")

class CodeGeneratorState(TypedDict):
    request: str
    analysis: str
    planning: str
    code_file_path: str

def request_analysis(state: CodeGeneratorState):
    print("--- 📝 코드 생성 Tool: 요청 분석 시작 ---")

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
    system_prompt = """
    당신은 JAVA Spring Boot 전문 개발자이다. 주어진 요구사항을 분석하여라.
    누락되는 요구사항이 있어선 안된다.
    또한 분석된 요구사항이외의 내용이나 설명은 작성하지 말고, 분석된 요구사항만 작성하라.
    요구사항은 비정형 문서로 출력하라.
    """
    prompt = ChatPromptTemplate([
        ("system", system_prompt),
        ("human", "사용자 요청사항: {request}")
    ])
    chain = prompt | llm

    result = chain.invoke({
        "request": state["request"]
    })

    print("분석된 요구사항 -----------------")
    print(result.content)
    print("--------------------------------")

    return {
        "analysis": result.content
    }   


def generate_planning(state: CodeGeneratorState):
    print("--- 📝 코드 생성 Tool: 계획 생성 시작 ---")

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
    system_prompt = """
    당신은 JAVA Spring Boot 전문 개발자이다. 분석된 요구사항을 바탕으로 개발 계획을 작성하여라.
    또한 개발 계획이외의 내용이나 설명은 작성하지 말고, 개발 계획만 작성하라.
    """
    prompt = ChatPromptTemplate([
        ("system", system_prompt),
        ("human", "분석된 요구사항: {analysis}")
    ])
    chain = prompt | llm
    result = chain.invoke({
        "analysis": state["analysis"]
    })

    print("계획된 개발 계획 -----------------")
    print(result.content)
    print("--------------------------------")

    return {
        "planning": result.content
    }


def generate_code(state: CodeGeneratorState):
    print("--- 📝 코드 생성 Tool: 코드 생성 시작 ---")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
    system_prompt = """
    당신은 JAVA Spring Boot 전문 개발자이다. 계획된 개발 계획을 바탕으로 코드를 생성하여라.
    또한 코드 이외의 내용이나 설명은 작성하지 말고, 코드만 작성하라.
    """
    prompt = ChatPromptTemplate([
        ("system", system_prompt),
        ("human", "계획된 개발 계획: {planning}")
    ])
    chain = prompt | llm.with_structured_output(CodeResult)

    result = chain.invoke({
        "planning": state["planning"]
    })

    print("생성된 코드 -----------------")
    for code in result.code_content:
        print(code)
    print("--------------------------------")
    
    try:
        file_uuid = str(uuid.uuid4())

        file_root_path = Path(os.path.dirname(__file__)) / "code_files"
        file_path = file_root_path / f"{file_uuid}.java"

        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            for code_content in result.code_content:
                f.write(code_content)
                f.write("\n")

        print(f"--- 📝 코드 생성 Tool: 코드 생성 완료 ---")
        print(f"코드 파일 경로: {file_path}")

        return {
            "code_file_path": file_path
        }
    except Exception as e:
        return {
            "code_file_path": f"코드 파일을 생성하는 중 오류가 발생했습니다: {e}"
        }

@tool
def code_generator(request: str) -> str:
    """
    사용자 요청을 받아 JAVA 코드를 생성합니다.
    요청사항에는 코드 생성에 필요한 모든 정보가 포함되어야 합니다.
    
    Args:
        request: 사용자 요청사항
        
    Returns:
        str: 생성된 코드 파일 경로
    """

    print("📝 코드 생성 Tool 실행")
    print(request)
    print("--------------------------------")
    return run_code_generator(request)

def run_code_generator(request: str):
    load_dotenv()
    workflow = StateGraph(CodeGeneratorState)
    workflow.add_node("request_analysis", request_analysis)
    workflow.add_node("generate_planning", generate_planning)
    workflow.add_node("generate_code", generate_code)

    workflow.add_edge(START, "request_analysis")
    workflow.add_edge("request_analysis", "generate_planning")
    workflow.add_edge("generate_planning", "generate_code")
    workflow.add_edge("generate_code", END)

    app = workflow.compile()

    initial_state = {
        "request": request
    }

    final_state = app.invoke(initial_state)

    result = f"""
    코드 생성 완료
    코드 파일 경로: {final_state["code_file_path"]}
    """

    return result