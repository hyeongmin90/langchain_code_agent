import os
import operator
from typing import TypedDict, Annotated

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

# --- 1. 에이전트가 사용할 도구(Tools) 정의 ---

@tool
def read_file(file_path: str) -> str:
    """지정된 경로의 파일을 읽어 그 내용을 반환합니다."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return f"오류: {file_path} 파일을 찾을 수 없습니다."

@tool
def write_file(file_path: str, code: str) -> str:
    """지정된 경로의 파일에 새로운 코드를 덮어씁니다."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(code)
        return f"성공: {file_path} 파일에 성공적으로 코드를 작성했습니다."
    except Exception as e:
        return f"오류: 파일 작성 중 문제가 발생했습니다 - {e}"

# --- 2. 에이전트의 상태(State) 정의 ---
# 그래프의 각 노드(단계)가 공유하는 데이터의 형태를 정의합니다.

class AgentState(TypedDict):
    request: str  # 사용자의 최초 요청
    file_path: str # 수정할 파일 경로
    original_code: str # 원본 코드 내용
    plan: str # LLM이 생성한 작업 계획
    generated_code: str # LLM이 생성한 새로운 코드
    feedback: str # 파일 쓰기 결과 또는 오류 메시지
    # 대화 기록을 위해 메시지를 누적합니다.
    messages: Annotated[list, operator.add]

# --- 3. 그래프의 각 노드(Node) 정의 ---
# 각 노드는 에이전트가 수행하는 하나의 작업 단위를 의미합니다.

def plan_node(state: AgentState):
    """사용자의 요청을 바탕으로 코드 수정 계획을 수립하는 노드"""
    print("--- 📝 계획 수립 중... ---")
    
    # --- 여기에 LLM 호출 로직 추가 ---
    # 예: prompt = f"{state['request']} 요청을 처리하기 위한 계획을 세워줘."
    #     plan = llm.invoke(prompt)
    plan = "1. 'sample_code.py' 파일을 읽는다. 2. 파일 끝에 'add' 함수를 추가한다."
    
    return {"plan": plan}

def read_code_node(state: AgentState):
    """계획에 따라 파일을 읽는 노드"""
    print(f"--- 📖 '{state['file_path']}' 파일 읽는 중... ---")
    original_code = read_file.invoke(state['file_path'])
    return {"original_code": original_code}

def generate_code_node(state: AgentState):
    """원본 코드와 계획을 바탕으로 새 코드를 생성하는 노드"""
    print("--- 💻 코드 생성 중... ---")
    
    # --- 여기에 LLM 호출 로직 추가 ---
    # 예: prompt = f"""
    #     기존 코드: {state['original_code']}
    #     계획: {state['plan']}
    #     요청: {state['request']}
    #     위 정보를 바탕으로 수정된 전체 코드를 생성해줘.
    # """
    #     generated_code = llm.invoke(prompt)
    new_function = "\n\ndef add(a, b):\n    return a + b\n"
    generated_code = state['original_code'] + new_function
    
    return {"generated_code": generated_code}

def write_code_node(state: AgentState):
    """생성된 코드를 파일에 다시 쓰는 노드"""
    print(f"--- 💾 '{state['file_path']}' 파일 저장 중... ---")
    feedback = write_file.invoke({"file_path": state['file_path'], "code": state['generated_code']})
    return {"feedback": feedback}

# --- 4. 그래프(Graph) 구성 및 실행 ---

def main():
    # .env 파일에서 API 키 로드
    load_dotenv()

    # 그래프 생성
    workflow = StateGraph(AgentState)

    # 노드 추가
    workflow.add_node("plan", plan_node)
    workflow.add_node("read_code", read_code_node)
    workflow.add_node("generate_code", generate_code_node)
    workflow.add_node("write_code", write_code_node)

    # 엣지(Edge) 연결: 각 노드가 어떤 순서로 실행될지 정의
    workflow.add_edge(START, "plan") # 시작하면 plan 노드부터
    workflow.add_edge("plan", "read_code")
    workflow.add_edge("read_code", "generate_code")
    workflow.add_edge("generate_code", "write_code")
    workflow.add_edge("write_code", END) # write_code가 끝나면 종료

    # 그래프 컴파일
    app = workflow.compile()

    # --- 에이전트 실행 ---
    # 1. 에이전트가 수정할 샘플 파일 생성
    with open("sample_code.py", "w", encoding="utf-8") as f:
        f.write("# This is a sample Python file.\n\n")
        f.write("def hello():\n")
        f.write("    print('Hello, World!')\n")

    # 2. 실행할 작업 정의
    inputs = {
        "request": "두 숫자를 더하는 'add' 함수를 추가해줘.",
        "file_path": "sample_code.py",
        "messages": []
    }

    # 3. 그래프 실행 및 결과 확인
    final_state = app.invoke(inputs)

    print("\n--- ✅ 작업 완료 ---")
    print("최종 피드백:", final_state['feedback'])
    print(f"\n--- '{final_state['file_path']}' 최종 내용 ---")
    print(read_file.invoke(final_state['file_path']))

if __name__ == "__main__":
    main()