"""
Agentic Coder 워크플로우 및 오케스트라 에이전트

워크플로우:
START → Orchestrator → Specification Writer → Orchestrator
                     → Code Generator → Orchestrator
                     → Static Reviewer → Orchestrator
                     → END
"""

from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from .schemas import AgenticCoderState
from .agents import (
    requirement_analyst_agent,
    code_file_generator_agent,
    skeleton_code_generator_agent,
    file_writer_node,
    setup_project,
)

# ============================================
# 워크플로우 생성
# ============================================

def create_agentic_coder_workflow():

    workflow = StateGraph(AgenticCoderState)
    
    # 노드 추가
    workflow.add_node("requirement_analyst", requirement_analyst_agent)
    workflow.add_node("skeleton_code_generator", skeleton_code_generator_agent)
    workflow.add_node("code_file_generator", code_file_generator_agent)
    workflow.add_node("file_writer", file_writer_node)
    workflow.add_node("setup_project", setup_project)
    # 시작: START → Orchestrator (첫 분석)
    workflow.add_edge(START, "requirement_analyst")
    workflow.add_edge("requirement_analyst", "setup_project")
    workflow.add_edge("setup_project", "skeleton_code_generator")
    workflow.add_edge("skeleton_code_generator", "code_file_generator")
    workflow.add_edge("code_file_generator", "file_writer")
    workflow.add_edge("file_writer", END)

    return workflow.compile()


def generate_java_spring_boot_project(user_request: str,):
    """
    사용자 요청을 바탕으로 Java Spring Boot 프로젝트를 생성합니다.
    """
    load_dotenv()
    app = create_agentic_coder_workflow()
    initial_state = {
        "orchestrator_request": user_request,
    }
   
    final_state = app.invoke(
        initial_state,
        config={"recursion_limit": 150}  # 파일 50개까지 처리 가능
    )

    return final_state


# ============================================
# 메인 실행 (테스트용)
# ============================================

if __name__ == "__main__":
    # 테스트 실행
    user_request = """
    간단한 Todo 관리 API를 만들어줘.
    
    필요한 기능:
    - Todo 생성, 조회, 수정, 삭제
    - 제목, 내용, 완료 여부, 우선순위
    - 간단한 인증 (사용자별 Todo 관리)
    """
    
    result = generate_java_spring_boot_project(user_request)
    
    for _, file in result.get("generated_files", {}).items():
        print(f"📄 {file['file_path']}")

    total_tokens = 0
    for token_usage in result.get("token_usage_list", []):
        total_tokens += token_usage.total_tokens
    print(f"📊 총 토큰 사용량: {total_tokens}")
