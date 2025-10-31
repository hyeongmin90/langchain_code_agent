"""
4개 에이전트를 통합한 멀티 에이전트 워크플로우
"""

from langgraph.graph import StateGraph, START, END
from .schemas import MultiAgentState
import uuid
from .agents import (
    setup_project,
    analyst_agent,
    planner_agent,
    coder_agent,
    verifier_agent
)

def should_continue(state: MultiAgentState) -> str:
    """
    현재 상태에 따라 다음 노드를 결정합니다.
    """
    status = state["current_status"]
    
    if status == "analyzing":
        return "analyst"
    elif status == "planning":
        return "planner"
    elif status == "coding":
        return "coder"
    elif status == "verifying":
        return "verifier"
    elif status == "completed":
        return END
    elif status == "failed":
        return END
    else:
        return END

def create_workflow():
    """
    멀티 에이전트 워크플로우를 생성합니다.
    
    플로우:
    START → Analyst → Planner → Coder → Verifier
              ↑                           ↓
              └───────← (재시도) ─────────┘
              ↑
              └─────← (다음 Epic) ────────┘
    """
    
    workflow = StateGraph(MultiAgentState)
    
    workflow.add_node("setup_project", setup_project)
    workflow.add_node("analyst", analyst_agent)
    workflow.add_node("planner", planner_agent)
    workflow.add_node("coder", coder_agent)
    workflow.add_node("verifier", verifier_agent)
    
    workflow.add_edge(START, "setup_project")
    # workflow.add_edge("setup_project", END)
    workflow.add_edge("setup_project", "analyst")
    workflow.add_edge("analyst", "planner")
    workflow.add_edge("planner", "coder")
    workflow.add_edge("coder", "verifier")

    workflow.add_conditional_edges(
        "verifier",
        should_continue,
        {
            "planner": "planner",  # 다음 에픽 또는 재시도
            "coder": "coder",      # 코드 수정 (재시도)
            END: END               # 완료
        }
    )
    
    return workflow.compile()

def run_multi_agent_system(user_request: str):
    """
    멀티 에이전트 시스템을 실행합니다.
    
    Args:
        user_request: 사용자 요청 (예: "회원가입, 로그인, 게시판 기능이 있는 블로그 MVP")
    
    Returns:
        최종 상태
    """
    
    print("\n" + "="*80)
    print("🚀 멀티 에이전트 시스템 시작")
    print("="*80)
    print(f"사용자 요청: {user_request}\n")
    
    # 워크플로우 생성
    app = create_workflow()
    
    # 초기 상태
    initial_state = {
        "project_uuid": None,
        "user_request": user_request,
        "analyzed_user_request": None,
        "current_status": "analyzing",
        "current_epic_index": 0,
        "completed_epics": [],
        "retry_count": 0,
        "max_retries": 3,
        "all_generated_files": [],
        "token_usage_list": []
    }
    
    # 실행
    final_state = app.invoke(initial_state)
    
    # 결과 출력
    print("\n" + "="*80)
    print("🎉 멀티 에이전트 시스템 완료")
    print("="*80)
    
    if final_state.get("final_message"):
        print(f"\n{final_state['final_message']}")
    
    # 완료된 에픽 출력
    if final_state.get("epic_list"):
        print(f"\n완료된 에픽 ({len(final_state.get('completed_epics', []))}개):")
        for epic_id in final_state.get("completed_epics", []):
            epic = next((e for e in final_state["epic_list"].epics if e.id == epic_id), None)
            if epic:
                print(f"  ✅ [{epic.id}] {epic.title}")
    
    # 생성된 파일 출력
    all_files = final_state.get("all_generated_files", [])
    success_files = [f for f in all_files if f.status == "success"]
    
    print(f"\n생성된 파일 ({len(success_files)}개):")
    for file in success_files:
        print(f"  📄 {file.file_path}")
    
    # 토큰 사용량 출력
    token_usage_list = final_state.get("token_usage_list", [])
    if token_usage_list:
        print("\n" + "="*80)
        print("📊 토큰 사용량 상세")
        print("="*80)
        
        total_input = 0
        total_output = 0
        total_tokens = 0
        
        for usage in token_usage_list:
            print(f"\n{usage.step_name}:")
            print(f"  입력 토큰: {usage.input_tokens:,}")
            print(f"  출력 토큰: {usage.output_tokens:,}")
            print(f"  총 토큰: {usage.total_tokens:,}")
            
            total_input += usage.input_tokens
            total_output += usage.output_tokens
            total_tokens += usage.total_tokens
        
        print("\n" + "="*80)
        print("📊 총 토큰 사용량")
        print("="*80)
        print(f"총 입력 토큰: {total_input:,}")
        print(f"총 출력 토큰: {total_output:,}")
        print(f"전체 총 토큰: {total_tokens:,}")
        print("="*80)
    
    return final_state

