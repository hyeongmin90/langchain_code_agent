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
    specification_writer_agent,
    code_generator_agent,
    static_reviewer_agent,
    orchestrator_agent,
)




def orchestrator_router(state: AgenticCoderState) -> str:
    """
    오케스트라 에이전트의 결정에 따라 다음 노드로 라우팅
    """
    next_action = state.get("current_status", "completed")
    
    # next_action이 노드 이름이면 해당 노드로, 아니면 END
    if next_action in ["specification_writer", "code_generator", "static_reviewer"]:
        return next_action
    elif next_action == "completed":
        return END
    elif next_action == "failed":
        return END
    else:
        # 기본값: 완료
        return END


# ============================================
# 워크플로우 생성
# ============================================

def create_agentic_coder_workflow():
    """
    에이전틱 오케스트라 기반 Agentic Coder 워크플로우 생성
    
    구조:
    1. START → Orchestrator (초기 분석)
    2. Orchestrator → Specification Writer
    3. Specification Writer → Orchestrator (결과 분석)
    4. Orchestrator → Code Generator
    5. Code Generator → Orchestrator (결과 분석)
    6. Orchestrator → Static Reviewer
    7. Static Reviewer → Orchestrator (결과 분석 및 재시도 판단)
    8. Orchestrator → END
    
    플로우:
               ┌─────────────────┐
               │  Orchestrator   │ ◀─┐
               │   (LLM based)   │   │
               └─────────────────┘   │
                 │   │   │   │       │
                 ▼   ▼   ▼   ▼       │
               ┌───┬───┬───┬───┐     │
               │Sp │Co │Re │END│     │
               │ec │de │vi │   │     │
               └─┬─┴─┬─┴─┬─┴───┘     │
                 │   │   │           │
                 └───┴───┴───────────┘
    
    특징:
    - 중앙 집중식 에이전틱 오케스트레이션
    - LLM이 상황을 분석하고 동적으로 다음 단계 결정
    - 단순 조건문이 아닌 지능적 의사결정
    - 각 단계 후 오케스트라가 결과를 평가하고 판단
    """
    
    workflow = StateGraph(AgenticCoderState)
    
    # 노드 추가
    workflow.add_node("orchestrator", orchestrator_agent)
    workflow.add_node("specification_writer", specification_writer_agent)
    workflow.add_node("code_generator", code_generator_agent)
    workflow.add_node("static_reviewer", static_reviewer_agent)
    
    # 시작: START → Orchestrator (첫 분석)
    workflow.add_edge(START, "orchestrator")
    
    # Orchestrator → 조건부 라우팅 (LLM 결정에 따라)
    workflow.add_conditional_edges(
        "orchestrator",
        orchestrator_router,
        {
            "specification_writer": "specification_writer",
            "code_generator": "code_generator",
            "static_reviewer": "static_reviewer",
            END: END
        }
    )
    
    # 각 에이전트 → Orchestrator로 복귀 (결과 분석)
    workflow.add_edge("specification_writer", "orchestrator")
    workflow.add_edge("code_generator", "orchestrator")
    workflow.add_edge("static_reviewer", "orchestrator")
    
    return workflow.compile()


# ============================================
# 실행 함수
# ============================================

def run_agentic_coder(user_request: str, max_retries: int = 2):
    """
    Agentic Coder 시스템 실행
    
    Args:
        user_request: 사용자 요청 (예: "간단한 Todo API를 만들어줘. CRUD 기능만 있으면 돼.")
        max_retries: 최대 재시도 횟수 (기본값: 2)
    
    Returns:
        최종 상태 (AgenticCoderState)
    
    예시:
        >>> result = run_agentic_coder("사용자 관리 API를 만들어줘. 회원가입, 로그인, 프로필 조회 기능.")
        >>> print(result["final_message"])
    """
    
    print("\n" + "="*100)
    print("🚀 Agentic Coder 시스템 시작")
    print("="*100)
    print(f"📝 사용자 요청: {user_request}")
    print(f"🔄 최대 재시도 횟수: {max_retries}")
    print("="*100)
    
    load_dotenv()
    # 워크플로우 생성
    app = create_agentic_coder_workflow()
    
    # 초기 상태
    initial_state = {
        "user_request": user_request,
        "specification": None,
        
        # 파일 계획 (오케스트라가 관리)
        "files_plan": [],
        "current_file_index": 0,
        "next_file_to_generate": None,
        
        # 코드 생성
        "generated_files": [],
        "current_file_code": None,
        
        # 리뷰
        "review_result": None,
        "review_passed": False,
        "issues_found": [],
        
        # 상태 관리
        "current_status": "start",  # 초기값: orchestrator가 판단하도록
        "retry_count": 0,
        "max_retries": max_retries,
        
        # 최종 결과
        "final_code": None,
        "final_message": None,
        
        # 토큰 사용량
        "token_usage_list": [],
    }
    
    # 실행
    print("\n🎬 워크플로우 실행 시작...\n")
    # 파일이 많을 경우를 대비해 충분한 recursion limit 설정
    # 예상: START(1) + 명세(3) + 파일생성(파일수×2) + 리뷰(2) + END(1) = 약 7 + 파일수×2
    final_state = app.invoke(
        initial_state,
        config={"recursion_limit": 150}  # 파일 50개까지 처리 가능
    )
    
    # 결과 출력
    print("\n" + "="*100)
    print("🎉 Agentic Coder 시스템 완료")
    print("="*100)
    
    if final_state.get("final_message"):
        print(f"\n📢 {final_state['final_message']}\n")
    
    # 파일 계획 요약
    if final_state.get("files_plan"):
        print(f"📋 파일 계획: {len(final_state['files_plan'])}개 파일")
        for fp in final_state["files_plan"]:
            print(f"   - {fp['file_name']} ({fp['file_path']})")
        print()
    
    # 생성된 파일 목록
    if final_state.get("generated_files"):
        print(f"📄 생성된 파일: {len(final_state['generated_files'])}개")
        for file in final_state["generated_files"]:
            print(f"   - {file['file_path']}/{file['file_name']}")
        print()
    
    # 리뷰 결과
    if final_state.get("review_passed"):
        print("✅ 정적 리뷰: PASS")
    else:
        print("❌ 정적 리뷰: FAIL")
        if final_state.get("issues_found"):
            print(f"   발견된 이슈: {len(final_state['issues_found'])}개")
            for issue in final_state["issues_found"][:5]:  # 최대 5개만 표시
                print(f"   - {issue}")
            if len(final_state["issues_found"]) > 5:
                print(f"   ... 외 {len(final_state['issues_found']) - 5}개")
    
    print("\n" + "="*100)
    
    return final_state


# ============================================
# 결과 내보내기 (선택사항)
# ============================================

def export_code_to_files(final_state: AgenticCoderState, output_dir: str = "./generated_code"):
    """
    생성된 코드를 실제 파일로 저장
    
    Args:
        final_state: 실행 완료된 상태
        output_dir: 출력 디렉토리
    """
    import os
    from pathlib import Path
    
    if not final_state.get("generated_files"):
        print("❌ 생성된 코드가 없습니다.")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📦 코드 파일 저장 중: {output_dir}")
    
    for file_info in final_state["generated_files"]:
        file_path = output_path / file_info["file_path"]
        file_path.mkdir(parents=True, exist_ok=True)
        
        full_file_path = file_path / file_info["file_name"]
        
        with open(full_file_path, "w", encoding="utf-8") as f:
            f.write(file_info["code_content"])
        
        print(f"   ✅ {full_file_path}")
    
    print(f"\n✅ 총 {len(final_state['code_files'])}개 파일 저장 완료!")


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
    
    result = run_agentic_coder(user_request, max_retries=2)
    
    # 코드 파일로 저장
    if result.get("review_passed"):
        export_code_to_files(result, output_dir="./generated_todo_api")

