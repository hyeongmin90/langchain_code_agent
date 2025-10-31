"""
멀티 에이전트 시스템 메인 진입점

4개의 에이전트:
1. Analyst Agent - 사용자 요청을 Epic List로 분해
2. Planner Agent - Epic을 Task List로 분해
3. Coder Agent - Task List를 파일로 생성
4. Verifier Agent - 도메인 단위 검증
"""

import os
from dotenv import load_dotenv
from .workflow import run_multi_agent_system


def main():
    """메인 함수"""
    load_dotenv()
    
    # API 키 확인
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ 오류: GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다.")
        print("   .env 파일에 GOOGLE_API_KEY를 설정해주세요.")
        return
    
    print("\n" + "="*80)
    print("🤖 멀티 에이전트 코드 생성 시스템")
    print("="*80)
    print("\n이 시스템은 4개의 에이전트가 협력하여 코드를 생성합니다:")
    print("  1. 📊 Analyst Agent - 요청 분석 및 Epic 분해")
    print("  2. 📋 Planner Agent - Epic별 Task 계획")
    print("  3. 💻 Coder Agent - 파일 생성")
    print("  4. 🔍 Verifier Agent - 품질 검증")
    print("\n" + "="*80)
    
    # 사용자 입력
    user_request = input("\n요청사항을 입력하세요: ")
    
    if not user_request.strip():
        print("❌ 요청사항이 비어있습니다.")
        return
    
    # 멀티 에이전트 시스템 실행
    try:
        final_state = run_multi_agent_system(user_request)
        
        # 결과 저장 (선택 사항)
        save = input("\n결과를 파일로 저장하시겠습니까? (y/n): ")
        if save.lower() == 'y':
            import json
            from datetime import datetime
            
            result_file = f"result-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
            
            # 직렬화 가능한 형태로 변환
            result_data = {
                "project_uuid": final_state.get("project_uuid"),
                "user_request": final_state.get("user_request"),
                "status": final_state.get("current_status"),
                "completed_epics": final_state.get("completed_epics", []),
                "final_message": final_state.get("final_message"),
                "generated_files": [
                    {
                        "file_name": f.file_name,
                        "file_path": f.file_path,
                        "status": f.status
                    }
                    for f in final_state.get("all_generated_files", [])
                ]
            }
            
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
            
            print(f"\n✅ 결과가 {result_file}에 저장되었습니다.")
    
    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
