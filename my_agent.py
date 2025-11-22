"""
Code Agent (Synchronous & Multi-Step)
- langchain.agents.create_agent 기반
- 파일 시스템 제어 (목록, 읽기, 쓰기, 편집) 및 터미널 명령어 실행
- 동기 실행 + 자체 계획 및 연속적인 도구 호출
"""
import shutil
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessageChunk
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
from colorama import init, Fore, Back, Style

# 분리된 모듈 import
import agent_context
from agent_utils import UserInterruptedException, check_esc_pressed, clear_key_buffer
from agent_tools import AGENT_TOOLS
from ui_utils import (
    PreviewHandler,
    print_tool_result,
    print_ai_response_start,
    print_separator,
    print_welcome_message,
    get_separator_line
)

# 플랫폼별 초기화
init(autoreset=True, strip=False, convert=False)
import platform
if platform.system() == "Windows":
    import ctypes
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)

# ==========================================
# 애플리케이션 클래스
# ==========================================
class AgentApp:
    def __init__(self):
        # 전역 컨텍스트에 현재 인스턴스 설정
        agent_context.app_instance = self
        
        self.auto_approve_mode = False
        self.user_interrupted = False
        self.session_counter = 1
        self.thread_id = f"session-{self.session_counter:03d}"
        self.agent = self._create_my_agent()

    def _create_my_agent(self):
        """LangChain 에이전트를 생성하고 설정합니다."""
        model = ChatOpenAI(model="gpt-5-mini")

        system_prompt = (
            "당신은 로컬 파일 시스템을 관리하는 전문 AI 개발자입니다. "
            "사용자의 요청을 완수하기 위해 필요한 만큼 도구를 여러 번 사용할 수 있습니다. "
            "최대한 사용자에게 질문을 피하라. 최대한 자동으로 작업을 완료하라."
            "작업 시작 전 계획을 세우고, 순차적으로 도구를 사용하세요. 병렬 처리는 허용하지 않습니다."
            "모든 대화는 한국어로 진행합니다."
            f"사용자 환경: {platform.system()}"
        )
        # agent_tools.py에서 가져온 도구 목록 사용
        return create_agent(model=model, tools=AGENT_TOOLS, checkpointer=InMemorySaver(), system_prompt=system_prompt)

    def _handle_special_commands(self, user_input: str):
        """특수 명령어(/allow, /deny, /status)를 처리합니다."""
        cmd = user_input.lower()
        if cmd == '/allow':
            self.auto_approve_mode = True
            print(f"\n{Fore.GREEN}✓ 자동 승인 모드 활성화됨{Style.RESET_ALL}\n")
        elif cmd == '/deny':
            self.auto_approve_mode = False
            print(f"\n{Fore.YELLOW}✓ 수동 승인 모드 활성화됨{Style.RESET_ALL}\n")
        elif cmd == '/status':
            mode = "자동 승인" if self.auto_approve_mode else "수동 승인"
            color = Fore.GREEN if self.auto_approve_mode else Fore.YELLOW
            print(f"\n{Fore.CYAN}현재 상태: {color}{mode}{Style.RESET_ALL}\n")
        elif cmd == '/reset':
            self._reset_session()
        else:
            print(f"\n{Fore.RED}알 수 없는 명령어: {user_input}{Style.RESET_ALL}\n")

    def _reset_session(self):
        """대화 기록을 초기화하기 위해 새 세션 ID를 생성합니다."""
        self.session_counter += 1
        self.thread_id = f"session-{self.session_counter:03d}"
        print(f"\n{Fore.YELLOW}대화 기록이 초기화되었습니다. {Style.RESET_ALL}")

    def run(self):
        """메인 애플리케이션 루프를 실행합니다."""
        print_welcome_message()
        
        while True:
            try:
                user_input = input(f"{Fore.WHITE}{Style.BRIGHT}> {Style.RESET_ALL}").strip()
                if not user_input: continue
                
                print(get_separator_line(color=Fore.LIGHTBLACK_EX))

                if user_input.lower() in ['종료', 'quit', 'exit', 'q']:
                    print(f"\n{Fore.GREEN}안녕히 가세요!{Style.RESET_ALL}")
                    break
                
                if user_input.startswith('/'):
                    self._handle_special_commands(user_input)
                    continue
                
                self.chat(user_input)
                
            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}종료되었습니다.{Style.RESET_ALL}")
                break
    
    def chat(self, user_input: str):
        """에이전트와 동기적으로 채팅하고 스트리밍 출력을 처리합니다."""
        self.user_interrupted = False
        clear_key_buffer()

        config = {"configurable": {"thread_id": self.thread_id}, "recursion_limit": 100}
        preview_handler = PreviewHandler()
        
        ai_response_started = False
        current_tool_name = None
        tool_header_printed = False
        seen_tool_results = set()

        def _handle_tool_call_chunk(msg_chunk):
            nonlocal current_tool_name, tool_header_printed, ai_response_started
            for chunk in msg_chunk.tool_call_chunks:
                # 새 도구 호출 시작 감지
                if "name" in chunk and chunk["name"]:
                    current_tool_name = chunk["name"]
                    tool_header_printed = False
                    ai_response_started = False
                    # 파일 관련 도구인 경우, 미리보기 세션 시작
                    if current_tool_name in ["write_file"]:
                        preview_handler.start_session(tool_name=current_tool_name)

                # 조용한 도구는 헤더를 출력하지 않음
                silent_tools = ["read_file", "list_files"]
                if current_tool_name and not tool_header_printed and current_tool_name not in silent_tools:
                    print(f"\n{Back.YELLOW}{Fore.BLACK} 🔧 {current_tool_name} {Style.RESET_ALL}")
                    tool_header_printed = True
                
                if preview_handler.preview_active:
                    preview_handler.handle_chunk(chunk)

        try:
            for event in self.agent.stream({"messages": [HumanMessage(content=user_input)]}, config, stream_mode="messages"):
                if self.user_interrupted or check_esc_pressed():
                    self.user_interrupted = True
                    clear_key_buffer()
                    raise UserInterruptedException("사용자가 응답을 중단했습니다.")

                msg, _ = event
                
                if isinstance(msg, AIMessageChunk) and msg.tool_call_chunks:
                    _handle_tool_call_chunk(msg)
                elif hasattr(msg, 'tool_calls') and msg.tool_calls:
                    preview_handler.finish_preview(final_msg=msg)
                    current_tool_name = None
                elif isinstance(msg, AIMessageChunk) and msg.content and not msg.tool_call_chunks:
                    if not ai_response_started:
                        print_ai_response_start()
                        ai_response_started = True
                    print(f"{Fore.GREEN}{msg.content}{Style.RESET_ALL}", end="", flush=True)
                elif msg.__class__.__name__ == 'ToolMessage':
                    preview_handler.cancel_preview()
                    tool_result_id = getattr(msg, 'tool_call_id', None)
                    
                    # 조용한 도구는 결과를 출력하지 않음
                    silent_tools = ["read_file", "list_files"]
                    tool_name = getattr(msg, 'name', current_tool_name)
                    
                    if tool_result_id and tool_result_id not in seen_tool_results:
                        if tool_name not in silent_tools:
                            print_tool_result(msg.content)
                        seen_tool_results.add(tool_result_id)
                    ai_response_started = False

            if ai_response_started: print()
            print_separator()

        except UserInterruptedException:
            preview_handler.cancel_preview()
            clear_key_buffer()
            print(f"\n{Fore.RED}사용자가 작업을 중단했습니다.{Style.RESET_ALL}")
            print_separator()
        except Exception as e:
            preview_handler.cancel_preview()
            clear_key_buffer() 
            print(f"\n{Fore.RED}오류 발생: {e}{Style.RESET_ALL}\n")

# ==========================================
# 메인 실행 블록
# ==========================================
if __name__ == "__main__":
    load_dotenv()
    app = AgentApp()
    app.run()