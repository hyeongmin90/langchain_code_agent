import platform
import time
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessageChunk
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
from colorama import init, Fore, Style


from agent import context as agent_context
from agent.utils import UserInterruptedException, check_esc_pressed, clear_key_buffer, log_message, update_token_usage
from agent.debug import PromptInspector
from agent.tools import AGENT_TOOLS
from agent.ui import (
    PreviewHandler,
    print_ai_response_start,
    print_separator,
    print_welcome_message,
    get_separator_line,
)

# 플랫폼별 초기화
init(autoreset=True, strip=False, convert=False)
if platform.system() == "Windows":
    import ctypes
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)

# ==========================================
# 애플리케이션 클래스   
# ==========================================
class AgentApp:
    def __init__(self):
        agent_context.app_instance = self
        
        self.background_processes = []
        self.auto_approve_mode = False
        self.user_interrupted = False
        self.session_counter = 1
        self.thread_id = f"session-{self.session_counter:03d}"
        self.agent = self._create_my_agent()

    def _create_my_agent(self):
        """LangChain 에이전트를 생성하고 설정합니다."""
        model = ChatOpenAI(model="gpt-5-mini")
        # model = ChatOpenAI(model="gpt-5-mini", callbacks=[PromptInspector()])

        system_prompt = (
            "당신은 로컬 파일 시스템을 관리하는 전문 AI 개발자입니다. "
            "사용자의 요청을 완수하기 위해 필요한 만큼 도구를 여러 번 사용할 수 있습니다. "
            "최대한 사용자에게 질문을 피하라. 최대한 자동으로 작업을 완료하라."
            "작업 시작 전 계획을 세우고, 순차적으로 도구를 사용하세요. 병렬 처리는 허용하지 않습니다."
            "모든 작업은 사용자가 도중에 중지할 수 있다."
            "모든 대화는 한국어로 진행합니다."
        )

        return create_agent(
            model=model, 
            tools=AGENT_TOOLS, 
            checkpointer=InMemorySaver(), 
            system_prompt=system_prompt,
            debug=False
        )

    def _handle_special_commands(self, user_input: str):
        """특수 명령어(/allow, /deny, /status)를 처리합니다."""
        log_message(f"SPECIAL COMMAND: {user_input}")
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
    
    def _cleanup_background_processes(self):
        """백그라운드 프로세스들을 종료합니다."""
        import subprocess as sp
        while self.background_processes:
            bg_info = self.background_processes.pop(0)
            process = bg_info['process']
            pid = bg_info['pid']
            log_message(f"BACKGROUND PROCESS TERMINATING: {pid}")
            try:
                if platform.system() == "Windows":
                    sp.run(["taskkill", "/F", "/T", "/PID", str(pid)], 
                           capture_output=True, timeout=5)
                else:
                    process.terminate()
                    process.wait(timeout=2)

                print(f"\n{Fore.YELLOW}백그라운드 프로세스 종료됨: PID {pid}{Style.RESET_ALL}")
                log_message(f"BACKGROUND PROCESS TERMINATED SUCCESSFULLY: {pid}")
            except Exception as e:
                print(f"\n{Fore.RED}백그라운드 프로세스 종료 실패 (PID {pid}): {e}{Style.RESET_ALL}\n")
                log_message(f"BACKGROUND PROCESS TERMINATION ERROR: {e}")
        
        time.sleep(0.5)
    
    def _cleanup_log_files(self):
        """모든 로그 파일을 삭제합니다."""
        try:
            log_dir = agent_context.CODE_DIR / "temp_logs"
            deleted_count = 0
            failed_count = 0
            for log_file in log_dir.glob("*.log"):
                try:
                    log_file.unlink()
                    deleted_count += 1
                    log_message(f"LOG FILE DELETED: {log_file.name}")
                except Exception as e:
                    failed_count += 1
                    self._log_message(f"LOG FILE DELETE ERROR: {log_file.name} - {e}")
            
            if deleted_count > 0:
                print(f"{Fore.CYAN}로그 파일 {deleted_count}개가 정리되었습니다.{Style.RESET_ALL}")
            if failed_count > 0:
                print(f"{Fore.YELLOW}로그 파일 {failed_count}개는 삭제할 수 없습니다 (프로세스가 사용 중).{Style.RESET_ALL}")
        except Exception as e:
            log_message(f"LOG CLEANUP ERROR: {e}")

    def run(self):
        """메인 애플리케이션 루프를 실행합니다."""
        print_welcome_message()
        self._log_message(f"APPLICATION: 대화가 시작되었습니다.")
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
                log_message(f"APPLICATION: 종료되었습니다.")
                break
        
        # 정리 작업
        self._cleanup_background_processes()
        self._cleanup_log_files()
        
        # 토큰 사용량 출력
        print(f"\n{Fore.CYAN}입력 토큰 수: {agent_context.INPUT_TOKEN_COUNT}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}출력 토큰 수: {agent_context.OUTPUT_TOKEN_COUNT}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}총 토큰 사용량: {agent_context.TOTAL_TOKEN_USAGE}{Style.RESET_ALL}")

        log_message(f"APPLICATION: 총 토큰 사용량: {agent_context.TOTAL_TOKEN_USAGE}")
        log_message(f"APPLICATION: 입력 토큰 수: {agent_context.INPUT_TOKEN_COUNT}")
        log_message(f"APPLICATION: 출력 토큰 수: {agent_context.OUTPUT_TOKEN_COUNT}")    
                
    
    def chat(self, user_input: str):
        """에이전트와 동기적으로 채팅하고 스트리밍 출력을 처리합니다."""
        self.user_interrupted = False
        clear_key_buffer()
        
        log_message(f"USER: {user_input}")

        config = {"configurable": {"thread_id": self.thread_id}, "recursion_limit": 100}
        preview_handler = PreviewHandler()
        
        ai_response_started = False
        current_tool_name = None
        tool_header_printed = False

        def _handle_tool_call_chunk(msg_chunk):
            nonlocal current_tool_name, tool_header_printed, ai_response_started
            for chunk in msg_chunk.tool_call_chunks:
                
                if "name" in chunk and chunk["name"]:
                    current_tool_name = chunk["name"]
                    tool_header_printed = False
                    ai_response_started = False
                  
                    if current_tool_name in ["write_file"]:
                        preview_handler.start_session(tool_name=current_tool_name)
                
                if preview_handler.preview_active:
                    preview_handler.handle_chunk(chunk)

        ready_to_exit = False 

        try:
            for event in self.agent.stream({"messages": [HumanMessage(content=user_input)]}, config, stream_mode="messages"):
                
                msg, _ = event
                
                if msg.__class__.__name__ == 'ToolMessage':
                    log_message(f"TOOL MESSAGE: {msg.content}")
                elif hasattr(msg, "content") and msg.content:
                    log_message(f"AI: {msg.content}")
                elif hasattr(msg, "tool_call_chunks") and msg.tool_call_chunks:
                    log_message(f"TOOL CALL CHUNKS: {msg.tool_call_chunks}")
                else:
                    log_message(f"MESSAGE: {str(msg)}")
                
                update_token_usage(msg)
                
                if ready_to_exit:
                    print_separator()
                    return

                if self.user_interrupted and msg.__class__.__name__ != 'ToolMessage':
                    continue

                # ---------------------------------------------------------
                # 1. 도구 실행 결과 (ToolMessage)
                # ---------------------------------------------------------
                if msg.__class__.__name__ == 'ToolMessage':
                    if 'preview_handler' in locals(): preview_handler.cancel_preview()
                    ai_response_started = False
                    
                    if self.user_interrupted:
                        print(f"\n{Fore.GREEN}✓ 작업이 중단되었습니다.{Style.RESET_ALL}")
                        ready_to_exit = True
                        continue

                # ---------------------------------------------------------
                # 2. AI 텍스트 응답 (AIMessageChunk)
                # ---------------------------------------------------------
                elif isinstance(msg, AIMessageChunk) and msg.content:
                    if ready_to_exit:
                        print(f"{Fore.GREEN} 대기 상태 복귀{Style.RESET_ALL}")
                        print_separator()
                        return 

                    if self.user_interrupted: continue
                    if check_esc_pressed():
                        self.user_interrupted = True
                        raise UserInterruptedException("텍스트 생성 중단")

                    if not msg.tool_call_chunks:
                        if not ai_response_started:
                            print_ai_response_start()
                            ai_response_started = True
                        print(f"{Fore.GREEN}{msg.content}{Style.RESET_ALL}", end="", flush=True)

                # ---------------------------------------------------------
                # 3. 도구 호출 생성 (AIMessageChunk with tool_calls)
                # ---------------------------------------------------------
                elif isinstance(msg, AIMessageChunk) and msg.tool_call_chunks:

                    if ready_to_exit:
                        print(f"{Fore.GREEN} 대기 상태 복귀{Style.RESET_ALL}")
                        print_separator()
                        return
                    
                    if self.user_interrupted or ready_to_exit: continue
                    
                    if check_esc_pressed():
                        self.user_interrupted = True
                        raise UserInterruptedException("도구 호출 생성 중단")

                    _handle_tool_call_chunk(msg)

            if ai_response_started: print()
            print_separator()

        except UserInterruptedException:
            if 'preview_handler' in locals(): preview_handler.cancel_preview()
            clear_key_buffer()
            print(f"\n{Fore.RED}사용자가 작업을 중단했습니다.{Style.RESET_ALL}")
            print_separator()
            log_message(f"USER INTERRUPTED: 사용자가 작업을 중단했습니다.")
           
        except Exception as e:
            if 'preview_handler' in locals(): preview_handler.cancel_preview()
            clear_key_buffer() 
            print(f"\n{Fore.RED}오류 발생: {e}{Style.RESET_ALL}\n")
            log_message(f"ERROR: {e}")

# ==========================================
# 메인 실행 블록
# ==========================================
if __name__ == "__main__":
    load_dotenv()
    app = AgentApp()
    app.run()

