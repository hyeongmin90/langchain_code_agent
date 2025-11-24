import shutil
import time
import unicodedata
import re
import json
from colorama import Fore, Style, Back

# ==========================================
# 텍스트 너비 계산 및 줄바꿈 유틸리티
# ==========================================

def get_char_width(char):
    """문자의 폭을 반환 (전각=2, 반각=1)"""
    return 2 if unicodedata.east_asian_width(char) in ('F', 'W') else 1

def wrap_text_wide(text, width):
    """한글 너비를 고려하여 텍스트를 줄바꿈합니다."""
    lines = []
    current_line = []
    current_width = 0
    for char in text:
        char_width = get_char_width(char)
        if current_width + char_width > width:
            lines.append("".join(current_line))
            current_line = [char]
            current_width = char_width
        else:
            current_line.append(char)
            current_width += char_width
    if current_line:
        lines.append("".join(current_line))
    if not lines and not text:
        return ['']
    return lines

# ==========================================
# 기본 CLI UI 출력 유틸리티
# ==========================================

def get_separator_line(char: str = '═', color: str = Fore.CYAN, length: int = None) -> str:
    """지정된 길이와 스타일의 구분선 문자열을 반환합니다."""
    if length is None:
        try:
            length, _ = shutil.get_terminal_size()
        except OSError:
            length = 80
    return f"{color}{char * length}{Style.RESET_ALL}"

def print_welcome_message():
    """환영 메시지와 사용법 팁을 출력합니다."""
    print(f"\n{get_separator_line()}")
    try:
        columns, _ = shutil.get_terminal_size()
    except OSError:
        columns = 80
    title = "🚀 LangChain My Code Agent 🚀"
    padding = (columns - len(title)) // 2
    print(f"{Fore.CYAN}{Style.BRIGHT}{' ' * padding}{title}")
    print(get_separator_line())
    
    print(f"\n{Fore.YELLOW}💡 사용 팁:")
    print(f"   • AI 응답/명령어 중단: {Fore.RED}ESC{Fore.YELLOW} 키")
    print(f"   • 자동 승인 모드: {Fore.RED}/allow{Fore.YELLOW}")
    print(f"   • 수동 승인 모드: {Fore.RED}/deny{Fore.YELLOW}")
    print(f"   • 현재 상태 확인: {Fore.RED}/status{Fore.YELLOW}")
    print(f"   • 프로그램 종료: {Fore.RED}'quit', 'exit', '종료'{Fore.YELLOW} 입력\n")

def print_tool_result(result: str):
    """도구 실행 결과를 구분선과 함께 출력합니다."""
    print(f"\n{get_separator_line(char='─', color=Fore.WHITE)}")
    print(f"실행 결과:")
    
    content = result[:1000] + "..." if len(result) > 1000 else result
    print(f"{Fore.WHITE}{content}{Style.RESET_ALL}")
    print(get_separator_line(char='─', color=Fore.WHITE))

def print_ai_response_start():
    """AI 응답 시작을 알립니다."""
    print(f"\n{Back.GREEN}{Fore.BLACK}  AI 응답 {Style.RESET_ALL}")
    print(f"{Fore.GREEN}  ", end="", flush=True)

def print_separator():
    """터미널 너비에 맞는 표준 구분선을 출력합니다."""
    print(f"\n{get_separator_line(color=Fore.LIGHTBLACK_EX)}\n", end="")

# ==========================================
# 파일 내용 미리보기 핸들러 클래스
# ==========================================
class PreviewHandler:
    """파일 쓰기/수정 시 실시간 미리보기 UI를 관리하는 클래스"""
    def __init__(self, preview_update_interval=0.1):
        self.preview_update_interval = preview_update_interval
        self.preview_active = False
        self.header_printed = False
        self.target_key = None
        self.filename = None
        self.args_buffer = ""
        self.full_value_content = ""
        self.last_preview_update = 0
        self.file_content_lines_info = []
        self.last_printed_lines = 0

        try:
            self.cols, _ = shutil.get_terminal_size()
        except OSError:
            self.cols = 80
            
    def start_session(self, tool_name: str):
        self.preview_active = True
        self.header_printed = False
        self.target_key = "content" if tool_name == "write_file" else "replacement_text"
        self.filename = None
        self.args_buffer = ""
        self.full_value_content = ""
        self.last_preview_update = 0
        self.file_content_lines_info = []

    def _print_header(self):
        if self.header_printed:
            return
        safe_width = min(self.cols, 80)
        title = f"작성 중: {self.filename}" if self.filename else "파일 미리보기"
        
        print(get_separator_line(length=safe_width))
        print(f"{Fore.CYAN}{title} {Style.RESET_ALL}")
        print(get_separator_line(length=safe_width))
        
        self.header_printed = True

    def _update_screen(self):
        if not self.header_printed:
            return
            
        safe_width = min(self.cols, 80)
        display_width = self.cols - 6
        
        # 이전 출력 지우기
        if self.last_printed_lines > 0:
            # 내용 줄 + 하단 구분선 1줄
            print(f"\033[{self.last_printed_lines + 1}A", end='')
            print(f"\r\033[J", end='') # 커서 위치부터 화면 끝까지 지우기

        # 표시할 내용 계산
        clean_content = self.full_value_content.replace("\\n", "\n").replace('\\"', '"')
        real_lines = clean_content.split('\n')
        visual_lines_with_info = []
        
        for logical_idx, line in enumerate(real_lines):
            logical_line_num = logical_idx + 1
            if line:
                wrapped = wrap_text_wide(line, display_width)
                for w_idx, w_line in enumerate(wrapped):
                    visual_lines_with_info.append((str(logical_line_num) if w_idx == 0 else ".", w_line))
            else:
                visual_lines_with_info.append((str(logical_line_num), ''))
        
        self.file_content_lines_info = visual_lines_with_info
        
        # 새 내용 출력
        for line_mark, display_line in visual_lines_with_info:
            mark_str = f"{Fore.CYAN}{line_mark:>4}{Style.RESET_ALL}" if line_mark != "." else f"{Fore.BLACK}{line_mark:>4}{Style.RESET_ALL}"
            print(f"{mark_str}│ {Fore.YELLOW}{display_line}{Style.RESET_ALL}")
            
        print(get_separator_line(length=safe_width), flush=True)
        
        # 다음 업데이트를 위해 출력한 줄 수 저장
        self.last_printed_lines = len(visual_lines_with_info)

    def handle_chunk(self, chunk: dict):
        if not self.preview_active or "args" not in chunk or not chunk["args"]:
            return
        self.args_buffer += chunk["args"]
        
        if not self.filename:
            match = re.search(r'"filename"\s*:\s*"([^"]+)"', self.args_buffer)
            if match:
                self.filename = match.group(1)

        # 파일명이 있고, 헤더가 아직 출력되지 않았다면 출력
        if self.filename and not self.header_printed:
            self._print_header()

        content_to_preview = ""
        force_update = False
        match_start = re.search(rf'"{self.target_key}"\s*:\s*"', self.args_buffer)
        if match_start:
            start_index = match_start.end()
            potential_content = self.args_buffer[start_index:]
            match_end = re.search(r'(?<!\\)"', potential_content)
            if match_end:
                content_to_preview = potential_content[:match_end.start()]
                force_update = True
            else:
                content_to_preview = potential_content
        self.full_value_content = content_to_preview

        current_time = time.time()
        if force_update or (current_time - self.last_preview_update >= self.preview_update_interval):
            self._update_screen()
            self.last_preview_update = time.time()


    def cancel_preview(self):
        if self.preview_active:
            self.preview_active = False
            if self.header_printed:
                print(f"\n{Fore.GREEN}작성 처리 완료{Style.RESET_ALL}\n")