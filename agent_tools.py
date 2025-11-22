import os
import re
import subprocess
import platform
import time
from pathlib import Path

from langchain_core.tools import tool
from colorama import Fore, Style

# 공유 컨텍스트 및 유틸리티 함수 import
from agent_context import app_instance, approval_lock, BASE_DIR
from agent_utils import is_safe_path, check_esc_pressed, UserInterruptedException, clear_key_buffer
from ui_utils import get_separator_line, wrap_text_wide # wrap_text_wide 추가

# ==========================================
# 도구(Tools) 정의 및 관련 헬퍼
# ==========================================

def _request_approval(prompt: str) -> bool:
    """사용자에게 작업을 승인받는 중앙 함수"""
    if app_instance and app_instance.auto_approve_mode:
        print(f"{Fore.GREEN}[자동 승인] {prompt}{Style.RESET_ALL}")
        return True
    
    with approval_lock:
        print(f"\n\n{get_separator_line(char='=', color=Fore.YELLOW, length=80)}")
        print(f"[승인 요청] 다음 작업을 실행하려 합니다:")
        print(f"   {Fore.CYAN}{prompt}{Style.RESET_ALL}")
        print(get_separator_line(char='=', color=Fore.YELLOW, length=80))
        
        approval = input(f"\n{Fore.YELLOW}실행하시겠습니까? (y/n): {Style.RESET_ALL}").strip().lower()
        return approval == 'y'

def _build_tree(directory: Path, prefix: str = "", max_depth: int = 6, current_depth: int = 0) -> tuple[str, int]:
    if current_depth >= max_depth:
        return f"{prefix}└─ ... \n", 1
    
    try:
        excluded_names = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', '.idea', '.vscode'}
        items = [item for item in sorted(directory.iterdir(), key=lambda x: (not x.is_dir(), x.name))
                 if item.name not in excluded_names and not item.name.startswith('.')]
    except PermissionError:
        return f"{prefix}└─ ... (접근 권한 없음)\n", 1
    
    tree_lines, item_count = [], 0
    max_files_in_subdir = 4
    is_root = (current_depth == 0)
    
    files = [item for item in items if item.is_file()]
    dirs = [item for item in items if item.is_dir()]
    
    for i, item in enumerate(dirs):
        is_last_dir = (i == len(dirs) - 1) and len(files) == 0
        connector = "└─ " if is_last_dir else "├─ "
        tree_lines.append(f"{prefix}{connector} {item.name}/\n")
        item_count += 1
        
        extension = "   " if is_last_dir else "│  "
        subtree, sub_count = _build_tree(item, prefix + extension, max_depth, current_depth + 1)
        tree_lines.append(subtree)
        item_count += sub_count
    
    files_to_show = files if is_root else files[:max_files_in_subdir]
    
    for i, item in enumerate(files_to_show):
        is_last = (i == len(files_to_show) - 1)
        connector = "└─ " if is_last else "├─ "
        tree_lines.append(f"{prefix}{connector} {item.name}\n")
        item_count += 1
    
    if not is_root and len(files) > max_files_in_subdir:
        remaining = len(files) - max_files_in_subdir
        tree_lines.append(f"{prefix}└─ ... {remaining}개 파일\n")
        item_count += 1
    
    return "".join(tree_lines), item_count

@tool
def list_files(path: str = ".", max_depth: int = 1) -> str:
    """
    파일 목록을 트리 형태로 조회합니다.
    max_depth: 최대 깊이 (기본값: 1)
    """

    if not is_safe_path(path, BASE_DIR):
        return "보안 경고: 작업 디렉토리 외부 경로는 조회할 수 없습니다."
    try:
        target = (BASE_DIR / path).resolve()
        if not target.exists(): return "경로가 존재하지 않습니다."
        if not target.is_dir(): return "디렉터리가 아닙니다."
        
        tree, count = _build_tree(target, "", max_depth)
        print(f"\n{get_separator_line(char='─', color=Fore.WHITE)}")
        print(f"\n{Style.BRIGHT}ReadFileList{Style.RESET_ALL} {path}")
        print(f"Found {count} items")
        print(f"\n{get_separator_line(char='─', color=Fore.WHITE)}")
        
        return f"{target}\n{tree}\n총 {count}개 항목"
    except Exception as e:
        return f"오류: {e}"

@tool
def read_file(filename: str) -> str:
    """파일 내용을 읽습니다."""
    if not is_safe_path(filename, BASE_DIR):
        return "보안 경고: 작업 디렉토리 외부 파일은 읽을 수 없습니다."
    try:
        print(f"\n{get_separator_line(char='─', color=Fore.WHITE)}")
        print(f"\n{Style.BRIGHT}FileRead{Style.RESET_ALL} {filename}")
        print(f"\n{get_separator_line(char='─', color=Fore.WHITE)}")

        target = (BASE_DIR / filename).resolve()
        if not target.is_file(): 
            return "파일이 아니거나 존재하지 않습니다."
        return f"내용:\n{target.read_text(encoding='utf-8')}"
    except Exception as e:
        return f"읽기 오류: {e}"

@tool
def write_file(filename: str, content: str) -> str:
    """파일을 새로 쓰거나 덮어씁니다."""
    if not is_safe_path(filename, BASE_DIR):
        return "보안 경고: 작업 디렉토리 외부에 파일을 쓸 수 없습니다."
    target = (BASE_DIR / filename).resolve()
    if not _request_approval(f"파일 쓰기: {Fore.CYAN}{str(target)}{Fore.YELLOW}"):
        return "사용자가 파일 쓰기를 거부했습니다."
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding='utf-8')
        return f"저장 완료: {str(target)}"
    except Exception as e:
        return f"쓰기 오류: {e}"

@tool
def edit_file(filename: str, target_text: str, replacement_text: str) -> str:
    """파일의 특정 부분을 수정합니다."""
    if not is_safe_path(filename, BASE_DIR):
        return "보안 경고: 작업 디렉토리 외부 파일은 수정할 수 없습니다."
    target_path = (BASE_DIR / filename).resolve()
    if not target_path.exists(): return "오류: 파일이 존재하지 않습니다."

    prompt_content = f"파일 수정: {Fore.CYAN}{filename}{Style.RESET_ALL}\n\n" \
                     f"{Fore.RED}" + "\n-".join(wrap_text_wide(target_text, 70)) + f"{Style.RESET_ALL}\n\n" \
                     f"{Fore.GREEN}" + "\n+".join(wrap_text_wide(replacement_text, 70)) + Style.RESET_ALL

    if not _request_approval(prompt_content):
        return "사용자가 파일 수정을 거부했습니다."
    try:
        content = target_path.read_text(encoding='utf-8')
        if content.count(target_text) != 1:
            return f"오류: target_text가 {content.count(target_text)}번 발견. 유일한 항목을 지정해야 합니다."
        new_content = content.replace(target_text, replacement_text)
        target_path.write_text(new_content, encoding='utf-8')
        return f"수정 완료: {filename}"
    except Exception as e:
        return f"수정 오류: {e}"

def _adjust_command_for_windows(command: str) -> str:
    if platform.system() == "Windows":
        command = command.replace("./", "", 1) if command.startswith("./") else command
        command = command.replace("gradlew", "gradlew.bat", 1) if command.startswith("gradlew") else command
    return command

def _decode_bytes_output(output_bytes: bytes) -> str:
    if not output_bytes: return ""
    for enc in ("utf-8", "cp949", "latin-1"):
        try: return output_bytes.decode(enc)
        except UnicodeDecodeError: continue
    return str(output_bytes)

@tool
def run_terminal_command(command: str, background: bool = False) -> str:
    """터미널 명령어를 실행합니다."""
    command = _adjust_command_for_windows(command)
    
    danger_patterns = ["rm -rf /", "rm -rf", "sudo", "mkfs", ":(){ :|:& };:"]
    if any(pat in command.lower() for pat in danger_patterns):
        return "보안 경고: 위험한 명령어 패턴이 감지되어 실행이 차단되었습니다."

    if not _request_approval(f"명령어 실행: {Fore.CYAN}{command}{Fore.RED}"):
        return "사용자가 명령 실행을 거부했습니다."

    if app_instance:
        app_instance.clear_key_buffer()

    print(f"   실행 중... {("백그라운드" if background else "")}")

    try:
        if background:
            log_filename = f"{command.split()[0]}_output.log"
            log_path = BASE_DIR / log_filename
            with open(str(log_path), "w", encoding="utf-8") as log_file:
                process = subprocess.Popen(command, shell=True, cwd=str(BASE_DIR), stdout=log_file, stderr=subprocess.STDOUT)
            return f"백그라운드 실행 시작 (PID: {process.pid}), 로그 파일: {log_path}"
        else:
            process = subprocess.Popen(command, shell=True, cwd=str(BASE_DIR), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            while process.poll() is None:
                if check_esc_pressed():
                    process.terminate()
                    clear_key_buffer()  # 키 버퍼 정리
                    raise UserInterruptedException("사용자가 명령어 실행을 중단했습니다.")
                time.sleep(0.1)
            out, err = process.communicate()
            output, error = _decode_bytes_output(out), _decode_bytes_output(err)
            response = f"명령: `{command}`\n"
            if output: response += f"[출력]\n{output}\n"
            if error: response += f"[에러]\n{error}\n"
            if not output and not error: response += "(성공/출력 없음)"
            return response
    except UserInterruptedException as e:
        if app_instance: 
            app_instance.user_interrupted = True
            clear_key_buffer()
        return "사용자가 명령어 실행을 중단했습니다."
    except Exception as e:
        return f"실행 실패: {e}"

# 에이전트 생성 시 사용할 도구 목록
AGENT_TOOLS = [list_files, read_file, write_file, edit_file, run_terminal_command]
