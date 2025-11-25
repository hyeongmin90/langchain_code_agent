
import subprocess
import platform
import time
import shutil
from pathlib import Path

from langchain_core.tools import tool
from colorama import Fore, Style
from . import context
from .context import approval_lock, BASE_DIR, CODE_DIR
from .utils import is_safe_path, check_esc_pressed, UserInterruptedException, clear_key_buffer
from .ui import get_separator_line, wrap_text_wide, TerminalOutputViewer

# ==========================================
# 도구(Tools) 정의 및 관련 헬퍼
# ==========================================


def _request_approval(prompt: str) -> bool:
    """사용자에게 작업을 승인받는 중앙 함수"""
    app = context.app_instance
    if app and app.auto_approve_mode:
        print(f"{Fore.GREEN}[자동 승인]\n {prompt}{Style.RESET_ALL}")
        return True
    
    with approval_lock:
        print(f"\n{get_separator_line(color=Fore.YELLOW)}")
        print(f"\n{prompt}")
        print(f"\n{get_separator_line(color=Fore.YELLOW)}")
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
    if not _request_approval(f"Write File: {str(target)}"):
        return "사용자가 파일 쓰기를 거부했습니다."
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding='utf-8')
        return f"저장 완료: {str(target)}"
    except Exception as e:
        return f"쓰기 오류: {e}"

def format_diff_with_lines(start_line: int, old_text: str, new_text: str) -> str:
    try:
        cols, _ = shutil.get_terminal_size()
    except:
        cols = 80
    
    display_width = max(cols - 10, 40) 
    result = []

    old_lines = old_text.split('\n')
    for i, line in enumerate(old_lines):
        current_line_num = start_line + i
        if line:
            wrapped = wrap_text_wide(line, display_width)
            for w_idx, w_line in enumerate(wrapped):
                num_str = str(current_line_num) if w_idx == 0 else "."
                result.append(f"{num_str:>4}│ {Fore.RED}- {w_line}{Style.RESET_ALL}")
        else:
            result.append(f"{current_line_num:>4}│ {Fore.RED}- {Style.RESET_ALL}")

    result.append(get_separator_line(char='─', color=Fore.WHITE))

    new_lines = new_text.split('\n')

    for i, line in enumerate(new_lines):
        current_line_num = start_line + i
        if line:
            wrapped = wrap_text_wide(line, display_width)
            for w_idx, w_line in enumerate(wrapped):
                num_str = str(current_line_num) if w_idx == 0 else "."
                result.append(f"{num_str:>4}│ {Fore.GREEN}+ {w_line}{Style.RESET_ALL}")
        else:
            result.append(f"{current_line_num:>4}│ {Fore.GREEN}+{Style.RESET_ALL}")

    return "\n".join(result)

@tool
def edit_file(filename: str, target_text: str, replacement_text: str) -> str:
    """파일의 특정 부분을 수정합니다."""
    if not is_safe_path(filename, BASE_DIR):
        return "보안 경고: 작업 디렉토리 외부 파일은 수정할 수 없습니다."
    
    target_path = (BASE_DIR / filename).resolve()
    if not target_path.exists(): return "오류: 파일이 존재하지 않습니다."

    try:
        content = target_path.read_text(encoding='utf-8')
        
        count = content.count(target_text)
        if count == 0:
            return "오류: 파일에서 target_text를 찾을 수 없습니다."
        elif count > 1:
            return f"오류: target_text가 {count}번 발견되었습니다. 더 긴 문맥을 사용하여 유일한 부분을 지정해주세요."

        start_index = content.find(target_text)
        start_line = content[:start_index].count('\n') + 1

        print(f"\n{get_separator_line(char='─', color=Fore.WHITE)}")
        print(f"\n{Style.BRIGHT}Edit File{Style.RESET_ALL} {filename}")
        
        prompt_content = format_diff_with_lines(start_line, target_text, replacement_text)

        if not _request_approval(prompt_content):
            return "사용자가 파일 수정을 거부했습니다."
        
        new_content = content.replace(target_text, replacement_text)
        target_path.write_text(new_content, encoding='utf-8')
        
        return f"수정 완료: {filename} ({start_line}번째 줄 수정됨)"

    except Exception as e:
        return f"수정 오류: {e}"
    finally:
        print(f"\n{get_separator_line(char='─', color=Fore.WHITE)}")


def _decode_bytes_output(output_bytes: bytes) -> str:
    if not output_bytes: return ""
    for enc in ("utf-8", "cp949", "latin-1"):
        try: return output_bytes.decode(enc)
        except UnicodeDecodeError: continue
    return str(output_bytes)

@tool
def run_terminal_command(command: str, max_display_time: float = 10.0) -> str:
    """
    터미널 명령어를 실행한다.
    max_display_time 초 동안의 실행 로그를 반환하며, 초과시 백그라운드로 전환된다.

    Args:
        command: 실행할 터미널 명령어
        max_display_time: 최대 표시 시간 (기본값: 10.0초)
    """

    danger_patterns = ["rm -rf /", "rm -rf", "sudo", "mkfs", ":(){ :|:& };:"]
    if any(pat in command.lower() for pat in danger_patterns):
        return "보안 경고: 위험한 명령어 패턴이 감지되어 실행이 차단되었습니다."
        
    print(f"\n{get_separator_line(char='─', color=Fore.WHITE)}")
    print(f"\n{Style.BRIGHT}RunTerminalCommand{Style.RESET_ALL}")
    print(f"\n{get_separator_line(char='─', color=Fore.WHITE)}")

    if not _request_approval(f"Run: {command}"):
        return "사용자가 명령 실행을 거부했습니다."

    app = context.app_instance

    log_dir = CODE_DIR / "temp_logs"
    log_dir.mkdir(exist_ok=True)
    
    log_path = log_dir / "latest.log"
    
    try:
        if platform.system() == "Windows" and not command.lower().startswith("powershell"):
            final_command = f"chcp 65001 >nul & {command}"
        else:
            final_command = command
        
        with open(str(log_path), "wb") as log_file:
            process = subprocess.Popen(
                final_command, 
                shell=True, 
                cwd=str(BASE_DIR), 
                stdout=log_file, 
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL
            )

        viewer = TerminalOutputViewer(str(log_path), max_lines=10)
        viewer.start(command)
        
        start_time = time.time()
        detached = False
        
        while process.poll() is None:
            viewer.update()
            
            if check_esc_pressed():
                process.terminate()
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
                
                viewer.stop(f"{Fore.RED}사용자가 중단했습니다.{Style.RESET_ALL}")
                clear_key_buffer()
                
                if app:
                    app.user_interrupted = True
                
                raise UserInterruptedException("사용자가 명령어 실행을 중단했습니다.")
            
            if time.time() - start_time > max_display_time:
                detached = True
                break
            
            time.sleep(0.1)
        
        if detached:
            viewer.stop(f"{Fore.CYAN} 백그라운드 전환(PID: {process.pid}){Style.RESET_ALL}")
            try:
                with open(str(log_path), 'rb') as f:
                    partial_output = _decode_bytes_output(f.read())
            except:
                partial_output = "(출력을 읽을 수 없습니다)"
            
            if app:
                app.background_processes.append(process)
            
            return f"{partial_output}\n...\n 백그라운드 전환 (PID: {process.pid})\n 로그 확인: view_last_terminal_log 도구 사용"
        
        else:
            time.sleep(0.1)
            viewer.update()
            
            return_code = process.returncode
            status_color = Fore.GREEN if return_code == 0 else Fore.RED
            
            viewer.stop(f"{status_color} 실행 완료 (종료코드: {return_code}){Style.RESET_ALL}")
            
            try:
                with open(str(log_path), 'rb') as f:
                    output = _decode_bytes_output(f.read())
            except:
                output = "(출력을 읽을 수 없습니다)"
            
            if len(output) > 2000:
                summary = f"...(생략)...\n{output[-2000:]}"
            else:
                summary = output if output else "(출력 없음)"
            
            return summary

    except UserInterruptedException as e:
        if app:
            app.user_interrupted = True
        try:
            with open(str(log_path), 'rb') as f:
                interrupted_output = _decode_bytes_output(f.read())
        except:
            interrupted_output = "(출력을 읽을 수 없습니다)"

        return f"{interrupted_output}\n...\n 사용자가 명령어 실행을 중단했습니다."
    except Exception as e:
        return f"실행 실패: {e}"

@tool
def view_last_terminal_log(lines: int = 50) -> str:
    """
    마지막으로 실행된 터미널 명령어의 로그를 확인합니다.
    
    Args:
        lines: 마지막 N줄만 표시 (기본값: 50)
    """
    log_dir = CODE_DIR / "temp_logs"
    log_path = log_dir / "latest.log"
    
    try:
        if not log_path.exists():
            return "아직 실행된 터미널 명령이 없습니다."
        
        with open(log_path, 'rb') as f:
            content = _decode_bytes_output(f.read())
        
        all_lines = content.split('\n')
        
        if len(all_lines) > lines:
            selected_lines = all_lines[-lines:]
            content = f"...(총 {len(all_lines)}줄 중 마지막 {lines}줄)\n" + "\n".join(selected_lines)
        else:
            content = "\n".join(all_lines)
        
        if not content.strip():
            content = "(출력 없음)"

        print(f"\n{get_separator_line(char='─', color=Fore.WHITE)}")
        print(f"\n{Style.BRIGHT}ViewLastTerminalLog{Style.RESET_ALL}")
        print(f"\n{get_separator_line(char='─', color=Fore.WHITE)}")
        
        return content
        
    except Exception as e:
        return f"로그 읽기 실패: {e}"

AGENT_TOOLS = [list_files, read_file, write_file, edit_file, run_terminal_command, view_last_terminal_log]
