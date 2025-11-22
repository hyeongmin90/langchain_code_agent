import threading
from pathlib import Path

# 애플리케이션의 여러 모듈에서 공유되는 전역 컨텍스트입니다.
# AgentApp 인스턴스가 생성될 때 app_instance에 할당됩니다.
app_instance = None

# 도구들 간의 동시 접근을 막기 위한 락
approval_lock = threading.Lock()

# 프로젝트의 기본 작업 디렉토리
BASE_DIR = Path.cwd()
