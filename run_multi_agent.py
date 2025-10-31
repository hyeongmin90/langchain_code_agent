"""
멀티 에이전트 시스템 실행 스크립트

사용법:
    python run_multi_agent.py
"""

import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.coder.coder import main

if __name__ == "__main__":
    main()

