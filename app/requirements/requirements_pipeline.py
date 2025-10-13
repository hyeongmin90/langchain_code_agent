import os
import json
import asyncio
from dotenv import load_dotenv
import operator
from typing import TypedDict, Annotated, List, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, START, END
import user_qustion_gen
import generate_full_specification


async def main():
    load_dotenv()
    # user_question_result = user_qustion_gen.main()
    temp = """
#### **1. 주제**

다수의 사용자를 지원하는 개인화된 웹 기반 투두 리스트(Todo List) 애플리케이션 개발. 사용자는 자신의 할 일 목록을 생성 및 관리하고, 생성 된 목록을 다른 사용자에게 수정이 불가능한 '읽기 전용' 상태로 공유할 수 있다.

---

#### **2. 기능**

**2.1. 사용자 관리 기능**
*   **회원가입:** 사용자는 시스템에 자신의 계정을 생성할 수 있다.
*   **로그인/로그아웃:** 사용자는 자신의 계정으로 시스템에 접속하고, 세션을 종료할 수 있다.
*   **사용자 인증:** 시스템은 각 사용자를 식별하여, 본인의 투두 리스트에만 접근하고 관리할 수 있도록 보장해야 한다.

**2.2. 투두(Todo) 관리 기능**
*   **할 일 생성:**
    *   사용자는 새로운 할 일을 목록에 추가할 수 있다.
    *   새로운 할 일 항목에는 다음 정보가 반드시 포함되어야 한다.
        *   **할 일 내용:** 수행해야 할 작업에 대한 텍스트 설명
        *   **마감일:** 작업 완료 목표 날짜
        *   **완료 여부:** 작업의 완료 상태 (예: 완료, 미완료)
        *   **중요도:** 작업의 우선순위를 나타내는 지표 (예: 높음, 보통, 낮음)
*   **할 일 조회:**
    *   사용자는 자신이 생성한 모든 할 일 목록을 볼 수 있다.
    *   목록은 마감일, 중요도 등 다양한 기준으로 정렬될 수 있어야 한다.
*   **할 일 수정:**
    *   사용자는 기존에 생성된 할 일의 내용, 마감일, 중요도를 변경할 수 있다.
*   **완료 상태 변경:**
    *   사용자는 할 일의 완료 여부 상태를 변경할 수 있다. (예: 미완료 -> 완료)
*   **할 일 삭제:**
    *   사용자는 더 이상 필요 없는 할 일을 목록에서 영구적으로 제거할 수 있다.

**2.3. 공유 기능**
*   **읽기 전용 공유:**
    *   사용자는 자신의 전체 투두 리스트 또는 특정 할 일 목록을 다른 사람에게 공유할 수 있다.
    *   공유는 고유한 URL 링크 생성 방식 등을 통해 이루어진다.
*   **접근 제어:**
    *   공유된 링크를 통해 투두 리스트에 접근한 타 사용자는 내용을 조회하는 것만 가능하다.
    *   타 사용자는 공유받은 투두 리스트의 어떤 항목(내용, 마감일, 완료 여부, 중요도)도 수정, 추가, 삭제할 수 없다.

---

#### **3. 범위, 가정, 제약**

**3.1. 프로젝트 범위 (Scope)**
*   **포함 (In-Scope):**
    *   개인 사용자 계정 생성 및 인증
    *   사용자별 독립적인 투두 리스트 데이터 관리
    *   개별 할 일 항목의 생성, 조회, 수정, 삭제 (CRUD)
    *   개별 할 일 항목은 '할 일 내용', '마감일', '완료 여부', '중요도' 속성을 가짐
    *   타인에게 투두 리스트를 '읽기 전용'으로 공유하는 기능
*   **제외 (Out-of-Scope):**
    *   **실시간 협업 및 공동 편집 기능:** 여러 사용자가 동시에 하나의 투두 리스트를 수정하는 기능은 포함하지 않는다.
    *   **댓글 및 커뮤니케이션 기능:** 공유된 목록에 댓글을 다는 등의 소통 기능은 제외한다.
    *   **파일 첨부 기능:** 할 일 항목에 파일이나 이미지를 첨부하는 기능은 포함하지 않는다.
    *   **알림 기능:** 마감일 알림 등 푸시 또는 이메일 알림 기능은 제외한다.

**3.2. 가정 (Assumptions)**
*   본 애플리케이션은 최신 버전의 웹 브라우저 환경에서 사용하는 것을 전제로 한다.
*   사용자 데이터는 서로 완벽하게 격리되어야 하며, 한 사용자가 다른 사용자의 데이터에 직접 접근하거나 수정할 수 없다고 가정한다.        
*   '중요도'는 '높음', '보통', '낮음' 등 사전에 정의된 몇 가지 단계로 구현된다고 가정한다.

**3.3. 제약 (Constraints)**
*   모든 공유 기능은 '읽기 전용(Read-only)'으로 엄격히 제한된다. 시스템 아키텍처는 공유된 데이터의 수정을 원천적으로 차단하도록 설계되어야 한다.
*   다중 사용자 환경을 지원하기 위해, 사용자별 데이터베이스 테이블 또는 스키마 분리 등 데이터 격리(Data Isolation) 방안이 반드시 마련되 어야 한다.
*   본 서비스는 웹 기반으로, 사용자의 인터넷 연결이 필수적이다.

---

#### **4. 기타 정보**

*   **주요 사용자:** 개인적인 일정 및 할 일 관리가 필요한 모든 웹 사용자
*   **기대 효과:** 사용자는 체계적인 할 일 관리를 통해 생산성을 향상시킬 수 있으며, 자신의 진행 상황을 타인에게 간편하게 공유(보고)할 수 있다.
    """
    final_user_stories = await generate_full_specification.main(temp)
    
if __name__ == "__main__":
    asyncio.run(main())