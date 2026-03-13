import os
import sys
import warnings
from dotenv import load_dotenv
from colorama import init, Fore, Style

# Pydantic 경고 무시
warnings.filterwarnings("ignore", category=UserWarning, module=r"pydantic\..*")
warnings.filterwarnings("ignore", message="Pydantic serializer warnings", category=UserWarning)

# 프로젝트 루트 경로 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

load_dotenv()
init(autoreset=True)

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pipeline.retriever import query_hybrid

# ──────────────────────────────────────────────
# 1. 강화된 시스템 프롬프트 정의
# ──────────────────────────────────────────────
SYSTEM_PROMPT = """당신은 Spring Framework 전문가입니다. 
제공된 문서 컨텍스트만을 사용하여 정확하고 기술적이며 코드 중심적인 답변을 제공하는 것이 목표입니다.

[제약 사항]
1. 반드시 제공된 문서를 주요 근거로 사용합니다.
2. 문서에 없는 내용은 만들어내지 않습니다.
3. 답을 찾을 수 없는 경우 그렇게 명확히 말합니다.
4. 가능하면 문서 내용을 근거로 설명합니다.
5. 전문적이고 간결하게 작성하세요. 항상 한국어로 답변하세요.

[Context]
{context}
"""

BANNER = f"""{Fore.CYAN}
╔══════════════════════════════════════════════════════╗
║         LangGraph RAG Agent  (Spring Docs)           ║
║         'exit' 또는 'q' 를 입력하면 종료됩니다       ║
╚══════════════════════════════════════════════════════╝
{Style.RESET_ALL}"""

def _format_docs(docs: list) -> str:
    """검색 결과 문서를 컨텍스트 문자열로 포맷팅"""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        header = doc.metadata.get("header", "N/A")
        formatted.append(f"[Document {i}] Source: {source}, Header: {header}\nContent:\n{doc.page_content}")
    return "\n\n".join(formatted)

def run():
    print(BANNER)
    
    # 모델 및 체인 초기화
    llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}")
    ])
    chain = prompt_template | llm | StrOutputParser()

    while True:
        try:
            user_input = input(f"{Fore.GREEN}User:{Style.RESET_ALL} ").strip()

            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print(f"\n{Fore.YELLOW}Goodbye!{Style.RESET_ALL}")
                break

            print(f"{Fore.CYAN}[*] 지식베이스 검색 중...{Style.RESET_ALL}")
            
            # 1. 단일 검색 (Simple Hybrid Search)
            docs = query_hybrid(user_input, k=5, use_reranker=True)
            context = _format_docs(docs)
            
            print(f"{Fore.CYAN}[*] 답변 생성 중... (검색된 문서: {len(docs)}개){Style.RESET_ALL}")
            
            # 2. 단일 답변 생성 (Single Request)
            answer = chain.invoke({
                "question": user_input,
                "context": context
            })
            
            print(f"\n{Fore.YELLOW}Agent:{Style.RESET_ALL}\n{answer}\n")

        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Interrupted. Goodbye!{Style.RESET_ALL}")
            break
        except Exception as e:
            print(f"\n{Fore.RED}Error:{Style.RESET_ALL} {e}\n")

if __name__ == "__main__":
    run()
