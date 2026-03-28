import os
import sys
import warnings
from dotenv import load_dotenv
from colorama import init, Fore, Style

warnings.filterwarnings("ignore", category=UserWarning, module=r"pydantic\..*")
warnings.filterwarnings("ignore", message="Pydantic serializer warnings", category=UserWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

load_dotenv()
init(autoreset=True)

from agent import build_graph
from agent.cache import SemanticCache


BANNER = f"""{Fore.CYAN}
╔══════════════════════════════════════════════════════╗
║         LangGraph RAG Agent  (Spring Docs)           ║
║         'exit' 또는 'q' 를 입력하면 종료됩니다       ║
╚══════════════════════════════════════════════════════╝
{Style.RESET_ALL}"""

CATEGORY_LABELS = {
    "spring-boot": "Spring Boot",
    "spring-data-jpa": "Spring Data JPA",
    "spring-data-redis": "Spring Data Redis",
    "spring-security": "Spring Security",
    "spring-cloud-gateway": "Spring Cloud Gateway",
    None: "전체",
}


def _print_step(label: str, value: str, color: str = Fore.CYAN) -> None:
    print(f"  {color}▶ {label}:{Style.RESET_ALL} {value}")


def run():
    graph = build_graph()
    cache = SemanticCache()

    if cache.available:
        print(f"{Fore.CYAN}[Cache] Redis 연결 성공 (semantic cache 활성화){Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}[Cache] Redis 연결 실패 - 캐시 없이 실행됩니다{Style.RESET_ALL}")

    print(BANNER)

    while True:
        try:
            user_input = input(f"{Fore.GREEN}User:{Style.RESET_ALL} ").strip()

            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print(f"\n{Fore.YELLOW}Goodbye!{Style.RESET_ALL}")
                break

            print()

            # 캐시 히트 확인
            cached_answer = cache.get(user_input)
            if cached_answer is not None:
                _print_step("캐시", "유사 질문 캐시 히트", Fore.GREEN)
                print(f"\n{Fore.YELLOW}Agent:{Style.RESET_ALL}\n{cached_answer}\n")
                continue

            state = {
                "question": user_input,
                "rewritten_query": None,
                "category": None,
                "should_rewrite": False,
                "documents": [],
                "answer": "",
            }

            final_answer = ""

            for step_output in graph.stream(state, stream_mode="updates"):
                node_name = list(step_output.keys())[0]
                node_result = step_output[node_name]

                if node_name == "analyze":
                    should_rw = node_result.get("should_rewrite", False)
                    category = node_result.get("category")
                    cat_label = CATEGORY_LABELS.get(category, str(category))

                    _print_step(
                        "분석",
                        f"카테고리={cat_label}  |  재작성={'필요' if should_rw else '불필요'}",
                        Fore.MAGENTA,
                    )

                elif node_name == "rewrite":
                    rewritten = node_result.get("rewritten_query", "")
                    _print_step("쿼리 재작성", rewritten, Fore.BLUE)

                elif node_name == "retrieve":
                    doc_count = len(node_result.get("documents", []))
                    _print_step("검색 완료", f"{doc_count}개 문서 검색됨", Fore.YELLOW)

                elif node_name == "generate":
                    final_answer = node_result.get("answer", "")
                    print(f"\n{Fore.YELLOW}Agent:{Style.RESET_ALL}\n{final_answer}\n")

            # 답변 생성 후 캐시 저장
            if final_answer:
                cache.set(user_input, final_answer)

        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Interrupted. Goodbye!{Style.RESET_ALL}")
            break
        except Exception as e:
            print(f"\n{Fore.RED}Error:{Style.RESET_ALL} {e}\n")


if __name__ == "__main__":
    run()
