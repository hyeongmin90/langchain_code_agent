import os
import sys
import uuid
from dotenv import load_dotenv

# 프로젝트 루트를 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langsmith import Client
from langsmith.evaluation import evaluate
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from pipeline.retriever import query_hybrid
from openevals.prompts import CORRECTNESS_PROMPT, RAG_GROUNDEDNESS_PROMPT, RAG_RETRIEVAL_RELEVANCE_PROMPT
from openevals.llm import create_llm_as_judge
from agent.graph import build_graph

load_dotenv()

# 평가 심사위원(Judge)으로 사용할 LLM 설정
judge_llm = ChatOpenAI(model="gpt-5-mini", temperature=0)

# ==========================================
# 0. 에이전트 초기화
# ==========================================

# 1) Agentic RAG (기존 고도화된 에이전트)
agentic_rag = build_graph()

# 2) Simple RAG (단순 검색 후 답변 체인)
def get_simple_rag_chain():
    llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
    prompt = PromptTemplate.from_template(
        """You are a Spring Boot Expert. Answer the user question accurately using only the provided context.
If you don't know the answer based on the context, say that you don't know.
answer in korean language.

Question: {question}

Context:
{context}

Answer:"""
    )
    return prompt | llm

simple_rag_chain = get_simple_rag_chain()

# ==========================================
# 1. 평가 대상 함수 (Target Functions)
# ==========================================

def predict_agentic_rag(inputs: dict) -> dict:
    """고도화된 LangGraph 에이전트 평가용 함수"""
    question = inputs["question"]
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    # 그래프 실행
    result = agentic_rag.invoke({"question": question}, config)
    
    # AgentState에서 결과 추출 (agent/state.py 참고)
    prediction = result.get("answer", "No answer generated.")
    # 검색된 컨텍스트 수집 (nodes.py에서 retrieved_docs에 저장한다고 가정)
    docs = result.get("documents", [])

    context = "\n\n".join([doc.page_content for doc in docs])
            
    return {
        "prediction": prediction,
        "context": context
    }

def predict_simple_rag(inputs: dict) -> dict:
    """단순 검색 후 답변 체인 평가용 함수"""
    question = inputs["question"]
    
    # 1. 검색 (Retrieve)
    docs = query_hybrid(question, k=5, use_reranker=True)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # 2. 생성 (Generate)
    result = simple_rag_chain.invoke({"question": question, "context": context})
    
    return {
        "prediction": result.content,
        "context": context
    }

# ==========================================
# 2. 평가 지표 (Evaluators)
# ==========================================

qa_evaluator = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT,
    judge=judge_llm,
    feedback_key="correctness",
    continuous=True,
    use_reasoning=False
)

context_evaluator = create_llm_as_judge(
    prompt=RAG_GROUNDEDNESS_PROMPT,
    judge=judge_llm,
    feedback_key="groundedness",
    continuous=True,
    use_reasoning=False
)

retrieval_evaluator = create_llm_as_judge(
    prompt=RAG_RETRIEVAL_RELEVANCE_PROMPT,
    judge=judge_llm,
    feedback_key="retrieval_relevance",
    continuous=True,
    use_reasoning=False
)

def correctness(run, example):
    return qa_evaluator(
        inputs=example.inputs["question"],
        reference_outputs=example.outputs["answer"],
        outputs=run.outputs["prediction"]
    )

def groundedness(run, example):
    return context_evaluator(
        context=run.outputs["context"],
        outputs=run.outputs["prediction"]
    )

def retrieval_relevance(run, example):
    return retrieval_evaluator(
        context=run.outputs["context"],
        inputs=example.inputs["question"]
    )

# ==========================================
# 3. 평가 실행
# ==========================================

def run_evaluation():
    dataset_name = "eval_dataset_hard_short_v1_spring_docs"
    # dataset_name = "evl_test_dataset"

    
    client = Client()
    try:
        client.read_dataset(dataset_name=dataset_name)
    except Exception:
        print(f"데이터셋 {dataset_name}을 찾을 수 없습니다.")
        return

    print(f"=== '{dataset_name}' 데이터셋을 이용한 비교 평가 시작 ===")
    
    evaluators = [correctness, groundedness, retrieval_relevance]

    # 1. Agentic RAG 평가
    print("\n1. Agentic RAG (Advanced Agent) 평가 중...")
    evaluate(
        predict_agentic_rag,
        data=dataset_name,
        evaluators=evaluators,
        experiment_prefix="Agentic-RAG-Hard_v2",
        max_concurrency=5
    )

    # 2. Simple RAG 평가
    print("\n2. Simple RAG (Basic Chain) 평가 중...")
    evaluate(
        predict_simple_rag,
        data=dataset_name,
        evaluators=evaluators,
        experiment_prefix="Simple-RAG-Hard",
        max_concurrency=5
    )
    
    print("\n✅ 모든 평가가 완료되었습니다. LangSmith에서 결과를 비교해보세요.")

if __name__ == "__main__":
    run_evaluation()
