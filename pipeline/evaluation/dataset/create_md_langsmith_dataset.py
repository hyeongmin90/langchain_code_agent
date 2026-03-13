import os
import sys
import random
import concurrent.futures
import warnings
from tqdm import tqdm
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module=r"pydantic\..*")
# warnings.filterwarnings("ignore", message="Pydantic serializer warnings", category=UserWarning)

# 프로젝트 루트를 경로에 추가 (pipeline/evaluation/dataset/ -> Root)
ROOT_DIR = Path(__file__).parent.parent.parent.parent
sys.path.append(str(ROOT_DIR))

from langsmith import Client
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# Import storage to get VectorStore directly
from pipeline.storage import get_vectorstore

# 환경 변수 로드
load_dotenv()

class QAPair(BaseModel):
    question: str = Field(description="Generated question strictly based on the text")
    answer: str = Field(description="Ground truth answer to the question explicitly stated in the text")

class QAPairs(BaseModel):
    pairs: List[QAPair] = Field(description="List of QA pairs. Can be empty if the text lacks sufficient information.")

def generate_qa_pairs_from_chunk(content: str, max_pairs: int = 1) -> List[QAPair]:
    """
    LLM을 사용하여 텍스트에서 고난도 Q&A 쌍을 추출합니다.
    어휘적 중복을 최소화하고 추상화된 질문을 생성하도록 유도합니다.
    """
    
    llm = ChatOpenAI(model="gpt-5-mini", temperature=0.4)

    prompt = """
    당신은 Retrieval 시스템의 성능을 극한으로 테스트하기 위한 **최상위 난이도(Very Hard)** 데이터셋 생성 전문가입니다.

    웹 문서에서 추출된 하나의 텍스트 청크가 주어집니다.
    이 청크의 내용을 기반으로 최대 {max_pairs}개의 질문과 답변(Q&A) 쌍을 생성하세요.

    이 데이터셋의 목적은 **"사용자가 아주 짧고 불친절하게 질문했을 때, 시스템이 의도를 파악하여 정확한 문서를 찾아낼 수 있는가"**를 평가하는 것입니다.

    다음 지침을 반드시 따르세요.

    [질문 생성 핵심 규칙 - 필수]
    1. **극단적 간결함 (Extreme Conciseness):** 질문은 반드시 5~10단어 이내의 짧은 문장이어야 합니다. 상세한 설명이나 부연 설명을 모두 제거하세요.
    2. **키워드 제거 (Keyword Stripping):** 텍스트에 등장하는 핵심 기술 용어, 클래스명, 메서드명, 프로퍼티명을 질문에 직접 노출하지 마세요. (예: 'Redis' -> '인메모리 저장소', 'WebSecurityConfigurerAdapter' -> '보안 설정 방식')
    3. **추상화 및 모호함 (Abstraction & Ambiguity):** 실제 사용자가 채팅창에 툭 던지는 듯한 말투를 사용하세요. 구체적인 상황을 설명하기보다 "어떻게 하나요?", "~의 차이가 뭔가요?"와 같이 핵심 의도만 짧게 담으세요.
    4. **의미적 재구성:** 텍스트의 문장 구조를 그대로 따르지 말고, 사용자가 가질만한 근본적인 의문이나 문제 상황으로 재구성하세요.

    [문서 품질 평가]
    1. 텍스트가 너무 짧거나(300자 미만 권장 제외 로직은 외부에서 처리됨) 내용이 부실하면 질문을 생성하지 마세요.

    [답변 작성 방식]
    1. 답변은 반드시 제공된 텍스트 안에서 명확하게 확인할 수 있어야 합니다. (외부 지식 금지)
    2. 답변은 질문의 의도에 맞게 핵심만 간결하고 전문적인 톤으로 작성하세요.

    [출력 형태]
    질문과 답변은 반드시 **영어(English)**로 작성해야 합니다. 
    (실제 평가는 영문 검색 성능을 위주로 진행하므로)

    Text:
    {text}
    """
    
    prompt_template = PromptTemplate.from_template(prompt)
    chain = prompt_template | llm.with_structured_output(QAPairs)
    
    try:
        # 텍스트가 너무 길 경우 자름
        truncated_content = content[:15000] if len(content) > 15000 else content
        result = chain.invoke({"text": truncated_content, "max_pairs": max_pairs})
        
        return result.pairs[:max_pairs]
    except Exception as e:
        print(f"Q&A 생성 중 오류 발생: {e}")
        return []

def create_dataset_from_crawled_md(
    collection_name: str = "spring_docs", 
    num_samples: int = 50, 
    max_pairs_per_chunk: int = 1
):
    print(f"=== VectorStore({collection_name}) 기반 고난도 평가용 데이터셋 구축 시작 (대상: {num_samples}개 청크) ===")
    
    # 1. DB에서 문서 가져오기
    vectorstore = get_vectorstore(collection_name)
    db_data = vectorstore.get()
    
    documents = db_data.get('documents', [])
    metadatas = db_data.get('metadatas', [])
    ids = db_data.get('ids', [])
    
    if not documents:
        print(f"오류: '{collection_name}' 컬렉션에 데이터가 없습니다. create_test_vectorstores.py를 먼저 실행하세요.")
        return
        
    print(f"VectorStore에서 총 {len(documents)}개의 청크를 발견했습니다.")

    # 품질 좋은 청크 선별 (길이 300자 이상)
    viable_indices = [i for i, doc in enumerate(documents) if len(doc) > 300]
    print(f"품질 검사를 통과한 청크(300자 이상): {len(viable_indices)}개")

    if not viable_indices:
        print("질문을 생성하기에 충분한 길이의 청크가 없습니다.")
        return

    # 랜덤 샘플링
    actual_samples = min(num_samples, len(viable_indices))
    sampled_indices = random.sample(viable_indices, actual_samples)
    print(f"이 중 {actual_samples}개의 청크를 랜덤으로 추출하여 QA 생성을 진행합니다.")
    
    client = Client()
    
    # 데이터셋 이름 업데이트 (고난도 버전임을 명시)
    dataset_name_eng = f"eval_dataset_hard_short_v1_{collection_name}"
    try:
        dataset = client.read_dataset(dataset_name=dataset_name_eng)
        print(f"데이터셋 '{dataset_name_eng}'이(가) 이미 존재합니다. 해당 데이터셋에 추가합니다.")
    except Exception:
        # 데이터셋이 없는 경우 새로 생성
        dataset = client.create_dataset(
            dataset_name=dataset_name_eng,
            description=f"[{collection_name}] 고난도(Lexical Isolation, Abstraction 적용) RAG 평가용 데이터셋"
        )
        print(f"LangSmith 데이터셋 '{dataset_name_eng}' 생성 완료.")

    dataset_records = []


    def process_chunk(idx):
        content = documents[idx]
        metadata = metadatas[idx] if metadatas else {}
        doc_id = ids[idx] if ids else f"chunk_{idx}"
        
        qa_pairs = generate_qa_pairs_from_chunk(content, max_pairs=max_pairs_per_chunk)
        
        all_qa_pairs = []
        for pair in qa_pairs:
            all_qa_pairs.append({
                "question": pair.question,
                "answer": pair.answer,
                "chunk_content": content,
                "source": metadata.get("source", "Unknown"),
                "header": metadata.get("header", "Unknown"),
                "chunk_id": doc_id
            })
            
        return all_qa_pairs

    # 병렬 처리로 속도 향상
    print(f"LLM을 사용하여 Q&A 쌍을 평가 및 생성하는 중... (청크당 최대 {max_pairs_per_chunk}개)")
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_chunk = {executor.submit(process_chunk, idx): idx for idx in sampled_indices}

        for future in tqdm(concurrent.futures.as_completed(future_to_chunk), total=len(sampled_indices), desc="Generating Q&A"):
            try:
                qa_data = future.result()
                if not qa_data:
                    continue
                
                for item in qa_data:
                    dataset_records.append({
                        "question": item["question"],
                        "expected_answer": item["answer"],
                        "context": item["chunk_content"],
                        "source": f"{item['source']}"
                    })
            except Exception as e:
                print(f"병렬 처리 중 예외 발생: {e}")

    if not dataset_records:
        print("경고: 생성된 Q&A 쌍이 하나도 없습니다. 문서 내용이 모두 부실하거나 오류가 발생했을 수 있습니다.")
        return

    print(f"\n총 {len(dataset_records)}개의 Q&A 쌍이 성공적으로 생성되었습니다.")
    print("LangSmith에 업로드를 시작합니다...")
    
    for record in tqdm(dataset_records, desc="Uploading to LangSmith"):
        try:
            client.create_example(
                inputs={
                    "question": record["question"]
                },
                outputs={
                    "answer": record["expected_answer"],
                    "context": record["context"], # 정답 비교 및 검증에 원본 컨텍스트 사용 가능
                    "source": record["source"]
                },
                dataset_name=dataset_name_eng
            )
        except Exception as e:
            print(f"업로드 중 오류 발생: {e}")
            
    print(f"\n데이터셋 구축이 완료되었습니다!")
    print(f"LangSmith Dashboard에서 '{dataset_name_eng}' 데이터셋을 확인하세요.")

if __name__ == "__main__":
    # 데이터셋 구성 인자: num_samples=50 (50개 청크 랜덤추출), max_pairs_per_chunk=1
    create_dataset_from_crawled_md(num_samples=50, max_pairs_per_chunk=1)
