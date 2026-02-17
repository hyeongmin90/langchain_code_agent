import sys
import os
import asyncio
from dotenv import load_dotenv

# Add project root to sys.path to ensure module imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.crawler import fetch_spring_boot_docs
from data_pipeline.processor import split_text_with_llm_async
from data_pipeline.storage import add_documents

async def process_page(sem, page, log_dir):
    """
    Async task to process a single page:
    1. Split text with LLM (Async)
    2. Save logs
    3. Store in Vector DB
    """
    url_link = page['url']
    content = page['content']
    
    async with sem:
        print(f"  > [Start] Processing: {url_link}")
        
        # 2. Process (LLM Async)
        try:
            chunks = await split_text_with_llm_async(content)
        except Exception as e:
            print(f"  ! [Error] Failed to split {url_link}: {e}")
            return

        # Add metadata & Log to file
        log_filename = url_link.split('/')[-1]
        if not log_filename or log_filename.endswith('/'):
             path_parts = url_link.rstrip('/').split('/')
             log_filename = path_parts[-1] if path_parts else "index"
        
        # Sanitize filename
        log_filename = "".join([c for c in log_filename if c.isalpha() or c.isdigit() or c in (' ', '.', '_', '-')]).rstrip()
        log_path = os.path.join(log_dir, f"{log_filename}_chunks.txt")
        
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"Source: {url_link}\n")
            f.write(f"Total Chunks: {len(chunks)}\n")
            f.write("="*50 + "\n\n")
            
            for i, chunk in enumerate(chunks):
                chunk.metadata["source"] = url_link
                
                # Write to log
                f.write(f"=== Chunk {i+1} ===\n")
                f.write(f"[Summary]\n{chunk.page_content}\n\n")
                f.write(f"[Original Content]\n{chunk.metadata.get('original_content', '')}\n")
                f.write("-" * 50 + "\n\n")
        
        print(f"  - [Done] {len(chunks)} chunks from {url_link}")

        # 3. Store (Synchronous call, but fast enough or acceptable for now)
        if chunks:
            add_documents(chunks)

async def run_pipeline_async(url="https://docs.spring.io/spring-boot/reference/"):
    load_dotenv()
    
    print("=== Starting Async RAG Data Pipeline (Limit: 5) ===")
    
    # Directory for chunk logs
    log_dir = "llm_chunk_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Limit pages for testing if needed
    max_pages = None 
    print(f"Limiting to {max_pages} pages..." if max_pages else "No page limit set.")

    # Concurrency Control
    sem = asyncio.Semaphore(5)
    tasks = []

    # 1. Crawl (Generator) - Sync
    # We iterate the generator and spawn async tasks
    print("Fetching pages from crawler...")
    
    for i, page in enumerate(fetch_spring_boot_docs(url, max_pages=max_pages)):
        task = asyncio.create_task(process_page(sem, page, log_dir))
        tasks.append(task)
    
    if tasks:
        print(f"\nScheduled {len(tasks)} tasks. Waiting for completion...")
        await asyncio.gather(*tasks)
    else:
        print("No pages found or crawled.")

    print("\n=== Pipeline Completed ===")

if __name__ == "__main__":
    asyncio.run(run_pipeline_async())
