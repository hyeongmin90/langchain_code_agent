import os
import sys
from dotenv import load_dotenv

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_pipeline.crawler import get_content
from data_pipeline.processor import split_text_with_llm

def test_single_page(url):
    load_dotenv()
    
    print(f"=== Testing Single Page Chunking ===")
    print(f"Target URL: {url}")
    
    # 1. Fetch Content
    print("\n[Step 1] Fetching content...")
    content = get_content(url)
    
    if not content:
        print("Error: Failed to fetch content or content is empty.")
        return

    print(f"Fetched {len(content)} characters.")
    
    # 2. Split with LLM
    print("\n[Step 2] Splitting with LLM (gpt-5-mini)...")
    try:
        chunks = split_text_with_llm(content)
    except Exception as e:
        print(f"Error during splitting: {e}")
        return

    print(f"Generated {len(chunks)} chunks.")
    
    # 3. Output Results
    print("\n[Step 3] Chunk Details:")
    print("="*60)
    
    log_filename = "test_single_page_result.txt"
    with open(log_filename, "w", encoding="utf-8") as f:
        f.write(f"Source: {url}\n")
        f.write(f"Total Chunks: {len(chunks)}\n")
        f.write("="*60 + "\n\n")
        
        for i, chunk in enumerate(chunks):
            header = f"=== Chunk {i+1} ==="
            summary = f"[Summary]\n{chunk.page_content}"
            context = f"[Context]\n{chunk.metadata.get('context', '')}"
            original = f"[Original Content]\n{chunk.metadata.get('original_content', '')}"
            divider = "-"*60
            
            # Print to console (abbreviated)
            print(header)
            print(summary)
            print(context)
            print(f"[Original Content] ({len(chunk.metadata.get('original_content', ''))} chars)")
            print(divider)
            
            # Write to file (full)
            f.write(f"{header}\n{summary}\n\n{context}\n\n{original}\n{divider}\n\n")

    print(f"\n[Info] Full results saved to {log_filename}")

if __name__ == "__main__":
    # Default URL if not provided
    target_url = "https://docs.spring.io/spring-boot/reference/features/task-execution-and-scheduling.html"
    
    # Allow command line argument
    if len(sys.argv) > 1:
        target_url = sys.argv[1]
        
    test_single_page(target_url)
