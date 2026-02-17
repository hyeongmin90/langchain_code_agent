import sys
import os
from dotenv import load_dotenv

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessageChunk
from langgraph.checkpoint.memory import InMemorySaver
from colorama import init, Fore, Style

from data_pipeline.storage import query_documents

# Initialize Colorama
init(autoreset=True)

# Define the Tool
@tool
def search_spring_boot_docs(query: str) -> str:
    """
    Searches the Spring Boot reference documentation for relevant information.
    Use this tool to find answers to questions about Spring Boot configuration, features, and usage.
    
    Args:
        query: The search query string.
    """
    try:
        results = query_documents(query, k=5)
        
        if not results:
            return "No results found in the documentation."
            
        output = ""
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get("source", "Unknown")
            original = doc.metadata.get("original_content", "N/A")
            summary = doc.page_content
            
            output += f"--- Result {i} ---\n"
            output += f"Source: {source}\n"
            output += f"Summary: {summary}\n"
            output += f"Content: {original[:1000]}...\n" 
            output += "\n"
            
        return output
    except Exception as e:
        return f"Error during search: {e}"

def run_rag_agent():
    load_dotenv()
    
    # Initialize Model
    llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
    
    # Define Tools
    tools = [search_spring_boot_docs]
    
    # System Prompt (String)
    system_prompt = (
        "You are a Spring Boot Expert RAG Agent.\n"
        "Answer user questions accurately using the provided documentation.\n"
        "ALWAYS use the 'search_spring_boot_docs' tool to verify information before answering.\n"
        "If you cannot find the answer in the search results, admit it honestly.\n"
        "Provide clear, code-centric answers where applicable.\n"
        "Answer in Korean."
    )
    
    # Create Agent (Using user's custom/specific create_agent function if available in env)
    # The user insists on 'create_agent' which is likely imported from langchain.agents in their setup.
    try:
        agent = create_agent(
            model=llm,
            tools=tools,
            checkpointer=InMemorySaver(),
            system_prompt=system_prompt,
            debug=False
        )
    except ImportError:
        print("Error: 'create_agent' not found in langchain.agents. Please check your environment.")
        return

    print(f"{Fore.CYAN}=== Spring Boot RAG Agent (Type 'exit' to quit) ==={Style.RESET_ALL}")
    
    thread_id = "rag-cli-session"
    config = {"configurable": {"thread_id": thread_id}}
    
    while True:
        try:
            user_input = input(f"\n{Fore.GREEN}User: {Style.RESET_ALL}").strip()
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break
                
            if not user_input:
                continue
            
            print(f"\n{Fore.YELLOW}Agent:{Style.RESET_ALL} ", end="", flush=True)
            
            # Stream execution
            for event in agent.stream({"messages": [HumanMessage(content=user_input)]}, config, stream_mode="messages"):
                msg, _ = event
                
                # Print Text Content
                if isinstance(msg, AIMessageChunk) and msg.content:
                    # Skip tool calls if they are just chunks of arguments
                    if not msg.tool_call_chunks: 
                        print(msg.content, end="", flush=True)

                # Optional: Indicate Tool Usage
                # if msg.__class__.__name__ == 'ToolMessage':
                #    print(f"\n[Tool Result] {msg.content[:50]}...", end="")

            print() # Newline after response

        except KeyboardInterrupt:
            print("\nInterrupted.")
            break
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    run_rag_agent()
