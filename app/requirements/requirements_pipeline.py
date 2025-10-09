import os
import json
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
import requirements_analyzer
import decompose_into_user_stories


def main():
    load_dotenv()
    first_request = input("👉 ")
    user_question_result = user_qustion_gen.main(first_request)
    user_stories = decompose_into_user_stories.main(user_question_result)
    requirements_analysis_result = requirements_analyzer.main(user_stories)
    print("--------------------------------")
    print(user_question_result)
    print(requirements_analysis_result["functional_requirements"])
    print("--------------------------------")

if __name__ == "__main__":
    main()