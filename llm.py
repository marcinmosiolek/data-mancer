import os

from langchain_openai import ChatOpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("The OPENAI_API_KEY environment variable is not set.")

llm = ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY)
