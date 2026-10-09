from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
#this code is using the chat model instead of llm, 
#   bcz the newer updates force us to use chat models
llm = ChatOpenAI(model='gpt-6-luna')

response = llm.invoke("What is the capital of Pakistan?")

print(response.content)