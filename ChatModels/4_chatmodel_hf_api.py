from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

llm = HuggingFaceEndpoint(
    repo_id="zai-org/GLM-5.3",
    # provider="auto",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

response = model.invoke("What is the capital of Bangladesh?")

print(response.content)
