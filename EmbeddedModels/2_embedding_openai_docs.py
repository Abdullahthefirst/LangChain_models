from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)
#setting higher dimension value of vector leads to capturing more context
#smaller dimension vector results in less cost
#

documents = [
    "Delhi is the capital of India",
    "Islamabad is the capital of Pakistan", 
    "What is the oil price in Pakistan and India?"

]

response = embedding.embed_documents(documents)

print(str(response))
print('-------------------------------------------------')
print(response)
#the response contains list of lists
