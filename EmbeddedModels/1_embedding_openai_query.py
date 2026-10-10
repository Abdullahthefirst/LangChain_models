from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)
#setting higher dimension value of vector leads to capturing more context
#smaller dimension vector results in less cost
#

response = embedding.embed_query("Delhi is the capital of India")

print(str(response))