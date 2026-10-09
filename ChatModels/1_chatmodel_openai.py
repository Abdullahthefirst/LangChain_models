from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-6-luna', max_completion_tokens=50)

response = model.invoke("What is capital of Pakistan?")

# print(response)
# print('-----------------------------------')
print(response.content)
#the main unique feature of chat models is 
#structured message output, instead of just string, role based
#you use chat models 90% of the time