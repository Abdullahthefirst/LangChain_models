from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()
document = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = "Tell me about bumrah"


embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=300)

document_embeddings = embedding.embed_documents(document)

query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], document_embeddings)
scores = scores[0]

#arguments of cosine_similarity() should both be 2D list 
print(scores)
print('-----------------------------------')
print(list(enumerate(scores)))
#the enumerate number reserves the index
score_index = sorted(list(enumerate(scores)), key=lambda x: x[1])

best_index, best_score = score_index[-1] #the last one holding the highest score
print(query)
print(document[best_index])
print("similarity score is:", best_score)