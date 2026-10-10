from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

text = "Delhi is the capital of India"

vector = embedding.embed_query(text)
#for documents, a similar method is used as in file 2 of embeddings
print(str(vector))