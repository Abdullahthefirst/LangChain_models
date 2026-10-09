from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
#you'll need dependencies libraries

# 1. Initialize local model pipeline with valid dictionary syntax
llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
        "temperature": 0.5,
        "max_new_tokens": 100,
        "do_sample": True,
    },
)

# 2. Wrap pipeline in ChatHuggingFace
model = ChatHuggingFace(llm=llm)

# 3. Invoke and print the generated content
response = model.invoke("What is capital of Pakistan?")
print(response.content)