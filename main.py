from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

from vector import retriever

model = OllamaLLM(model="llama3.2")

template = """
You are an expert in answering questions about a pizza restoraunt.

Here are some relevant reviews: {reviews}


Here is the question to answer: {question}
"""


prompt = ChatPromptTemplate.from_template(template)


chain = prompt | model 


while True:
    print("\n\n-------------------------")
    question_user = input("Ask your question (q to quit): ")
    print("\n\n")
    if question_user == "q":
        break

    reviews = retriever.invoke(question_user)

    result = chain.invoke({"reviews": reviews, "question": question_user})
    print(result)