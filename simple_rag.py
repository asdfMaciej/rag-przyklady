from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

import os
os.environ["OPENAI_API_KEY"] = "api-key"

knowledge_base = ["strawberry color is blue"]
# Stworzenie embedingów z bazy wiedzy
vectorstore = Chroma.from_texts(texts=knowledge_base, embedding=OpenAIEmbeddings())
# Stworzenie retriever'a używającego embedingów
retriever = vectorstore.as_retriever()

prompt = PromptTemplate.from_template(
    """Answer the question based solely on the provided context.
     If the context does not provide necessary information, answer 'I don't know.'
     Context: {context}
     Question: {question}
     """
)

# RunnablePassthrough - przekaże to co przekażmy do chain poprzez .invoke
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | ChatOpenAI(model="gpt-4o")
    | StrOutputParser()
)

print("What color is strawberry? ->", rag_chain.invoke("What color is strawberry?"))
print("What color is banana? ->", rag_chain.invoke("What color is banana?"))
