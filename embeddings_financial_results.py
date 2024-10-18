import openai
from dotenv import load_dotenv

load_dotenv()
openai_client = openai.OpenAI()

def generate_embedding(text: str)  -> list[float]:
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small",
    )

    return response.data[0].embedding

knowledge = [
    "Nie mogę się doczekać, aż ta prezentacja się skończy",
    "Ale się najem dzisiaj wieczorem",
    "Zysk za ostatni kwartał wynosił 100 zł"
]

question = "Ile moja firma zarobiła?"

knowledge_embeddings = [generate_embedding(document) for document in knowledge]
question_embedding = generate_embedding(question)

from sklearn.metrics.pairwise import cosine_similarity

results = cosine_similarity([question_embedding], knowledge_embeddings)
for question, result in zip(knowledge, results[0]):
    print(f"Wynik: {result:.3f}, pytanie: {question}")

