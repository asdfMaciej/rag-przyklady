import openai
from dotenv import load_dotenv

load_dotenv()
openai_client = openai.OpenAI()

def call_llm(messages: list) -> str:
    chat_completion = openai_client.chat.completions.create(
        messages=messages,
        model='gpt-4o-mini',
        temperature=0.2
    )

    return chat_completion.choices[0].message.content

CHAT_HISTORY = [
    {"role": "user", "content": "Czy są dostępne kierunki związane z fryzjerstwem?"},
    {"role": "assistant", "content": "Tak: 1. Fryzjerstwo z certyfikatem, 2. Stylizacja fryzur"},
]
USER_QUERY = "czym się różnią?"

SYSTEM_PROMPT = "You will receive a conversation between an user and an assistant.\n"
SYSTEM_PROMPT += "Respond with an unambiguous version of the last message.\n"
SYSTEM_PROMPT += "For example, if the user was talking about planting a tree, "
SYSTEM_PROMPT += 'you might convert "how long will it take?" to "how long will it take to plant a tree?".'

LLM_MESSAGES = [
    {"role": "system", "content": SYSTEM_PROMPT},
    *CHAT_HISTORY,
    {"role": "user", "content": USER_QUERY}
]

rewritten_query = call_llm(LLM_MESSAGES)
print(f"Previous query: {USER_QUERY}")
print(f"Rewritten query: {rewritten_query}")

