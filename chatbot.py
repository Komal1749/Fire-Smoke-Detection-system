import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is missing")

client = Groq(api_key=api_key)


def ask_groq(question):

    messages = [
        {
            "role": "system",
            "content": """
You are a helpful AI assistant integrated into a
Fire and Smoke Detection project.

Answer the user's questions clearly and simply.

You can answer general questions as well as questions
about AI, Machine Learning, Deep Learning, CNN,
MobileNetV2, Fire Detection and Smoke Detection.

Do not claim that you can detect fire from text.
For actual image detection, the project's CNN/MobileNetV2
model is used.
"""
        },
        {
            "role": "user",
            "content": question
        }
    ]

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages,
        temperature=0.7,
        max_tokens=500
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    print("Testing Groq...")

    answer = ask_groq("What is CNN?")

    print("\nANSWER:")
    print(answer)