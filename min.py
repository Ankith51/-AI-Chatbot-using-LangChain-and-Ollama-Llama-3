from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Define the prompt template
template = """
Answer the question below.

Here is the conversation history:
{context}

Question: {question}

Answer:
"""

# Load the model
model = OllamaLLM(model="llama3:2b")  # Use correct model notation
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def handle_conversation():
    context = ""
    print("Welcome to the AI chatbot! Type 'exit' to quit.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        # Invoke the model with conversation history
        response = chain.invoke({"context": context, "question": user_input})

        # Extract and print the response
        bot_response = response if isinstance(response, str) else response.get("text", "I'm not sure.")
        print("Bot:", bot_response)

        # Append to conversation history
        context += f"\nUser: {user_input}\nAI: {bot_response}"

if __name__ == "__main__":
    handle_conversation()
