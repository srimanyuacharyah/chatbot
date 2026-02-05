import os
from dotenv import load_dotenv
load_dotenv()
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage, AssistantMessage
from azure.core.credentials import AzureKeyCredential

# Configuration
endpoint = "https://models.github.ai/inference"
model = "gpt-4o-mini"
token = os.environ.get("GITHUB_TOKEN")

if not token:
    raise ValueError("GITHUB_TOKEN environment variable not set")

# Create client
client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token),
    model=model
)

# Conversation history
conversation_history = [
    SystemMessage(content="You are a helpful assistant.")
]

def get_ai_response(user_input):
    """Get response from AI"""
    conversation_history.append(UserMessage(content=user_input))
    
    try:
        # Get AI response
        response = client.complete(
            messages=conversation_history,
            temperature=1.0,
            top_p=1.0,
            max_tokens=1000,
            model=model
        )
        
        # Extract response
        ai_response = response.choices[0].message.content
        conversation_history.append(AssistantMessage(content=ai_response))
        return ai_response
        
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        ai_response = get_ai_response(user_input)
        print("AI:", ai_response)
