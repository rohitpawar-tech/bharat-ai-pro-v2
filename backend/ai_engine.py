import requests
import json
import re
from ddgs import DDGS

# --- Configuration ---
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"

# --- Advanced Prompt Engineering ---
SYSTEM_PROMPT = """
You are Bharat AI Pro, a powerful AI assistant similar to ChatGPT.

You help users with:
* Programming
* DevOps
* Web development
* AI / Machine learning
* Math
* General questions

Rules:
* Give direct answers
* Avoid generic filler sentences
* Provide complete code examples
* Format code blocks properly using Markdown
* Be friendly and helpful.
"""

# --- Smart Greeting System ---
GREETINGS = {"hi", "hello", "hey", "greetings", "namaste"}
GREETING_RESPONSES = [
    "Hello! I am Bharat AI Pro. How can I assist you today?",
    "Hi there! Ready to help you with code, research, or questions.",
    "Hey! What can I do for you?"
]

# --- Response Cleanup System ---
FILLER_PHRASES = [
    "To accomplish this task,",
    "Regarding your query,",
    "In order to do this,",
    "Please note that",
    "It is important to mention"
]

def check_greeting(message: str):
    """Checks if the message is a simple greeting and returns a response locally."""
    msg_lower = message.strip().lower()
    if msg_lower in GREETINGS or msg_lower.startswith("hi ") or msg_lower.startswith("hello "):
        import random
        return random.choice(GREETING_RESPONSES)
    return None

def web_search(query: str) -> str:
    """Performs a DuckDuckGo search and returns formatted context."""
    try:
        results = []

        with DDGS() as ddgs:
            for res in ddgs.text(query, max_results=3):
                results.append(res)

        if not results:
            return ""

        context = "Web Search Results:\n"
        for i, res in enumerate(results):
            context += f"{i+1}. {res.get('title', '')}: {res.get('body', '')}\n"

        return context + "\n"

    except Exception as e:
        print(f"Search Error: {e}")
        return ""

def clean_response(text: str) -> str:
    """Removes generic filler phrases from the AI response."""
    for phrase in FILLER_PHRASES:
        if text.lower().startswith(phrase.lower()):
            text = text[len(phrase):].lstrip(", ")
    return text.strip()

def build_prompt(conversation_history: list, user_message: str, search_context: str) -> str:
    """Constructs the full prompt including system message, history, and context."""
    prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|>"

    # Conversation Memory System
    for msg in conversation_history:
        role = msg['role']  # 'user' or 'assistant'
        content = msg['content']
        prompt += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"

    # Web Search AI Integration
    if search_context:
        prompt += f"<|start_header_id|>system<|end_header_id|>\n\n{search_context}<|eot_id|>"

    prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{user_message}<|eot_id|>"
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"

    return prompt

def generate_stream_response(messages: list, user_message: str):
    """
    Generator function that yields chunks from Ollama.
    Handles Web Search, Memory, and Streaming.
    """
    # 1. Check Greeting
    greeting = check_greeting(user_message)
    if greeting:
        yield greeting
        return

    # 2. Perform Web Search
    search_context = web_search(user_message)

    # 3. Build Prompt
    full_prompt = build_prompt(messages, user_message, search_context)

    # 4. Ollama Payload
    payload = {
        "model": MODEL_NAME,
        "prompt": full_prompt,
        "stream": True,
        "options": {
            "temperature": 0.7,
            "num_ctx": 4096
        }
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=300)
        response.raise_for_status()

        full_text = ""
        for line in response.iter_lines():
            if line:
                json_data = json.loads(line)
                if "response" in json_data:
                    chunk = json_data["response"]
                    full_text += chunk
                    yield chunk

                if json_data.get("done", False):
                    break

    except requests.exceptions.RequestException as e:
        yield f"**Error connecting to Ollama:** {str(e)}"
    except Exception as e:
        yield f"**Unexpected error:** {str(e)}"