import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3"

def query_ollama(prompt: str, temperature: float = 0.7) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "temperature": temperature
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        print(f"[Ollama Error] Ensure Ollama is running: {e}")
        return "ERROR"