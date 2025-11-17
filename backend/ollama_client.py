import requests

def ask_ollama(model, prompt):
    """
    Send a prompt to a local Ollama model and return the response text.
    """
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(url, json=data)
    output = response.json()
    return output.get("response", "")
