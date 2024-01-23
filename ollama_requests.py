import requests

url = "http://localhost:11434/api/generate"
headers = {'Content-Type': 'application/json'}


def generate_response(prompt: str,
                      model: str = 'llama2',
                      temp: float = 0,
                      verbose: bool = False,
                      ):
    """
    Uses the ollama API to generate a response.
    This functions uses the option raw=True so, the provided prompt must a full prompt (with the special tokens).
    """
    if verbose:
        print(prompt)

    data = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "raw": True,
        "options": {
            "temperature": temp
        }
    }

    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes

        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None


def generate_response_default_tm(system: str,
                                 prompt: str,
                                 model: str = 'llama2',
                                 temp: float = 0
                                 ):
    """
    Uses the ollama API to generate a response. This functions uses the default template (in the modelfile.md)
    """
    data = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {
            "temperature": temp
        }
    }

    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes

        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None

