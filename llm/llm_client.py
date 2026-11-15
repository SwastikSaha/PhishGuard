
import requests


OLLAMA_URL = "http://localhost:11434/api/chat"

MODEL_NAME = "qwen2.5:0.5b"

TIMEOUT = 120


class OllamaClient:

    def __init__(
        self,
        model: str = MODEL_NAME,
        timeout: int = TIMEOUT,
    ):
        self.model = model
        self.timeout = timeout

        self.session = requests.Session()

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.2,
                "top_p": 0.9,
            },
        }

        try:
            response = self.session.post(
                OLLAMA_URL,
                json=payload,
                timeout=self.timeout,
            )

            response.raise_for_status()

            data = response.json()

            message = data.get("message", {})
            content = message.get("content")

            if not isinstance(content, str) or not content.strip():
                raise RuntimeError(
                    "Ollama returned an empty response."
                )

            return content

        except requests.ConnectionError as error:
            raise RuntimeError(
                "Cannot connect to Ollama. "
                "Check whether the Ollama server is running."
            ) from error

        except requests.Timeout as error:
            raise RuntimeError(
                "Ollama inference timed out."
            ) from error

        except requests.RequestException as error:
            raise RuntimeError(
                f"Ollama API request failed: {error}"
            ) from error

        except ValueError as error:
            raise RuntimeError(
                "Ollama returned an invalid JSON response."
            ) from error

