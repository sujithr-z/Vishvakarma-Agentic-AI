"""LLM Client for local Ollama server."""
import os
import json
import urllib.request
import urllib.error
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen:1.8b")


class OllamaClient:
    """Minimal, robust client for local Ollama LLM execution."""

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_URL,
        model: str = DEFAULT_MODEL,
        timeout: int = 45,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def is_available(self) -> bool:
        """Check if Ollama server is reachable and model is available."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as res:
                if res.status == 200:
                    data = json.loads(res.read().decode("utf-8"))
                    models = [m.get("name") for m in data.get("models", [])]
                    # Check if model or model tag is in installed list
                    return any(self.model in m for m in models) or len(models) > 0
            return False
        except Exception:
            return False

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.1,
        model: Optional[str] = None,
    ) -> str:
        """
        Generate completion from Ollama.
        
        Args:
            prompt: Main prompt content.
            system: Optional system instruction.
            temperature: Sampling temperature (default 0.1 for deterministic JSON).
            model: Optional model override.
            
        Returns:
            Generated response string.
        """
        target_model = model or self.model
        payload = {
            "model": target_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "num_predict": 1024,
                "stop": ["\n\nUSER:", "\n\n### AGENT", "\n\n====="]
            },
        }
        if system:
            payload["system"] = system

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result.get("response", "").strip()
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"Failed to connect to Ollama at {self.base_url}: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Ollama generation error: {e}") from e
