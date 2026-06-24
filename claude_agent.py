import os
import time
import enum
import httpx
import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from dotenv import load_dotenv

load_dotenv()

class ModelType(enum.Enum):
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
    PYTORCH = "pytorch"

@dataclass
class ModelResponse:
    text: str
    model_used: str
    provider: ModelType
    latency_seconds: float
    raw_response: Any = None
    usage: Dict[str, int] = field(default_factory=dict)

# --- Base Provider ---
class AIModelProvider(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> ModelResponse:
        pass

# --- Claude Cloud Engine ---
class ClaudeCloudProvider(AIModelProvider):
    def __init__(self, model_name: str = "claude-3-5-sonnet"):
        super().__init__(model_name)
        import anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else None

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> ModelResponse:
        if not self.client:
            raise ValueError("ANTHROPIC_API_KEY environment variable is missing.")
        start_time = time.perf_counter()
        
        message_params = {
            "model": self.model_name,
            "max_tokens": kwargs.get("max_tokens", 1024),
            "messages": [{"role": "user", "content": prompt}]
        }
        if system_prompt:
            message_params["system"] = system_prompt

        response = self.client.messages.create(**message_params)
        return ModelResponse(
            text=response.content[0].text if hasattr(response.content, '__getitem__') else response.content.text,
            model_used=self.model_name,
            provider=ModelType.ANTHROPIC,
            latency_seconds=time.perf_counter() - start_time,
            raw_response=response,
            usage={"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
        )

# --- Local Ollama Engine ---
class OllamaLocalProvider(AIModelProvider):
    def __init__(self, model_name: str, base_url: str = "http://localhost:11434"):
        super().__init__(model_name)
        self.base_url = base_url

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> ModelResponse:
        start_time = time.perf_counter()
        payload = {"model": self.model_name, "prompt": prompt, "stream": False}
        if system_prompt:
            payload["system"] = system_prompt
            
        with httpx.Client(base_url=self.base_url) as client:
            response = client.post("/api/generate", json=payload, timeout=60.0)
        response.raise_for_status()
        data = response.json()
        
        return ModelResponse(
            text=data.get("response", ""),
            model_used=self.model_name,
            provider=ModelType.OLLAMA,
            latency_seconds=time.perf_counter() - start_time,
            raw_response=data
        )

# --- Local Native PyTorch Transformer Engine ---
class PyTorchNativeProvider(AIModelProvider):
    def __init__(self, model_name: str, model_instance: Any = None, tokenizer: Any = None):
        super().__init__(model_name)
        self.model = model_instance
        self.tokenizer = tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.model:
            self.model.to(self.device).eval()

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> ModelResponse:
        if not self.model or not self.tokenizer:
            raise ValueError("PyTorch model instance or tokenizer was not initialized.")
            
        start_time = time.perf_counter()
        full_prompt = f"{system_prompt}\n{prompt}" if system_prompt else prompt
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=kwargs.get("max_length", 128))
            
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return ModelResponse(
            text=generated_text,
            model_used=self.model_name,
            provider=ModelType.PYTORCH,
            latency_seconds=time.perf_counter() - start_time,
            raw_response=outputs
        )

# --- Core Router Orchestration Layer ---
class RoutingConfig:
    def __init__(self):
        self.rules: List[Callable[[str], Optional[str]]] = []

    def add_rule(self, rule_func: Callable[[str], Optional[str]]):
        self.rules.append(rule_func)

class OrchestrationRouter:
    def __init__(self, providers: Dict[str, AIModelProvider], routing_config: RoutingConfig, fallback_model: str):
        self.providers = providers
        self.routing_config = routing_config
        self.fallback_model = fallback_model

    def route(self, prompt: str) -> str:
        for rule in self.routing_config.rules:
            model = rule(prompt)
            if model and model in self.providers:
                return model
        return self.fallback_model

    def execute(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> ModelResponse:
        model_name = self.route(prompt)
        provider = self.providers.get(model_name, self.providers[self.fallback_model])
        
        try:
            return provider.generate(prompt, system_prompt, **kwargs)
        except Exception as e:
            print(f"Fallback Active: {model_name} failed due to: {str(e)}")
            return self.providers[self.fallback_model].generate(prompt, system_prompt, **kwargs)
