import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from claude_agent import PyTorchNativeProvider, RoutingConfig, OrchestrationRouter, ClaudeCloudProvider, OllamaLocalProvider

def initialize_pytorch_engine(model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"):
    """
    Initializes a highly memory-efficient, lightweight local PyTorch transformer.
    Falls back gracefully to CPU if a CUDA/MPS GPU is not available.
    """
    print(f"Initializing local PyTorch pipeline using weights from: {model_id}")
    
    # 1. Detect hardware architecture acceleration
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
        print("Hardware Acceleration Found: Using NVIDIA CUDA GPU.")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
        print("Hardware Acceleration Found: Using Apple Silicon MPS GPU.")
    else:
        device = "cpu"
        dtype = torch.float32
        print("Hardware Acceleration Status: None. Falling back to native CPU.")

    # 2. Safely initialize Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=dtype,
        trust_remote_code=True
    )
    
    # 3. Instantiate the unified provider wrapper
    pytorch_provider = PyTorchNativeProvider(
        model_name=model_id,
        model_instance=model,
        tokenizer=tokenizer
    )
    return pytorch_provider

def run_orchestrated_pipeline():
    """
    Sets up the central routing system, configures custom rules, 
    and handles automatic fallbacks across all 3 layers.
    """
    # Initialize our functional engines
    # Note: Anthropic key will look for your .env setup; Ollama checks localhost
    cloud_provider = ClaudeCloudProvider(model_name="claude-3-5-sonnet")
    ollama_provider = OllamaLocalProvider(model_name="qwen3-coder:latest")
    
    try:
        pytorch_provider = initialize_pytorch_engine()
    except Exception as e:
        print(f"Skipping native PyTorch setup due to offline mode or lack of memory: {e}")
        pytorch_provider = None

    # Assemble the active provider mapping dictionary
    providers = {
        "claude-3-5-sonnet": cloud_provider,
        "qwen3-coder:latest": ollama_provider
    }
    if pytorch_provider:
        providers[pytorch_provider.model_name] = pytorch_provider

    # Define Intelligent Routing Rules based on task description
    config = RoutingConfig()
    
    # Rule A: Math or custom formatting goes to your highly customized PyTorch model
    if pytorch_provider:
        config.add_rule(lambda prompt: pytorch_provider.model_name if "compute" in prompt.lower() or "math" in prompt.lower() else None)
    
    # Rule B: Simple short programmatic functions run locally via Ollama to avoid cloud API billing
    config.add_rule(lambda prompt: "qwen3-coder:latest" if len(prompt) < 40 and "code" in prompt.lower() else None)

    # Instantiate the Orchestrator Router (Claude serves as the final robust fallback layer)
    router = OrchestrationRouter(providers, routing_config=config, fallback_model="claude-3-5-sonnet")

    # --- Test Drive the Engine ---
    print("\n--- Running Live Orchestrated Test Executions ---")
    
    # Test 1: Should fire Ollama
    resp1 = router.execute("Write a python code snippet for sorting arrays.")
    print(f"Prompt 1 Model Selected: {resp1.model_used} | Output: {resp1.text[:60]}...")

    # Test 2: Should fire Claude Cloud (Complex reasoning, long, non-code)
    resp2 = router.execute("Please write a detailed 500-word historical essay outlining architectural paradigms during the Renaissance period.")
    print(f"Prompt 2 Model Selected: {resp2.model_used} | Output: {resp2.text[:60]}...")

if __name__ == "__main__":
    run_orchestrated_pipeline()
