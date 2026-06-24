import pytest
import torch
from unittest.mock import MagicMock, patch
from claude_agent import RoutingConfig, OrchestrationRouter, PyTorchNativeProvider, ModelType, ModelResponse

def test_dynamic_routing_rules():
    config = RoutingConfig()
    config.add_rule(lambda p: "local-pytorch" if "compute" in p.lower() else None)
    
    mock_local = MagicMock()
    mock_cloud = MagicMock()
    
    providers = {"local-pytorch": mock_local, "claude-3-5-sonnet": mock_cloud}
    router = OrchestrationRouter(providers, config, fallback_model="claude-3-5-sonnet")
    
    assert router.route("Run a complex heavy data compute loop") == "local-pytorch"
    assert router.route("Hello world conversational prompt") == "claude-3-5-sonnet"

def test_pytorch_provider_mock_execution():
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    
    # Mock tokenizers returning dictionary formats matching transformers behavior
    mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
    mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
    mock_tokenizer.decode.return_value = "Generated text from PyTorch"
    
    provider = PyTorchNativeProvider(
        model_name="local-llama", 
        model_instance=mock_model, 
        tokenizer=mock_tokenizer
    )
    
    response = provider.generate("Test prompt context")
    assert response.text == "Generated text from PyTorch"
    assert response.provider == ModelType.PYTORCH
