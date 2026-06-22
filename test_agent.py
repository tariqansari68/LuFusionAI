import unittest
from unittest.mock import patch, MagicMock
import torch
from train import initialize_model
from claude_agent import ask_claude

class TestLuLuFusionAI(unittest.TestCase):

    def test_hardware_detection(self):
        """Ensure hardware initialization returns a valid device string."""
        device = initialize_model()
        self.assertIn(device, ["cuda", "mps", "cpu"])

    @patch('claude_agent.Anthropic')
    def test_claude_agent_mock(self, mock_anthropic):
        """Test Claude agent without spending real API credits or exposing keys."""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        mock_client.messages.create.return_value.content.text = "Mocked Response"
        
        response = ask_claude("Hello", model_name="claude-3-5-sonnet-20241022")
        self.assertEqual(response, "Mocked Response")

if __name__ == '__main__':
    unittest.main()
