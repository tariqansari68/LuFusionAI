import os
from anthropic import Anthropic
from dotenv import load_dotenv

# Load API keys securely from a local .env file
load_dotenv()

def ask_claude(prompt: str, model_name: str = "claude-3-5-sonnet-20241022"):
    """
    Sends a query to Anthropic's Claude API and returns the text response.
    """
    # Initialize the client (automatically grabs ANTHROPIC_API_KEY from environment)
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    try:
        message = client.messages.create(
            model=model_name,
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        # Extract and return the plain text response content
        return message.content.text
    except Exception as e:
        return f"Error connecting to Anthropic API: {str(e)}"

if __name__ == "__main__":
    test_prompt = "Explain how transformer attention layers process tokens."
    print("Sending request to Anthropic...")
    response = ask_claude(test_prompt)
    print(f"\n[Claude Response]:\n{response}")
