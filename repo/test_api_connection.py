"""
Test API connection to verify if the 503 error is real or a configuration issue.
"""
import os
from openai import OpenAI

# Read API key from environment or use placeholder
api_key = os.environ.get("OPENAI_API_KEY", "sk-xxx")

print(f"Testing API connection...")
print(f"Base URL: https://az.gptplus5.com/v1")
print(f"API Key: {api_key[:10]}..." if len(api_key) > 10 else "API Key: NOT SET")
print("-" * 60)

try:
    client = OpenAI(
        base_url='https://az.gptplus5.com/v1',
        api_key=api_key
    )
    
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ]
    )
    
    print("[OK] API connection successful!")
    print(f"Response: {completion.choices[0].message}")
    print(f"Model: {completion.model}")
    print(f"Usage: {completion.usage}")
    
except Exception as e:
    print(f"[ERROR] API connection failed!")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    print()
    print("Troubleshooting:")
    if "401" in str(e):
        print("- 401 Unauthorized: API key is invalid or missing")
        print("- Set environment variable: $env:OPENAI_API_KEY='your-actual-key'")
    elif "503" in str(e):
        print("- 503 Service Unavailable: API server is down or overloaded")
        print("- Try again later or contact API provider")
    else:
        print("- Check network connection")
        print("- Verify base_url is correct")
