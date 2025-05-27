import os
import asyncio
import aiohttp
from dotenv import load_dotenv

load_dotenv()

async def test_azure_config():
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key = os.environ.get("AZURE_OPENAI_KEY")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
    
    print(f"Endpoint: {endpoint}")
    print(f"API Key: {'*' * 5 + api_key[-5:] if api_key else 'NOT SET'}")
    print(f"Deployment: {deployment}")
    
    if not endpoint.endswith('/'):
        endpoint += '/'
    
    # Use a valid API version - this is crucial
    url = f"{endpoint}openai/deployments/{deployment}/chat/completions?api-version=2024-05-01-preview"
    print(f"Full URL: {url}")
    
    # Actual connection test
    try:
        headers = {
            "Content-Type": "application/json",
            "api-key": api_key,
        }
        
        payload = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."}, 
                {"role": "user", "content": "Hello, this is a test."}
            ],
            "max_tokens": 100
        }
        
        print("Attempting to connect to Azure OpenAI...")
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    print("✅ Connection successful!")
                    print(f"Response: {data['choices'][0]['message']['content']}")
                    return True
                else:
                    error_text = await response.text()
                    print(f"❌ Connection failed with status {response.status}")
                    print(f"Error details: {error_text}")
                    return False
    except Exception as e:
        print(f"❌ Exception occurred: {str(e)}")
        return False

if __name__ == "__main__":
    asyncio.run(test_azure_config())