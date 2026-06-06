import os
from google import genai
from tavily import TavilyClient

# Replace with your NEW keys
GENAI_KEY = "AIzaSyBrqD9MQxqhBgHnO_FiEk89HjY1h1QeHHM"
TAVILY_KEY = "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"

def test():
    print("--- 1. Testing Gemini Brain ---")
    try:
        client = genai.Client(api_key=GENAI_KEY)
        res = client.models.generate_content(model="gemini-2.5-flash", contents="Hi")
        print(f"Gemini Success: {res.text}")
    except Exception as e:
        print(f"Gemini FAILED: {e}")

    print("\n--- 2. Testing Tavily Researcher ---")
    try:
        tavily = TavilyClient(api_key=TAVILY_KEY)
        res = tavily.search(query="Top jobs in Malaysia 2026")
        print(f"Tavily Success: Found {len(res['results'])} results")
    except Exception as e:
        print(f"Tavily FAILED: {e}")

if __name__ == "__main__":
    test()