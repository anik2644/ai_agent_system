import os
from google import genai
from dotenv import load_dotenv  # 1. Add this import

# 2. Add this to load the variables from your .env file
load_dotenv()

# Now this will actually find the key
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("Error: GOOGLE_API_KEY not found. Check your .env file formatting.")
else:
    client = genai.Client(api_key=api_key)

    try:
        for model in client.models.list():
            print(f"Model Found: {model.name}")
    except Exception as e:
        print(f"Connection successful, but API error: {e}")