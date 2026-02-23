import os
from typing import Optional
import requests
from google import genai
from google.genai import types  # Required for function declaration
from dotenv import load_dotenv

# 1. Load Environment & Configure Client
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)


# 2. Real API Client (Keep this as is)
class DoctorApiClient:
    def __init__(self):
        self.base_url = "http://0.0.0.0:8000/api/v1/doctors"

    def get_doctor_by_id(self, doctor_id: int):
        response = requests.get(f"{self.base_url}/{doctor_id}")
        return response.json() if response.status_code == 200 else {"error": "Doctor not found"}

    def search_doctors(self, specialization: Optional[str] = None, location: Optional[str] = None,
                       min_fee: Optional[float] = None, max_fee: Optional[float] = None):
        params = {"specialization": specialization, "location": location, "min_fee": min_fee, "max_fee": max_fee}
        response = requests.get(f"{self.base_url}/search", params=params)
        return response.json() if response.status_code == 200 else {"error": "No doctors found"}

    def get_metadata_locations(self):
        response = requests.get(f"{self.base_url}/metadata/locations")
        return response.json() if response.status_code == 200 else {"error": "Locations not found"}

    def get_metadata_specializations(self):
        response = requests.get(f"{self.base_url}/metadata/specializations")
        return response.json() if response.status_code == 200 else {"error": "Specializations not found"}


# 3. Define the Tools for Gemini
# Note: In the new SDK, functions passed to tools are automatically handled if you use them correctly.
def fetch_doctor_details(doctor_id: int):
    """Get detailed information about a specific doctor by their ID."""
    return DoctorApiClient().get_doctor_by_id(doctor_id)


def find_doctors(specialization: Optional[str] = None, location: Optional[str] = None,
                 min_fee: Optional[float] = None, max_fee: Optional[float] = None):
    """Search for doctors based on specialization, city location, or price range."""
    return DoctorApiClient().search_doctors(specialization, location, min_fee, max_fee)


def list_available_locations():
    """Returns a list of all cities where doctors are available."""
    return DoctorApiClient().get_metadata_locations()


def list_available_specializations():
    """Returns a list of all available specializations."""
    return DoctorApiClient().get_metadata_specializations()


# 4. Create a Tool collection
tools = [fetch_doctor_details, find_doctors, list_available_locations, list_available_specializations]

# 5. Start the Agent (Using Gemini 2.0 Flash as seen in your output)
# We use automatic function calling via the config
# Change from 'gemini-2.0-flash' to 'gemini-1.5-flash'
# Use the exact string from your successful list_models() output
# Change your model_id variable to exactly this:
model_id = 'gemini-2.0-flash'



def run_agent(user_query: str):
    print(f"\nUser: {user_query}")

    # Send message with automatic tool calling enabled
    response = client.models.generate_content(
        model=model_id,
        contents=user_query,
        config=types.GenerateContentConfig(
            tools=tools,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=False)
        )
    )

    print(f"Agent: {response.text}")


# --- Test Runs ---
if __name__ == "__main__":
    # run_agent("I'm looking for a cardiologist in New York who charges less than 200 dollars.")
    run_agent("Can you give me more details about doctor number 2?")