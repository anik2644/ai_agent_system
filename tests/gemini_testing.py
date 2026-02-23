import os
import google.generativeai as genai
from typing import Optional
import requests
from decimal import Decimal

# 1. Configuration (Replace with your actual API key)
genai.configure(api_key="AIzaSyBdwBBr3kvhAeYlfcTfy6x6SeWnI-sGR0k")


# 2. Real API Client
class DoctorApiClient:
    def __init__(self):
        self.base_url = "http://0.0.0.0:8000/api/v1/doctors"

    def get_doctor_by_id(self, doctor_id: int):
        """Fetch a doctor by their ID."""
        response = requests.get(f"{self.base_url}/{doctor_id}")
        if response.status_code == 200:
            return response.json()  # assuming JSON response
        return {"error": "Doctor not found"}

    def search_doctors(self, specialization: Optional[str] = None,
                       location: Optional[str] = None,
                       min_fee: Optional[float] = None,
                       max_fee: Optional[float] = None,
                       page: int = 0, size: int = 10):
        """Search doctors based on given filters."""
        params = {
            "specialization": specialization,
            "location": location,
            "min_fee": min_fee,
            "max_fee": max_fee,
            "page": page,
            "size": size
        }
        response = requests.get(f"{self.base_url}/search", params=params)
        if response.status_code == 200:
            return response.json()  # assuming JSON response
        return {"error": "No doctors found"}

    def get_metadata_locations(self):
        """Fetch metadata for available locations."""
        response = requests.get(f"{self.base_url}/metadata/locations")
        if response.status_code == 200:
            return response.json()  # assuming JSON response
        return {"error": "Locations not found"}

    def get_metadata_specializations(self):
        """Fetch metadata for available specializations."""
        response = requests.get(f"{self.base_url}/metadata/specializations")
        if response.status_code == 200:
            return response.json()  # assuming JSON response
        return {"error": "Specializations not found"}


# 3. Define the Tools for Gemini
def fetch_doctor_details(doctor_id: int):
    """Get detailed information about a specific doctor by their ID."""
    client = DoctorApiClient()
    return client.get_doctor_by_id(doctor_id)


def find_doctors(specialization: Optional[str] = None,
                 location: Optional[str] = None,
                 min_fee: Optional[float] = None,
                 max_fee: Optional[float] = None):
    """Search for doctors based on specialization, city location, or price range."""
    client = DoctorApiClient()
    return client.search_doctors(specialization=specialization, location=location,
                                 min_fee=min_fee, max_fee=max_fee)


def list_available_locations():
    """Returns a list of all cities where doctors are available."""
    client = DoctorApiClient()
    return client.get_metadata_locations()


def list_available_specializations():
    """Returns a list of all available specializations."""
    client = DoctorApiClient()
    return client.get_metadata_specializations()


# 4. Initialize the Agent (Ensure Gemini or other LLM API is available)
tools = [fetch_doctor_details, find_doctors, list_available_locations, list_available_specializations]
model = genai.GenerativeModel(
    model_name='gemini-1.5-flash',  # Use the appropriate model for your case
    tools=tools
)

# 5. Start the Agentic Loop
chat = model.start_chat(enable_automatic_function_calling=True)


def run_agent(user_query: str):
    """Run the agent with a given user query."""
    print(f"\nUser: {user_query}")
    response = chat.send_message(user_query)
    print(f"Agent: {response.text}")


# --- Test Runs ---
if __name__ == "__main__":
    # Example 1: Extraction of Params
    run_agent("I'm looking for a cardiologist in New York who charges less than 200 dollars.")

    # Example 2: Specific ID
    run_agent("Can you give me more details about doctor number 45?")

    # Example 3: Metadata query for locations
    run_agent("Which cities do you have doctors in?")

    # Example 4: Metadata query for specializations
    run_agent("What are the available specializations for doctors?")