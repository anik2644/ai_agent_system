import os
import json
import re
from typing import Optional
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from agent_system.find_params_user_query import extract_query_params

# 1. Load the Qwen Model
model_name = "Qwen/Qwen2.5-3B-Instruct"
print(f"Loading tokenizer...({model_name})")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,       # Changed from torch_dtype
    device_map="auto"
)
print("Model loaded successfully!\n")


# 2. Real API Client
class DoctorApiClient:
    def __init__(self):
        self.base_url = "http://0.0.0.0:8000/api/v1/doctors"

    def get_doctor_by_id(self, doctor_id: int):
        response = requests.get(f"{self.base_url}/{doctor_id}")
        return response.json() if response.status_code == 200 else {"error": "Doctor not found"}

    def search_doctors(self, specialization: Optional[str] = None, location: Optional[str] = None,
                       min_fee: Optional[float] = None, max_fee: Optional[float] = None):
        params = {k: v for k, v in {
            "specialization": specialization,
            "location": location,
            "min_fee": min_fee,
            "max_fee": max_fee
        }.items() if v is not None}
        response = requests.get(f"{self.base_url}/search", params=params)
        return response.json() if response.status_code == 200 else {"error": "No doctors found"}

    def get_metadata_locations(self):
        response = requests.get(f"{self.base_url}/metadata/locations")
        return response.json() if response.status_code == 200 else {"error": "Locations not found"}

    def get_metadata_specializations(self):
        response = requests.get(f"{self.base_url}/metadata/specializations")
        return response.json() if response.status_code == 200 else {"error": "Specializations not found"}


# 3. Define callable functions
def fetch_doctor_details(doctor_id: int):
    """Get detailed information about a specific doctor by their ID."""
    return DoctorApiClient().get_doctor_by_id(doctor_id)


def find_doctors(specialization: Optional[str] = None, location: Optional[str] = None,
                 min_fee: Optional[float] = None, max_fee: Optional[float] = None):
    """Search for doctors based on specialization, city location, or price range."""
    return DoctorApiClient().search_doctors(specialization, location, min_fee, max_fee)

# After (CORRECT) ✅
# def find_doctors(specialization=None, location=None, min_fee=None, max_fee=None, hospital_name=None):
#     return DoctorApiClient().search_doctors(specialization, location, min_fee, max_fee, hospital_name)



def list_available_locations():
    """Returns a list of all cities where doctors are available."""
    return DoctorApiClient().get_metadata_locations()


def list_available_specializations():
    """Returns a list of all available specializations."""
    return DoctorApiClient().get_metadata_specializations()


# 4. Function registry — maps function names to callables
FUNCTION_REGISTRY = {
    "fetch_doctor_details": fetch_doctor_details,
    "find_doctors": find_doctors,
    "list_available_locations": list_available_locations,
    "list_available_specializations": list_available_specializations,
}

# 5. Tool definitions in Qwen-compatible format (OpenAI-style)
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "fetch_doctor_details",
            "description": "Get detailed information about a specific doctor by their ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doctor_id": {
                        "type": "integer",
                        "description": "The unique ID of the doctor."
                    }
                },
                "required": ["doctor_id"]
            }
        }
    },

        {
            "type": "function",
            "function": {
                "name": "find_doctors ",
                "description": "Retrieve detailed information about a doctor, including their associated hospital.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "specialization": {
                            "type": "string",
                            "description": "The medical specialization to filter by (e.g., 'Cardiology')."
                        },
                        "location": {
                            "type": "string",
                            "description": "The city to filter by (e.g., 'New York')."
                        },
                        "min_fee": {
                            "type": "number",
                            "description": "Minimum consultation fee."
                        },
                        "max_fee": {
                            "type": "number",
                            "description": "Maximum consultation fee."
                        },
                        "hospital_name": {
                            "type": "string",
                            "description": "The name of the hospital to filter doctors by (e.g., 'Mayo Clinic')."
                        }
                    },
                    "required": ["hospital_name"]
                }
            }
        }
    ,
    {
        "type": "function",
        "function": {
            "name": "list_available_locations",
            "description": "Returns a list of all cities where doctors are available.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_available_specializations",
            "description": "Returns a list of all available medical specializations.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

SYSTEM_PROMPT = """You are a structured parameter extractor for a doctor search system in Bangladesh.

Your ONLY task: extract search parameters from the user's message and call the correct tool.

## Parameters to Extract:
- specialization : Medical specialty (e.g., Cardiology, Neurology, General Physician)
- location       : Area, thana, or neighborhood in Dhaka (e.g., Dhanmondi, Paltan, Uttara)
- hospital_name  : Hospital or clinic name (e.g., Popular Hospital, Square Hospital)
- min_fee        : Minimum consultation fee in taka (numeric only)
- max_fee        : Maximum consultation fee in taka (numeric only)

## Rules:
- ONLY extract what the user explicitly mentioned
- NEVER guess, assume, or fill in missing fields
- ALWAYS call find_doctors tool — never respond in plain text
- Normalize spelling errors (e.g., "Nurologyst" → "Neurology", "Phisician" → "General Physician")
- Translate Bengali terms (e.g., "Hridrog Bisheshagya" → "Cardiology")
- Strip filler words ("I need", "please", "near", "around") — extract only values

## Examples:
User: "Cardiologist at Sentral Polise Hospital Rajarbag, fees 500 to 2000"
→ call find_doctors(specialization="Cardiology", hospital_name="Central Police Hospital", location="Rajarbag", min_fee=500, max_fee=2000)

User: "Hridrog Bisheshagya dorkar New Market area te"
→ call find_doctors(specialization="Cardiology", location="New Market")

User: "Brain doctor at Poplar Hospital, max 1500 taka"
→ call find_doctors(specialization="Neurology", hospital_name="Popular Hospital", max_fee=1500)
"""

def generate_response(messages: list, max_new_tokens: int = 1024) -> str:
    """Generate a response from the Qwen model using chat template."""
    text = tokenizer.apply_chat_template(
        messages,
        tools=TOOLS_SCHEMA,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )

    # Decode only the newly generated tokens
    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return response_text.strip()


def resolve_tool_params(raw_args: dict) -> dict:
    """
    Takes raw tool call arguments from LLM output,
    resolves them via extract_query_params,
    and returns a clean final dictionary.
    """
    # Step 1: Extract relevant fields from LLM tool call
    raw_location     = raw_args.get("location")
    raw_specialization = raw_args.get("specialization")
    raw_hospital     = raw_args.get("hospital_name")
    min_fee          = raw_args.get("min_fee")
    max_fee          = raw_args.get("max_fee")

    # Step 2: Pass through resolvers to normalize/validate
    resolved = extract_query_params(
        location=raw_location,
        specialization=raw_specialization,
        hospital_name=raw_hospital,
    )

    # Step 3: Build final resolved dictionary
    final_params = {
        # Resolved & normalized values
        "location":       resolved.thana.resolved_value     if resolved.thana         else None,
        "specialization": resolved.specialization.resolved_value if resolved.specialization else None,
        "hospital_name":  resolved.hospital.resolved_value  if resolved.hospital      else None,

        # Fee fields pass through directly (no resolver needed)
        "min_fee":        min_fee,
        "max_fee":        max_fee,

        # Confidence metadata (optional but useful for debugging)
        "location_confidence":       resolved.thana.confident          if resolved.thana         else False,
        "specialization_confidence": resolved.specialization.confident if resolved.specialization else False,
        "hospital_confidence":       resolved.hospital.confident       if resolved.hospital       else False,

        # Bonus hospital metadata from resolver
        "hospital_thana": resolved.hospital_thana,
        "hospital_type":  resolved.hospital_type,
    }

    return final_params



def parse_tool_calls(response_text: str) -> list:
    """
    Parse tool/function calls from the model's response.
    Qwen2.5-Instruct uses a structured format for tool calls.
    We handle multiple possible formats.
    """
    tool_calls = []

    # Pattern 1: Qwen's native tool_call format
    # <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    tool_call_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
    matches = re.findall(tool_call_pattern, response_text, re.DOTALL)

    for match in matches:
        try:
            call = json.loads(match)
            name = call.get("name") or call.get("function", {}).get("name")
            arguments = call.get("arguments") or call.get("function", {}).get("arguments", {})
            if name:
                tool_calls.append({"name": name, "arguments": arguments})
        except json.JSONDecodeError:
            continue

    if tool_calls:
        return tool_calls

    # Pattern 2: Direct JSON function call format
    # {"name": "function_name", "arguments": {...}}
    json_pattern = r'\{\s*"name"\s*:\s*"(\w+)"\s*,\s*"arguments"\s*:\s*(\{.*?\})\s*\}'
    matches = re.findall(json_pattern, response_text, re.DOTALL)

    for name, args_str in matches:
        try:
            arguments = json.loads(args_str)
            if name in FUNCTION_REGISTRY:
                tool_calls.append({"name": name, "arguments": arguments})
        except json.JSONDecodeError:
            continue

    if tool_calls:
        return tool_calls

    # Pattern 3: Check for function names mentioned with JSON-like arguments
    for func_name in FUNCTION_REGISTRY:
        pattern = rf'{func_name}\s*\(\s*(\{{.*?\}})\s*\)'
        matches = re.findall(pattern, response_text, re.DOTALL)
        for args_str in matches:
            try:
                arguments = json.loads(args_str)
                tool_calls.append({"name": func_name, "arguments": arguments})
            except json.JSONDecodeError:
                continue

    return tool_calls


def execute_tool_call(tool_call: dict) -> dict:
    """Execute a parsed tool call and return the result."""
    func_name = tool_call["name"]
    arguments = tool_call["arguments"]

    if func_name not in FUNCTION_REGISTRY:
        return {"error": f"Unknown function: {func_name}"}

    func = FUNCTION_REGISTRY[func_name]

    try:
        # Convert argument types as needed
        if func_name == "fetch_doctor_details" and "doctor_id" in arguments:
            arguments["doctor_id"] = int(arguments["doctor_id"])
        if "min_fee" in arguments and arguments["min_fee"] is not None:
            arguments["min_fee"] = float(arguments["min_fee"])
        if "max_fee" in arguments and arguments["max_fee"] is not None:
            arguments["max_fee"] = float(arguments["max_fee"])

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        result = func(**arguments)
        return result
    except Exception as e:
        return {"error": f"Error executing {func_name}: {str(e)}"}


def run_agent(user_query: str, max_iterations: int = 1):
    """
    Run the agent loop:
    1. Send user query to model
    2. Check if model wants to call tools
    3. Execute tools and feed results back
    4. Repeat until model gives a final text answer
    """
    print(f"\nUser: {user_query}")
    print("-" * 60)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ]

    for iteration in range(max_iterations):
        print(f"\n[Iteration {iteration + 1}] Generating response...")

        response_text = generate_response(messages)
        print(f"[Raw Model Output]: {response_text[:500]}...")

        # print("\n\n--------------------------------\n\n")

        # Parse tool calls from the response
        tool_calls = parse_tool_calls(response_text)

        print(tool_calls)

        if not tool_calls:

            # print("\n\n no tool call\n\n")
            # No tool calls found — this is the final answer
            # Clean up any remaining artifacts
            final_answer = response_text
            # Remove any tool-call XML tags if present but unparseable
            final_answer = re.sub(r'<tool_call>.*?</tool_call>', '', final_answer, flags=re.DOTALL).strip()
            print(f"\nAgent: {final_answer}")
            return final_answer

        # Execute each tool call
        print(f"[Tool Calls Detected]: {len(tool_calls)}")

        final_params = resolve_tool_params(tool_calls[0]["arguments"])
        print(f"\n[Final Resolved Params]:")
        for key, value in final_params.items():
            print(f"  {key:30s} → {value}")
        # Add assistant message with tool calls
        # messages.append({"role": "assistant", "content": response_text})

        #
        # for tc in tool_calls:
        #     print(f"  -> Calling: {tc['name']}({tc['arguments']})")
        #     result = execute_tool_call(tc)
        #     result_str = json.dumps(result, indent=2, default=str)
        #     print(f"  <- Result: {result_str[:300]}...")
        #
        #     # Add tool result as a new message
        #     messages.append({
        #         "role": "tool",
        #         "name": tc["name"],
        #         "content": result_str
        #     })
        #


    # print("\n------------------------------------\n")
    # # If we exhausted iterations, generate a final response
    # print("\n[Max iterations reached, generating final response...]")
    # response_text = generate_response(messages)
    # print(f"\nAgent: {response_text}")
    # return response_text


# --- Interactive Mode ---
def interactive_mode():
    """Run the agent in an interactive chat loop."""
    print("=" * 60)
    print("Doctor Finder Agent (Qwen2.5-7B-Instruct)")
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 60)

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not user_input:
            continue
        run_agent(user_input)


# # --- Test Runs ---
# if __name__ == "__main__":
#     # Single query test
#     run_agent("Can you give me more details about doctor number 2?")
#
#     # Uncomment for more tests:
#     # run_agent("I'm looking for a cardiologist in New York who charges less than 200 dollars.")
#     # run_agent("What locations do you have doctors in?")
#     # run_agent("What specializations are available?")
#
#     # Uncomment for interactive mode:
#     # interactive_mode()

if __name__ == "__main__":
    # ✅ ALL FIELDS (specialization + location + min_fee + max_fee)
    run_agent("I need a cardiologist at Poplar Hospital New Merket Dhaka.")
    run_agent("I need a doctor at Rajarbag Polise Hospital.")

    run_agent("I need a Nurologyst in Dhaka.")
    run_agent("I need a Genral Phisician in Paltan.")
    run_agent("Search for Medicen doctors in Uttara.")


    run_agent("I need a physician in Rajarbag, Dhaka.")
    run_agent("Find neurologists in Purana Palten.")
    run_agent("Show me doctors in Azimpur.")
    run_agent("Find a cardiologist near Topkhana Road, Dhaka.")

    run_agent("Hridrog Bisheshagya dorkar New Market area te.")

    run_agent("I need a Brain and Nerve Specialist in Paltan area.")

    run_agent("Find a Hridoy Doctor at Populer Hospital New Merket with fees between 800 and 2000 taka.")
    run_agent("I need a Brain Docter at Sentral Police Hospitall Rajarbag, budget under 1500 taka.")

    run_agent("Find Neuro Surgeon near Rajarbagh area.")

    run_agent("Hello, I am looking for a good heart specialist doctor please in New Market Dhaka area if possible.")

    run_agent("Cardilogest at Sentral Polise Hospitol Rajarbag with fees 500 to 2000.")
    run_agent("Generel Fisician at Poplar Medikal Newmerket, max fee 700 taka.")


    # run_agent("Find me a cardiologist in Dhanmondi, Dhaka with consultation fees between 500 and 1500 taka.")
    # run_agent("Search for a dermatologist in Gulshan, Dhaka with fees ranging from 800 to 2000 taka.")
    #
    #
    # # ✅ ALL FIELDS (hospital_name + specialization + location + min_fee + max_fee)
    # run_agent("Find me a cardiologist at Square Hospital in Dhanmondi, Dhaka with fees between 1000 and 3000 taka.")
    # run_agent("Search for a dermatologist at United Hospital in Gulshan, Dhaka with fees ranging from 800 to 2500 taka.")
    # run_agent("I need a neurologist at Labaid Hospital in Dhanmondi, Dhaka, my budget is between 1000 and 3000 taka.")
    # run_agent("Find me a dentist available at Popular Diagnostic Centre in Dhanmondi, Dhaka.")
    # run_agent("Are there any cardiologists at Square Hospital in Mirpur, Dhaka?")


    # ===== Search by Specialization =====
    # run_agent("Find me all cardiologists available.")
    # run_agent("I need a dentist. Can you search for one?")
    # run_agent("Show me all dermatologists in your system.")
    # run_agent("Are there any neurologists available?")
    # run_agent("I'm looking for an orthopedic surgeon.")
    #
    # # ===== Search by Location =====
    # run_agent("Find doctors available in Dhaka.")
    # run_agent("Show me all doctors near Chittagong.")
    # run_agent("Are there any doctors in Sylhet?")
    #
    # # ===== Search by Fee Range =====
    # run_agent("Find me doctors who charge less than 500 taka.")
    # run_agent("Show me doctors with consultation fees between 300 and 800 taka.")
    # run_agent("I'm on a budget. Can you find doctors with fees under 400?")
    # run_agent("What are the most affordable doctors available?")
    #
    # # ===== Combined Filters (Specialization + Location) =====
    # run_agent("Find me a cardiologist in Dhaka.")
    # run_agent("I need a dentist in Chittagong. Any options?")
    # run_agent("Are there any dermatologists available in Rajshahi?")
    #
    # # ===== Combined Filters (Specialization + Fee Range) =====
    # run_agent("Find a cardiologist who charges less than 1000 taka.")
    # run_agent("I need a cheap dentist. Maybe under 500 taka?")
    # run_agent("Show me neurologists with fees between 500 and 1500.")
    #
    # # ===== Combined Filters (Location + Fee Range) =====
    # run_agent("Find doctors in Dhaka who charge under 700 taka.")
    # run_agent("Show me affordable doctors in Chittagong, max budget 600.")
    #
    # # ===== All Filters Combined (Specialization + Location + Fee) =====
    # run_agent("Find me a cardiologist in Dhaka with fees under 1000 taka.")
    # run_agent("I need a dentist in Sylhet, and my budget is between 300 and 700 taka.")
    # run_agent("Are there any dermatologists in Chittagong charging less than 800?")
    #
    # # ===== Conversational / Natural Language Queries =====
    # run_agent("I've been having chest pain lately. What kind of doctor should I see and can you find one for me?")
    # run_agent("My skin has been breaking out a lot. Help me find a skin specialist nearby in Dhaka.")
    # run_agent("I need a good but affordable eye doctor. My budget is around 500 taka.")
    # run_agent("My teeth hurt really bad. Find me the cheapest dentist available.")
    #
    # # ===== Edge Cases =====
    # run_agent("Find me a doctor.")  # No filters at all
    # run_agent("Search for a brain surgeon in Antarctica.")  # Non-existent location
    # run_agent("Find doctors with fees between 0 and 10 taka.")  # Unrealistic fee range
    # run_agent("Show me all available doctors with no specific preference.")
    #
    # # ===== Follow-up Style Queries =====
    # run_agent("Can you give me more details about doctor number 2?")
    # run_agent("From the previous results, book an appointment with the first doctor.")
    # run_agent("Tell me the qualifications of doctor number 3.")