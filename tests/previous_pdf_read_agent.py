from transformers import pipeline


def initialize_llm():
    """Load the LLM once"""
    print("🧠 Loading TinyLlama model... (this may take a minute)")

    llm = pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_new_tokens=512,
        temperature=0.3,
        do_sample=True,
    )

    print("✅ Model loaded successfully!\n")
    return llm


def ask_question(llm, question: str) -> str:
    """Send a question to the LLM and return the answer"""
    prompt = (
        "<|system|>\n"
        "You are a helpful, concise assistant. "
        "Answer the user's question directly and clearly.\n"
        "<|user|>\n"
        f"{question}\n"
        "<|assistant|>\n"
    )

    result = llm(prompt)
    full_response = result[0]["generated_text"]

    # Extract only the assistant's reply
    if "<|assistant|>" in full_response:
        answer = full_response.split("<|assistant|>")[-1].strip()
    else:
        answer = full_response[len(prompt):].strip()

    return answer


def main():
    llm = initialize_llm()

    print("=" * 50)
    print("  📘 Terminal QA System")
    print("  Type your question and press Enter.")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 50)
    print()

    while True:
        try:
            question = input("❓ You: ").strip()

            if not question:
                continue

            if question.lower() in ("quit", "exit", "q"):
                print("👋 Goodbye!")
                break

            answer = ask_question(llm, question)
            print(f"\n💬 Answer: {answer}\n")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()