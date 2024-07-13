

# Simple chatbot rules
def chatbot_response(user_input):
    user_input = user_input.lower()
    if "hi" in user_input or "hello" in user_input:
        return "Hello! How can I assist you?"
    elif "how are you" in user_input:
        return "I'm doing well, thank you! How about you?"
    elif "bye" in user_input:
        return "Goodbye! Have a great day!"
    else:
        return "I'm sorry, I didn't quite catch that."

# Main loop
if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        print("Chatbot:", chatbot_response(user_input))
