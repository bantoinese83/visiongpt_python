from openai import OpenAI


class TextGenerator:
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)
        self.conversation_history = []

    def clear_conversation_history(self):
        self.conversation_history.clear()

    def generate_text(self, user_text, max_tokens=300):
        # Add the user's message to the conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_text
        })

        # Generate the AI's response
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=self.conversation_history,
            max_tokens=max_tokens,
        )

        # Add the AI's response to the conversation history
        self.conversation_history.append({
            "role": "assistant",
            "content": response.choices[0].message.content
        })

        # Return the actual text of the AI's response
        return response.choices[0].message.content
