import os
import textwrap
from flask_cors import CORS
from flask import Flask, request, jsonify
from langchain.memory import ConversationBufferMemory
from quering_vector_database import model_query

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/chat": {"origin": "*"}})

# Initialize LangChain components
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# System prompt to set AI behavior
SYSTEM_PROMPT = "Act as a virtual assistant for the platform's users. Provide clear explanation about iGameZ's concept, and guidance through the platform."

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = data.get("message")

    try:
        chat_history = memory.chat_memory.messages

        # If it's the first message, activate the AI agent prompt
        if not chat_history:
            full_message = f"{SYSTEM_PROMPT}\n\nUser: {message}"
        else:
            full_message = message

        bot_reply = model_query(full_message)
        filtred_bot_reply = getattr(bot_reply, "content", bot_reply)

        
        # Update memory with the conversation context
        memory.save_context({"input": message}, {"output": filtred_bot_reply})

        return jsonify({"reply": filtred_bot_reply})

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": "Failed to fetch response"}), 500

app.run(port=5000, debug=True)