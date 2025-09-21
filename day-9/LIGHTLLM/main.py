from agents import Agent, Runner ,set_tracing_disabled
from dotenv import load_dotenv
import os
import chainlit as cl
from agents.extensions.models.litellm_model import LitellmModel
# Load environment variables
load_dotenv()

# Disable tracing if needed
set_tracing_disabled(True)

# Set up Gemini API client
gemini_api_key = os.getenv("GEMINI_API_KEY")


Model = "gemini/gemini-2.0-flash"


agent = Agent(
    name="naina agent",
    instructions="You are a helpful assistant. Remember what the user tells you.",
    model=LitellmModel(api_key=gemini_api_key,model=Model),
      # ← MEMORY SET HERE
)

@cl.on_chat_start
async def hendel_chat():
    cl.user_session.set("history",[])
    await cl.Message(content="hello how can i help you").send()

@cl.on_message

async def hendel_message(message : cl.Message):

    history = cl.user_session.get("history")
    history.append({"role": "user","content": message.content})
    result = await Runner.run(
        agent,
        input=history
    )
    await cl.Message(content=result.final_output).send()
 
