from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI,set_tracing_disabled,RunConfig
from dotenv import load_dotenv
import os
import chainlit as cl
from  openai.types.responses import ResponseTextDeltaEvent

# Load environment variables
load_dotenv()

# Disable tracing if needed
set_tracing_disabled(True)

# Set up Gemini API client
gemini_api_key = os.getenv("GEMINI_API_KEY")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Set up model using the external Gemini client
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)


config = RunConfig(
    model=model,
    # model_provider=extarnal_client,
    tracing_disabled=True
)
agent = Agent(
    name="naina agent",
    instructions="You are a helpful assistant. Remember what the user tells you.",
    model=model,
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

    msg = cl.Message(content="")
    await msg.send()


    result = Runner.run_streamed(
        agent,
        input=history
    )


    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data,ResponseTextDeltaEvent):
            await msg.stream_token(event.data.delta)
    # await cl.Message(content=result.final_output).send()
 
# result = Runner.run_sync(
#     agent,
#     input="hello"
# )
# print(result.final_output)