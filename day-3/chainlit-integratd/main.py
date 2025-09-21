from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI,set_tracing_disabled,RunConfig,function_tool,WebSearchTool
from dotenv import load_dotenv
import os
import chainlit as cl

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
@function_tool
def get_user_data(min_age: int) -> list[dict]:
    "Retrieve user data based on a minimum age"
    users = [
        {"name": "Muneeb", "age": 22},
        {"name": "Muhammad Ubaid Hussain", "age": 25},
        {"name": "Azan", "age": 19},
    ]

    for user in users:
        if user["age"] < min_age:
            users.remove(user)

    return users



rishtey_wali_agent = Agent(
    name="Auntie",
    model="gpt-4o-mini",
    instructions="You are a warm and wise 'Rishtey Wali Auntie' who helps people find matches",
    tools=[get_user_data, WebSearchTool()]   # WebSearchTool will only work with OpenAI API key, if you want to use any other free use "browser-use"
)

@cl.on_message

async def hendel_message(
    message : cl.Message):

    result = await Runner.run(
        starting_agent=rishtey_wali_agent,
        input=message.content
    )
    await cl.Message(content=result.final_output).send()
 
