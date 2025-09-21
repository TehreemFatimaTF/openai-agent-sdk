from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from agents import (
    input_guardrail,
    RunContextWrapper,
    TResponseInputItem,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered
)
import chainlit as cl

# Load environment variables
load_dotenv()
set_tracing_disabled(disabled=True)


# Get Gemini API Key
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Setup the OpenAI Gemini client
client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Define the model
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=client
)

# Output schema using Pydantic
class OutputPython(BaseModel):
    is_python_ralated: bool
    reasoning: str

# Input guardrail agent
input_guardrails = Agent(
    name="Input Guardrails Checker",
    instructions="Check if the user's question is related to Python programming. If it is, return true; otherwise, return false.",
    model=model,
    output_type=OutputPython
)

# Guardrail function
@input_guardrail
async def input_guardrails_func(
    ctx: RunContextWrapper,
    agent: Agent,
    input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    result = await Runner.run(input_guardrails, input)
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=not result.final_output.is_python_ralated
    )

# Main agent
main_agent = Agent(
    name="Python Expert Agent",
    instructions="You are a Python expert agent. You only respond to Python-related questions.",
    model=model,
    input_guardrails=[input_guardrails_func]
)

# Chat start message
@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="👋 I am ready to assist you with Python questions.").send()

# Chat message handler
@cl.on_message
async def on_message(message: cl.Message):
    try:
        result = await Runner.run(
            main_agent,
            input=message.content
        )
        await cl.Message(content=str(result.final_output)).send()

    except InputGuardrailTripwireTriggered:
        await cl.Message(content="⚠️ Please ask a Python-related question.").send()
