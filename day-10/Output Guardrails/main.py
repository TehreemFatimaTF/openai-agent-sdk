from agents import (
    Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled,
    input_guardrail, output_guardrail,
    RunContextWrapper, TResponseInputItem,
    GuardrailFunctionOutput, InputGuardrailTripwireTriggered
)
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
from pydantic import BaseModel
import chainlit as cl

# Load environment variables
load_dotenv()
set_tracing_disabled(disabled=True)

# Setup Gemini model
gemini_api_key = os.getenv("GEMINI_API_KEY")
client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=client
)

# Output schema
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

@input_guardrail
async def input_guardrails_func(ctx: RunContextWrapper, agent: Agent, input: str | list[TResponseInputItem]) -> GuardrailFunctionOutput:
    result = await Runner.run(input_guardrails, input)
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=not result.final_output.is_python_ralated
    )

# Output guardrail
class outputMessage(BaseModel):
    response: str

class pythonOutput(BaseModel):
    is_python: bool
    reasoning: str

output_guard = Agent(
    name="Output Agent",
    instructions="Check if the output is Python-related and explain.",
    output_type=pythonOutput,
    model=model
)

@output_guardrail
async def python_guarrails(ctx: RunContextWrapper, agent: Agent, output: outputMessage) -> GuardrailFunctionOutput:
    result = await Runner.run(output_guard, output)
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=not result.final_output.is_python
    )

# Main agent
main_agent = Agent(
    name="Python Expert Agent",
    instructions="You are a Python expert agent. You only respond to Python-related questions.",
    model=model,
    input_guardrails=[input_guardrails_func],
    output_guardrails=[python_guarrails]  # Optional
)

# Chainlit Chat Events
@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="👋 I am ready to assist you with Python questions.").send()

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
