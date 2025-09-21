from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, set_trace_processors
from dotenv import load_dotenv
from agents.tracing.processors import ConsoleSpanExporter, BatchTraceProcessor, default_processor
import os

# Load environment variables
load_dotenv()

# Trace processors setup
export = ConsoleSpanExporter()
processor = BatchTraceProcessor(export)

# ✅ Call default_processor() to create the instance
set_trace_processors([processor, default_processor()])

gemini_api_key = os.getenv("GEMINI_API_KEY")

# Set up Gemini API client
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Set up model using the external Gemini client
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

agent = Agent(
    name="naina agent",
    instructions="You are a helpful assistant. Remember what the user tells you.",
    model=model,
)

result = Runner.run_sync(
    agent,
    input="hello"
)

print(result.final_output)
