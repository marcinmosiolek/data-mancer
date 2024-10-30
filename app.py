import chainlit as cl
import pandas as pd
import plotly.io as pio

from agents import DataConsulting
from llm import llm


@cl.on_chat_start
async def start():
    """
    Handler for chat start event.
    Prompts the user to upload an Excel file and initializes the agent.
    """
    files = None

    # Wait for the user to upload a file
    while not files:
        files = await cl.AskFileMessage(
            content="Please upload an Excel file to analyze.", accept=[".xlsx"]
        ).send()

    text_file = files[0]

    # Read the Excel file into a pandas DataFrame
    df = pd.read_excel(text_file.path)

    # Initialize the agent with the LLM and DataFrame
    agent = DataConsulting(llm, df)
    cl.user_session.set("agent", agent)

    # Notify the user that the file was loaded successfully
    await cl.Message(
        content=f"{df.head().to_markdown()}\n\nThe file was loaded successfully. Let me know what to do!"
    ).send()


def create_new_task(message: cl.Message) -> dict:
    """
    Creates a new task dictionary from the user's message.
    """
    return {
        "messages": [("human", message.content)],
        "current_data_frame": None,
        "current_code": None,
        "execution_error": None,
        "no_of_retries": 3,
        "current_task": message.content,
        "result_chart": None,
        "result_data_frame": None,
    }


@cl.on_message
async def on_message(message: cl.Message):
    """
    Handler for incoming user messages.
    Processes the message using the agent and sends back the response.
    """
    agent = cl.user_session.get("agent")
    if not agent:
        await cl.Message(content="Agent not initialized. Please restart the chat.").send()
        return

    res = await agent.arun(create_new_task(message))

    if not res:
        await cl.Message(content="No response from agent.").send()
        return

    # Handle DataFrame result
    if res.get("result_data_frame"):
        answer = res["result_data_frame"] + "\n\n" + res["messages"][-1].content
        await cl.Message(content=answer).send()
    # Handle Chart result
    elif res.get("result_chart"):
        figure = pio.from_json(res["result_chart"])
        elements = [cl.Plotly(name="Plotly Chart", figure=figure, display="inline")]
        await cl.Message(content="", elements=elements).send()
    # Handle general messages
    else:
        await cl.Message(content=res["messages"][-1].content).send()