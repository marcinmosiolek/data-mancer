import os
import threading
from abc import ABC, abstractmethod
from enum import Enum
from typing import Annotated, TypedDict, Dict
from typing import Any

import chainlit as cl
import pandas as pd
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import add_messages, StateGraph
from pydantic import BaseModel, Field
from tabulate import tabulate

generated_chart = None


class Task(Enum):
    PLOTING = "plotting"
    ANALYZING = "analyzing"
    GENERAL = "general"


class GraphState(TypedDict):
    messages: Annotated[list, add_messages]
    current_task: str
    current_code: str
    current_data_preview: str
    no_of_retries: int
    execution_error: str
    result_data_frame: str
    result_chart: str
    dispatched_task: Task


def extract_history(messages, history_len=20):
    return messages[-(history_len + 1): -1]


class GraphNode(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    async def arun(self, state: GraphState) -> Dict[str, Any]:
        pass


class BaseAgent(GraphNode):
    def __init__(self, prompt_file_name: str, llm: Any, output_structure: Any = None, name=None) -> None:
        super().__init__(name if name else prompt_file_name)
        self.llm = llm
        prompt_file_path = os.path.join("prompts", f"{prompt_file_name}.txt")

        try:
            with open(prompt_file_path, "r") as file:
                prompt_content = file.read()
            prompt_template = ChatPromptTemplate.from_template(prompt_content)
            if output_structure:
                self.chain = prompt_template | self.llm.with_structured_output(output_structure)
            else:
                self.chain = prompt_template | llm | StrOutputParser()
        except FileNotFoundError:
            raise ValueError(f"Prompt file '{prompt_file_path}' not found.")

    async def ainvoke(self, **kwargs: Any) -> Any:
        return await self.chain.ainvoke(kwargs)


class Contextualizer(BaseAgent):
    def __init__(self, llm):
        super().__init__(self.__class__.__name__, llm)

    async def arun(self, state: GraphState) -> Any:
        with cl.Step(name=self.name) as step:
            contextualized_question = await self.ainvoke(
                query=state["messages"][-1],
                history=extract_history(state["messages"]),
                data_preview=state["current_data_preview"]
            )
            step.output = contextualized_question

        return {"messages": AIMessage(contextualized_question), "current_task": contextualized_question}


class DispatchedTask(BaseModel):
    current_task: Task = Field(..., description="Task to be performed as decided by dispatcher")


class Dispatcher(BaseAgent):
    def __init__(self, llm):
        super().__init__(self.__class__.__name__, llm, DispatchedTask)

    async def arun(self, state: GraphState) -> Any:
        with cl.Step(name=self.name) as step:
            res = await self.ainvoke(
                task=state["messages"][-1],
                history=extract_history(state["messages"]),
                data_preview=state["current_data_preview"]
            )

            step.output = res.current_task

        return {"dispatched_task": res.current_task}


class GeneratedCode(BaseModel):
    code: str = Field(..., description="The generated code")


class CodeWriter(BaseAgent):
    def __init__(self, prompt_file_name: str, llm: Any, name) -> None:
        super().__init__(prompt_file_name, llm, GeneratedCode, name=name)

    async def arun(self, state: GraphState) -> Any:
        try:
            with cl.Step(name=self.name, language="python") as step:
                result = await super().ainvoke(task=state["current_task"], data_preview=state["current_data_preview"])
                step.output = result.code
            return {
                "messages": AIMessage(result.code),
                "current_code": result.code,
                "no_of_retries": 3,
                "execution_error": None
            }
        except Exception as e:
            return {"execution_error": str(e), "no_of_retries": state.get("no_of_retries", 0) - 1}


class PandasCodeWriter(CodeWriter):
    def __init__(self, llm):
        super().__init__(self.__class__.__name__, llm, name="Python")


class PlotlyCodeWriter(CodeWriter):
    def __init__(self, llm):
        super().__init__(self.__class__.__name__, llm, name="Plotly")


class CodeExplainer(BaseAgent):
    def __init__(self, llm):
        super().__init__(self.__class__.__name__, llm)

    async def arun(self, state: GraphState) -> Any:
        with cl.Step(name=self.name) as step:
            code_explanation = await self.ainvoke(code=state["current_code"], task=state["current_task"])
            step.output = code_explanation

        return {"messages": AIMessage(code_explanation)}


class Reporter(BaseAgent):
    def __init__(self, llm):
        super().__init__(self.__class__.__name__, llm)

    async def arun(self, state: GraphState) -> Any:
        with cl.Step(name=self.name) as step:
            answer = await self.ainvoke(table=state["result_data_frame"], task=state["current_task"])
            step.output = answer

        return {"messages": AIMessage(answer)}


class Generalist(BaseAgent):
    def __init__(self, llm):
        super().__init__(self.__class__.__name__, llm)

    async def arun(self, state: GraphState) -> Any:
        with cl.Step(name=self.name) as step:
            answer = await self.ainvoke(task=state["current_task"], history=extract_history(state["messages"]))
            step.output = answer

        return {"messages": AIMessage(answer)}


class CodeReviewer(BaseAgent):
    def __init__(self, llm):
        super().__init__(self.__class__.__name__, llm, GeneratedCode)

    async def arun(self, state: GraphState) -> Any:
        with cl.Step(name=self.name, language="python") as step:
            result = await super().ainvoke(
                task=state["current_task"],
                error=state["execution_error"],
                code=state["current_code"],
                data_preview=state["current_data_preview"]
            )
            step.output = result.code
        return {"messages": AIMessage(result.code), "current_code": result.code}


class CodeExecutor(GraphNode):
    def __init__(self, df):
        super().__init__(self.__class__.__name__)
        self.df = df

    async def arun(self, state: GraphState):
        with cl.Step(name=self.name) as step:
            exec_locals = {"df": self.df}
            try:
                exec(state["current_code"], {}, exec_locals)
                result = exec_locals.get("result")
                chart = exec_locals.get("chart")

                if result is None and chart is None:
                    raise ValueError("No 'result' or 'chart' was produced by the code execution.")

            except Exception as e:
                return {"error": str(e), "no_of_retries": state["no_of_retries"] - 1}

            if isinstance(result, pd.DataFrame):
                result_df = result.to_markdown()
                # Don't add large tables to history
                history_df = result.sample(20) if result.shape[0] > 20 else result
                history_df = tabulate(history_df, headers="keys", tablefmt="fancy_grid")
                print(history_df)

                step.output = result_df
                return {
                    "execution_error": None,
                    "result_data_frame": result_df,
                    "result_chart": None
                }
            elif chart:
                step.output = "Chart generated"
                return {
                    "execution_error": None,
                    "result_data_frame": None,
                    "result_chart": chart.to_json()
                }
            else:
                # In case result is not a DataFrame and chart is None
                step.output = str(result)
                return {
                    "messages": AIMessage(str(result)),
                    "execution_error": None,
                    "result_data_frame": None,
                    "result_chart": None
                }


class DataScientist(GraphNode):
    def __init__(self, llm, code_writer, df):
        super().__init__(f"{self.__class__.__name__}_{code_writer.name}")
        self.code_writer = code_writer
        self.code_executor = CodeExecutor(df)
        self.code_reviewer = CodeReviewer(llm)
        self.code_explainer = CodeExplainer(llm)

        graph_builder = StateGraph(GraphState)
        graph_builder.add_node(self.code_writer.name, self.code_writer.arun)
        graph_builder.add_node(self.code_executor.name, self.code_executor.arun)
        graph_builder.add_node(self.code_reviewer.name, self.code_reviewer.arun)
        graph_builder.add_node(self.code_explainer.name, self.code_explainer.arun)

        graph_builder.add_edge(START, self.code_writer.name)
        graph_builder.add_edge(self.code_writer.name, self.code_executor.name)
        graph_builder.add_conditional_edges(
            self.code_executor.name,
            self.is_working,
            {True: self.code_explainer.name, False: self.code_reviewer.name}
        )

        graph_builder.add_edge(self.code_reviewer.name, self.code_executor.name)
        graph_builder.add_edge(self.code_explainer.name, END)

        self.graph = graph_builder.compile()

    def is_working(self, state: GraphState) -> bool:
        return state["execution_error"] is None or state["no_of_retries"] <= 0

    async def arun(self, state: GraphState) -> Any:
        return await self.graph.ainvoke(state)


class DataConsulting(GraphNode):
    def __init__(self, llm, df):
        super().__init__(self.__class__.__name__)
        self.df = df
        self.data_scientist = DataScientist(llm, code_writer=PandasCodeWriter(llm), df=df)
        self.data_visualizer = DataScientist(llm, code_writer=PlotlyCodeWriter(llm), df=df)
        self.contextualizer = Contextualizer(llm)
        self.reporter = Reporter(llm)
        self.dispatcher = Dispatcher(llm)
        self.general = Generalist(llm)

        graph_builder = StateGraph(GraphState)
        graph_builder.add_node(self.data_scientist.name, self.data_scientist.arun)
        graph_builder.add_node(self.data_visualizer.name, self.data_visualizer.arun)
        graph_builder.add_node(self.contextualizer.name, self.contextualizer.arun)
        graph_builder.add_node(self.reporter.name, self.reporter.arun)
        graph_builder.add_node(self.dispatcher.name, self.dispatcher.arun)
        graph_builder.add_node(self.general.name, self.general.arun)

        graph_builder.add_edge(START, self.contextualizer.name)
        graph_builder.add_edge(self.contextualizer.name, self.dispatcher.name)
        graph_builder.add_conditional_edges(
            self.dispatcher.name,
            lambda x: x["dispatched_task"],
            {
                Task.PLOTING: self.data_visualizer.name,
                Task.ANALYZING: self.data_scientist.name,
                Task.GENERAL: self.general.name,
            }
        )

        graph_builder.add_edge(self.data_scientist.name, self.reporter.name)
        graph_builder.add_edge(self.reporter.name, END)
        graph_builder.add_edge(self.data_visualizer.name, END)
        graph_builder.add_edge(self.general.name, END)

        checkpointer = MemorySaver()
        self.graph = graph_builder.compile(checkpointer=checkpointer)

    async def arun(self, state: GraphState):
        state["current_data_preview"] = self.df.head(5).to_markdown()
        return await self.graph.ainvoke(state, config={"thread_id": threading.get_ident()})
