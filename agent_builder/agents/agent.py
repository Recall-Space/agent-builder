"""
An Agent class used for stateful agent interaction with tools and
LLMs (Language Learning Models).
"""

from textwrap import dedent
from typing import Any, Callable
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor
import logging


class Agent:
    """
    A stateful agent designed for interacting with LLMs and performing
    tasks with a toolset. The agent retains chat history, and its
    behavior is shaped by the user's input, goal, and previous interactions.

    Attributes
    ----------
    llm : object
        The base chat model from LangChain.
    user_name : str
        The user who is interacting with the agent.
    action_control : MultiActionAgentOutputParser
        The class responsible for parsing the output of an LLM call.
    format_intermediate_steps : Callable
        A function to convert (AgentAction, tool output) tuples into
        FunctionMessages for further processing.
    goal : str
        The main purpose or objective of the agent.
    return_intermediate_steps : bool
        A flag to determine whether intermediate steps should be
        included in the response. Default is True.
    action_sequence : str
        A formatted string representing actions performed by the agent.
        Example: "Step 1: Used `get-temperature` tool with `Mumbai`
        as input."
    agent : AgentExecutor
        The AgentExecutor, inheriting from LangChain's AgentExecutor.
    role : str
        The role of the agent in the interaction.
    delegation_agents : list
        A list of subsequent agents to which tasks can be delegated.
    """

    MEMORY_KEY = "chat_history"

    def __init__(
        self,
        llm: Any,
        goal: str,
        action_control: Any,
        format_intermediate_steps: Callable,
        tools: list = None,
        role: str = "",
        max_iterations: int = 15,
        return_intermediate_steps: bool = True,
        delegating_agents=None,
    ):
        """
        Initializes the stateful agent with its configuration,
        including tools, role, memory, and task delegation agents.

        Parameters
        ----------
        llm : Any
            The language model that the agent will use.
        goal : str
            The main purpose or objective of the agent.
        action_control : Any
            The output parser to control the agent's actions.
        format_intermediate_steps : Callable
            A function to format intermediate steps for logging.
        tools : list, optional
            A list of tools available to the agent. Default is an empty list.
        role : str, optional
            The role assigned to the agent. Default is an empty string.
        max_iterations : int, optional
            The maximum number of iterations the agent can take in a
            single task. Default is 15.
        return_intermediate_steps : bool, optional
            A flag to indicate whether intermediate steps should be
            returned in the response. Default is True.
        delegating_agents : list, optional
            A list of agents to which tasks can be delegated. Default
            is an empty list.
        """
        if tools is None:
            tools = []
        if delegating_agents is None:
            delegating_agents = []

        self.description = """
        An agent whose implicit task is realized by user's input, goal,
        and chat history.
        """

        self.llm = llm
        self.max_iterations = max_iterations
        self.return_intermediate_steps = return_intermediate_steps
        self.tools = tools
        self.goal = goal
        self.role = role
        self.action_control = action_control
        self.format_intermediate_steps = format_intermediate_steps
        self.agent = self.build_chain()
        self.action_sequence = ""
        self.delegation_agents = delegating_agents

        tool_names = ", ".join(
            [each_tool.__class__.__name__ for each_tool in self.tools]
        ).replace(", Tool", "")
        logging.warning(
            dedent(
                f"""

                Initializing StateFulAgent
                --------------------------
                name: {self.__class__.__name__}
                tools: {tool_names}
                model_name: {self.llm.model_name}
                provider: {self.llm.__class__.__name__}
                """
            )
        )

    def build_chain(self) -> AgentExecutor:
        """
        Builds a LangChain-based agent chain, with the defined tools,
        for executing tasks. The chain incorporates the tools, prompt,
        and LLM into a single pipeline.

        Returns
        -------
        AgentExecutor
            The agent's executor object, which is ready to process
            tasks.
        """
        if isinstance(self.llm, tuple):
            # Handle tuple LLM by selecting the first one
            self.llm = self.llm[0]

        # Bind tools to the LLM if tools are available
        temp_llm = self.llm.bind_tools(self.tools) if len(self.tools) > 0 else self.llm

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.goal),
                MessagesPlaceholder(variable_name=self.MEMORY_KEY),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        agent_chain = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: self.format_intermediate_steps(
                    x["intermediate_steps"]
                ),
                "chat_history": lambda x: x["chat_history"],
            }
            | prompt
            | temp_llm
            | self.action_control
        )

        return AgentExecutor(
            agent=agent_chain,
            tools=self.tools,
            verbose=True,
            max_iterations=self.max_iterations,
            return_intermediate_steps=self.return_intermediate_steps,
        )

    def get_tools(self):
        """
        Abstract method that should be implemented by subclasses
        to create a list of tools assigned to the agent.

        Returns
        -------
        list
            A list of tools assigned to the agent.
        """
        return []

    def invoke(self, input, chat_history, callbacks=None):
        """
        Invoke the agent synchronously with the given
        input and chat history.

        Parameters:
        -----------
        input : str
            The input string to be processed by the agent.
            This is added as the user message in executor chain.

        chat_history : list
            The chat history, provided as a list, to be
            considered by the agent. The schema
            has user, assistant messages.

        Returns:
        --------
        response : dict
            The response from the agent, which includes
            the agent's action sequence and any other relevant data.
        """
        response = self.agent.invoke(
            {"input": input, "chat_history": chat_history},
            config={"callbacks": callbacks},
        )
        self._save_agent_action_sequence(response)
        return response

    async def ainvoke(self, input, chat_history):
        """
        Invoke the agent asynchronously with the given
        input and chat history.

        Parameters:
        -----------
        input : str
            The input string to be processed by the agent.
            This is added as the user message in executor chain.

        chat_history : list
            The chat history, provided as a list, to be
            considered by the agent. The schema
            has user, assistant messages.

        Returns:
        --------
        response : dict
            The response from the agent, which includes
            the agent's action sequence and any other relevant data.
        """
        response = await self.agent.ainvoke(
            {"input": input, "chat_history": chat_history}
        )
        self._save_agent_action_sequence(response)
        return response

    async def astream(self, input: str, chat_history: list):
        """
        Stream agent's intermediate steps asynchronously, processing
        input and chat history step by step.

        Parameters
        ----------
        input : str
            The user input message to be processed by the agent.
        chat_history : list
            The chat history to be considered by the agent.

        Yields
        ------
        str
            The agent's action, intermediate step, or final result in
            a formatted string.
        """
        result = "Nothing from the astream"
        async for chunk in self.agent.astream(
            {"input": input, "chat_history": chat_history}
        ):
            # Process agent action
            message = ""
            if "actions" in chunk:
                for action in chunk["actions"]:
                    message = f"""<p><b style="color: #1f74bd;">{self.role}</b>: 
                        Will use the tool `{action.tool}` with inputs 
                        `{action.tool_input}`</p>"""
            elif "steps" in chunk:
                for step in chunk["steps"]:
                    message = f"""<p><b style="color: #1f74bd;">{self.role}</b>:
                        Tool executed, processing the output from tool</p>"""
            elif "output" in chunk:
                result = chunk
                self._save_agent_action_sequence(result)
                yield result
            else:
                raise ValueError("Unexpected chunk format")
            yield message

    def stream(self, input: str, chat_history: list):
        """
        Stream the agent's response using synchronous yield for each
        intermediate step, including actions, steps, and final output.

        Parameters
        ----------
        input : str
            The user input message to be processed by the agent.
        chat_history : list
            The chat history to be considered by the agent.

        Yields
        ------
        str
            The agent's action, intermediate step, or final result in
            a formatted string.
        """
        for chunk in self.agent.stream({"input": input, "chat_history": chat_history}):
            if "actions" in chunk:
                for action in chunk["actions"]:
                    message = f"""<p><b style="color: #1f74bd;">{self.role}</b>: 
                        Will use the tool `{action.tool}` with inputs 
                        `{action.tool_input}`</p>"""
                    yield message
            elif "steps" in chunk:
                for step in chunk["steps"]:
                    message = f"""<p><b style="color: #1f74bd;">{self.role}</b>: 
                        Tool executed, processing the output from tool</p>"""
                    yield message
            elif "output" in chunk:
                result = chunk
                yield result
            else:
                raise ValueError("Unexpected chunk format")

    def _save_agent_action_sequence(self, response: dict) -> str:
        """
        Save the agent's action sequence after processing the response.
        It processes the intermediate steps and formats them into a
        readable action sequence.

        Parameters
        ----------
        response : dict
            The response from the agent, containing the
            intermediate_steps.

        Returns
        -------
        str
            The formatted action sequence.
        """
        result = "Successful"
        try:
            if self.return_intermediate_steps:
                process_info = ""
                i = 0
                for action, _ in response["intermediate_steps"]:
                    i += 1
                    step_info = (
                        f"Step {i}: Used `{action.tool}` tool with "
                        f"`{action.tool_input}` as input.\n"
                    )
                    process_info += step_info
                self.action_sequence = process_info
        except Exception as e:
            logging.error(f"Error while _save_agent_action_sequence: {str(e)}")
            result = "Failed"
        return result
