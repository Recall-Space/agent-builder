"""
AgentBuilder to build agents.
"""

from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import (
    OpenAIToolsAgentOutputParser,
)
from agent_builder.agents.agent import Agent
from agent_builder.builders.base_builder import BaseBuilder


class AgentBuilder(BaseBuilder):
    """
    A builder class for constructing agents that interact with tools, memory, goals,
    and Language Learning Models (LLMs). This class follows a builder pattern,
    providing methods to configure various components of the agent and ensure
    the agent is fully customizable.

    The agent can be configured with:
    - LLMs (Language Learning Models)
    - Tools (for extended capabilities)
    - A goal or objective (to guide the agent's purpose)
    - Memory (for persistent knowledge or interaction history)
    - Action control mechanisms (to manage agent behavior)
    - Delegation agents (for task delegation)

    Attributes
    ----------
    agent : Agent
        The final agent object that is built by this builder.
    llm : Any
        The language model (e.g., GPT) that the agent will use for responses.
    tools : list
        A list of tools that the agent can use to perform specific tasks.
    goal : str
        The primary goal or objective of the agent.
    role : str
        The role assigned to the agent (e.g., assistant, expert).
    action_control : OpenAIToolsAgentOutputParser
        The action control mechanism to manage how the agent interprets
        and executes actions.
    format_intermediate_steps : Callable
        A function to format intermediate steps during the agent's execution process.
    memory_object : object
        The memory configuration used to enable memory-based capabilities for the agent.
    delegating_agents : list
        A list of other agents to which tasks can be delegated by the primary agent.

    Methods
    -------
    reset()
        Resets the builder to its initial state, clearing all configurations.
    set_llm(llm)
        Sets the language model for the agent.
    add_tool(tool)
        Adds a tool to the agent's toolset.
    set_role(role)
        Sets the role for the agent.
    set_action_control(action_control)
        Sets the action control mechanism for the agent.
    set_format_intermediate_steps(format_intermediate_steps)
        Sets the format function for intermediate steps in agent execution.
    add_delegation_agent(delegation_agent)
        Adds a delegation agent to the agent's list of delegation agents.
    set_goal(goal)
        Sets the primary goal for the agent.
    configure_memory(memory)
        Configures memory for the agent.
    get_instructions_json()
        Generates the agent's instructions as a JSON object (not implemented).
    render_instructions_json()
        Renders agent instructions from JSON to rebuild the agent (not implemented).
    validate()
        Validates all necessary components before building the agent.
    build()
        Builds and returns the fully configured agent.
    """

    def __init__(self):
        """Initialize the builder and reset all attributes to default values."""
        super().__init__()
        self.reset()

    def reset(self):
        """
        Resets the builder to its initial state, clearing all previously set configurations.
        This method is called after the agent is built to ensure the builder can be reused.
        """
        self.agent = None
        self.llm = None
        self.tools = []
        self.goal = ""
        self.role = ""
        self.action_control = OpenAIToolsAgentOutputParser()
        self.format_intermediate_steps = format_to_openai_tool_messages
        self.memory_object = None
        self.delegating_agents = []

    def set_llm(self, llm):
        """
        Sets the language model (LLM) for the agent.

        Parameters
        ----------
        llm : Any
            The language model that the agent will use for generating responses.
        """
        self.llm = llm

    def add_tool(self, tool):
        """
        Adds a tool to the agent's toolset, allowing the agent to perform specific tasks.

        Parameters
        ----------
        tool : Any
            The tool to be added to the agent's capabilities.
        """
        self.tools.append(tool)

    def set_role(self, role: str):
        """
        Sets the role that the agent will assume during interactions.

        Parameters
        ----------
        role : str
            The role or persona that the agent will embody (e.g., assistant, advisor).
        """
        self.role = role

    def set_action_control(self, action_control: object):
        """
        Sets the action control mechanism that manages how the agent processes and
        executes actions.

        Parameters
        ----------
        action_control : object
            The object responsible for handling the agent's action control (e.g., an output parser).
        """
        self.action_control = action_control

    def set_format_intermediate_steps(self, format_intermediate_steps: object):
        """
        Sets the function that formats intermediate steps during the agent's execution.

        Parameters
        ----------
        format_intermediate_steps : object
            A callable that formats intermediate steps in the agent's task execution.
        """
        self.format_intermediate_steps = format_intermediate_steps

    def add_delegation_agent(self, delegation_agent: object):
        """
        Adds another agent to which tasks can be delegated by the primary agent.

        Parameters
        ----------
        delegation_agent : object
            An agent to which tasks can be delegated by the primary agent.
        """
        self.delegating_agents.append(delegation_agent)

    def set_goal(self, goal: str):
        """
        Sets the primary goal or objective that the agent should aim to achieve.

        Parameters
        ----------
        goal : str
            The primary purpose or objective assigned to the agent.
        """
        self.goal = goal

    def configure_memory(self, memory: object):
        """
        Configures the memory object to enable memory-based functionality for the agent.

        Parameters
        ----------
        memory : object
            The memory configuration to be used by the agent.
        """
        self.memory_object = memory

    def get_instructions_json(self):
        """
        Generates the agent's building instructions as a JSON object.

        Returns
        -------
        dict
            The agent's configuration and instructions in JSON format.

        Raises
        ------
        NotImplementedError
            This method is not yet implemented.
        """
        raise NotImplementedError("This function is not yet implemented.")

    def render_instructions_json(self):
        """
        Renders the agent's instructions back into the builder, possibly to re-instantiate
        or rebuild an agent from a JSON configuration.

        Returns
        -------
        dict
            The agent's configuration rendered as JSON.

        Raises
        ------
        NotImplementedError
            This method is not yet implemented.
        """
        raise NotImplementedError("This function is not yet implemented.")

    def validate(self):
        """
        Validates that all necessary components for building the agent are properly set,
        such as LLM, goal, action control, and formatting functions.

        Raises
        ------
        ValueError
            If any required component is missing or not configured.
        """
        if not self.llm:
            raise ValueError("LLM must be set before building the agent.")
        if not self.goal:
            raise ValueError("Goal must be set before building the agent.")
        if not self.action_control:
            raise ValueError("Action control must be set before building the agent.")
        if not self.format_intermediate_steps:
            raise ValueError(
                "Format for intermediate steps must be set before building the agent."
            )

    def build(self) -> Agent:
        """
        Constructs and returns the fully built agent based on the configured parameters.

        The agent is built with the specified LLM, tools, goal, role, action control,
        and delegation agents.

        Returns
        -------
        Agent
            The fully constructed agent with the specified configuration.
        """
        self.validate()  # Ensure all required components are set

        # Build the agent with all configured parameters
        agent = Agent(
            llm=self.llm,
            goal=self.goal,
            action_control=self.action_control,
            format_intermediate_steps=self.format_intermediate_steps,
            tools=self.tools,
            role=self.role,
            delegating_agents=self.delegating_agents,
        )

        # Reset the builder to allow for reuse
        self.reset()

        return agent
