"""
VoiceAgentBuilder to build VoiceAgent instances.
"""

import logging
from typing import Any, Callable, List
from langchain_core.tools import BaseTool
from pydantic import BaseModel
from agent_builder.agents.voice_agent import VoiceAgent


class VoiceAgentBuilder:
    """
    A builder class for constructing voice agents that interact with tools, goals,
    and handle voice input/output streams using OpenAI's real-time API. This class
    follows a builder pattern, providing methods to configure various components
    of the VoiceAgent and ensure the agent is fully customizable.

    The VoiceAgent can be configured with:
    - API Key
    - Model URL
    - Tools (for extended capabilities)
    - A goal or objective (to guide the agent's purpose)
    - Role (to specify the agent's role in interaction)

    Attributes
    ----------
    agent : VoiceAgent
        The final VoiceAgent object that is built by this builder.
    api_key : str
        The API key used to authenticate with the OpenAI real-time API.
    model_url : str
        The URL of the model endpoint to connect to.
    tools : List[BaseTool]
        A list of tools that the agent can use to perform specific tasks.
    goal : str
        The primary goal or objective of the agent.
    role : str
        The role assigned to the agent (e.g., assistant, expert).

    Methods
    -------
    reset()
        Resets the builder to its initial state, clearing all configurations.
    set_api_key(api_key)
        Sets the API key for the VoiceAgent.
    set_model_url(model_url)
        Sets the model URL for the VoiceAgent.
    add_tool(tool)
        Adds a tool to the agent's toolset.
    set_tools(tools)
        Sets the entire list of tools for the agent.
    set_goal(goal)
        Sets the primary goal for the agent.
    set_role(role)
        Sets the role for the agent.
    validate()
        Validates all necessary components before building the VoiceAgent.
    build()
        Builds and returns the fully configured VoiceAgent.
    """

    def __init__(self):
        """Initialize the builder and reset all attributes to default values."""
        self.reset()

    def reset(self):
        """
        Resets the builder to its initial state, clearing all previously set configurations.
        This method is called after the agent is built to ensure the builder can be reused.
        """
        self.agent = None
        self.api_key = None
        self.model_url = (
            "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
        )
        self.tools: List[BaseTool] = []
        self.goal = ""
        self.role = ""

    def set_api_key(self, api_key: str):
        """
        Sets the API key used to authenticate with the OpenAI real-time API.

        Parameters
        ----------
        api_key : str
            The API key for authentication.
        """
        self.api_key = api_key

    def set_model_url(self, model_url: str):
        """
        Sets the model URL for the VoiceAgent.

        Parameters
        ----------
        model_url : str
            The model URL to connect to.
        """
        self.model_url = model_url

    def add_tool(self, tool: BaseTool):
        """
        Adds a tool to the agent's toolset.

        Parameters
        ----------
        tool : BaseTool
            The tool to be added to the agent's capabilities.
        """
        self.tools.append(tool)

    def set_goal(self, goal: str):
        """
        Sets the primary goal or objective that the agent should aim to achieve.

        Parameters
        ----------
        goal : str
            The primary purpose or objective assigned to the agent.
        """
        self.goal = goal

    def set_role(self, role: str):
        """
        Sets the role that the agent will assume during interactions.

        Parameters
        ----------
        role : str
            The role or persona that the agent will embody (e.g., assistant, advisor).
        """
        self.role = role

    def validate(self):
        """
        Validates that all necessary components for building the VoiceAgent are properly set,
        such as API key and goal.

        Raises
        ------
        ValueError
            If any required component is missing or not configured.
        """
        if not self.api_key:
            raise ValueError("API key must be set before building the VoiceAgent.")
        if not self.goal:
            raise ValueError("Goal must be set before building the VoiceAgent.")
        if not isinstance(self.tools, list):
            raise ValueError("Tools must be a list of BaseTool instances.")

    def build(self) -> "VoiceAgent":
        """
        Constructs and returns the fully built VoiceAgent based on the configured parameters.

        The VoiceAgent is built with the specified API key, model URL, tools, goal, and role.

        Returns
        -------
        VoiceAgent
            The fully constructed VoiceAgent with the specified configuration.
        """
        self.validate()  # Ensure all required components are set

        # Build the VoiceAgent with all configured parameters
        agent = VoiceAgent(
            api_key=self.api_key,
            model_url=self.model_url,
            goal=self.goal,
            tools=self.tools,
            role=self.role,
        )

        # Reset the builder to allow for reuse
        self.reset()

        return agent
   