"""
ToolBuilder class which provides flexibility to create tools
for Agent.
"""

from langchain.tools import StructuredTool
from agent_builder.builders.base_builder import BaseBuilder
from agent_builder.utils.araise_tool_exception_on_fail import (
    araise_tool_exception_on_fail,
)
from agent_builder.utils.raise_tool_exception_on_fail import (
    raise_tool_exception_on_fail,
)


class ToolBuilder(BaseBuilder):
    """
    A builder class for constructing tools that agents can use.
    Tools require essential components like a function, name, schema,
    and description to be set before being built. This class provides
    methods to configure and build these tools.

    Attributes
    ----------
    name : str
        The name of the tool.
    function : Callable
        The main function that the tool will execute.
    coroutine : Callable, optional
        The coroutine that the tool will execute, if applicable.
    return_direct : bool
        Whether the tool should return the result directly.
        Default is False.
    description : str
        A description of what the tool does.
    schema : object
        The schema that defines the structure of the tool's input.
    """

    def __init__(self):
        """Initializes the builder and resets all attributes."""
        super().__init__()

    def reset(self):
        """
        Resets the builder to its initial state, clearing all attributes.
        This is called after building a tool to prepare the builder
        for the next tool creation.
        """
        self.name = ""
        self.function = None
        self.coroutine = None
        self.return_direct = False
        self.hint = ""
        self.description = ""
        self.schema = None
        self.max_iterations = 1

    def set_name(self, name: str):
        """
        Sets the name of the tool.

        Parameters
        ----------
        name : str
            The name that will identify the tool.
        """
        self.name = name

    def set_function(self, function):
        """
        Sets the function for the tool.

        Parameters
        ----------
        function : Callable
            The function that the tool will execute when invoked.
        """
        self.function = function

    def set_coroutine(self, coroutine):
        """
        Sets the coroutine for the tool.

        Parameters
        ----------
        coroutine : Callable
            The asynchronous coroutine that the tool will execute.
        """
        self.coroutine = coroutine

    def set_hint(self, hint):
        """
        Sets the hint for the tool in case of exception.
        The hint guides the agent to reiterate in case of exception
        of tool.

        Parameters
        ----------
        coroutine : Callable
            The asynchronous coroutine that the tool will execute.
        """
        self.hint = hint

    def set_description(self, description: str):
        """
        Sets the description for the tool.

        Parameters
        ----------
        description : str
            A short description of what the tool does.
        """
        self.description = description

    def set_schema(self, schema: object):
        """
        Sets the schema that defines the input structure for the tool.

        Parameters
        ----------
        schema : object
            The schema that defines the tool's input parameters.
        """
        self.schema = schema

    def set_max_iterations(self, max_iterations: int):
        """
        Sets the maximum iterations for the tool.

        Parameters
        ----------
        max_iterations : int
            Number of iterations the tool could be called maximum.
        """
        self.max_iterations = max_iterations

    def validate(self):
        """
        Validates that all necessary components (name, function, schema,
        and description) are set before building the tool.

        Raises
        ------
        ValueError
            If any required attribute is not set before building.
        """
        if not self.function:
            raise ValueError("Tool must have a function.")
        if not self.schema:
            raise ValueError("Tool must have a schema.")
        if not self.name:
            raise ValueError("Tool must have a name.")
        if not self.description:
            raise ValueError("Tool must have a description.")

    def build(self) -> StructuredTool:
        """
        Constructs and returns the tool using the provided configuration.

        Returns
        -------
        tool : StructuredTool
            The tool that was constructed based on the configured parameters.
        """
        self.validate()
        self._add_exception_handling()
        if self.coroutine is None:
            print()
            tool = StructuredTool.from_function(
                func=self.function,
                name=self.name,
                description=self.description,
                args_schema=self.schema,
                return_direct=self.return_direct,
                handle_tool_error=True,
                max_iterations=self.max_iterations,
            )
        else:
            tool = StructuredTool.from_function(
                func=self.function,
                name=self.name,
                description=self.description,
                args_schema=self.schema,
                return_direct=self.return_direct,
                coroutine=self.coroutine,
                handle_tool_error=True,
                max_iterations=self.max_iterations,
            )

        # Reset the builder for the next tool creation
        self.reset()
        return tool

    def _add_exception_handling(self):
        """
        Adds decorator with given hint for exception handling of tool.

        """
        if self.hint == "":
            self.hint = f"{self.name} failed"
        self.function = raise_tool_exception_on_fail(hint=self.hint)(self.function)
        if self.coroutine is not None:
            self.coroutine = araise_tool_exception_on_fail(hint=self.hint)(
                self.coroutine
            )
