"""
This module provides a utility decorator for handling exceptions in
synchronous functions within the context of a tool-based architecture.

When an synchronous function that is decorated with this utility fails,
it raises a `ToolException`. This captures and reports details about the
failure to the agent, allowing the agent's execution to continue without
abrupt interruptions.

The decorator can be customized with a hint message that provides additional
context about the potential issue, helping to diagnose problems effectively.
"""

from textwrap import dedent
import functools
from langchain.tools.base import ToolException


def raise_tool_exception_on_fail(hint=""):
    """Implements a wrapper function that raises ToolException.
    This approach ensures that any function failure is captured and
    reported as an observation to the agent, instead of causing the
    entire execution process to terminate.

    Parameters
    ----------
    hint : str
        Custom message, used to give a hint to the agent of what could be wrong.

    Returns
    ------
    decorator : callable
        A decorator function that can be applied to other functions.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as error:
                nonlocal hint
                hint = f"hint: {hint}" if hint != "" else hint
                raise ToolException(
                    dedent(
                        f"""
                    {hint}
                    Error: {str(error)}
                """
                    )
                )

        return wrapper

    return decorator
