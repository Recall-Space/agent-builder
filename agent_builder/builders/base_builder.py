from abc import ABC, abstractmethod


class BaseBuilder(ABC):
    """
    Abstract base class for building components like agents or tools.
    This class defines the basic structure for configuring and building
    various components.
    """

    def __init__(self):
        self.reset()

    @abstractmethod
    def reset(self):
        """
        Resets the builder to its initial state.
        Must be implemented by concrete builders.
        """
        pass

    @abstractmethod
    def build(self):
        """
        Abstract method to finalize and build the component (agent, tool, etc.).
        Must be implemented by concrete builders.
        """
        pass
