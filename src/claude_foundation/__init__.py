"""Foundation layer for Claude-powered applications."""

from .chat import ChatSession, RefusalError, TurnResult
from .config import Settings

__all__ = ["ChatSession", "RefusalError", "Settings", "TurnResult"]
__version__ = "0.1.0"
