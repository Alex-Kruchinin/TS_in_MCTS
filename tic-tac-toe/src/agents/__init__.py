"""Agents that can choose moves in supported games."""

from src.agents.base_agent import Agent
from src.agents.random_agent import RandomAgent
from src.agents.tactical_agent import TacticalAgent
from src.agents.weak_tactical_agent import WeakTacticalAgent

__all__ = ["Agent", "RandomAgent", "TacticalAgent", "WeakTacticalAgent"]
