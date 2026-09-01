from __future__ import annotations

from game import Action, Move, OthelloState, PassMove


def print_turn(
    turn_number: int,
    state: OthelloState,
    action: Action,
    agent_name: str,
) -> None:
    """Print the board immediately before an agent's selected action"""

    if isinstance(action, Move):
        action_text = f"({action.row}, {action.col})"
    elif isinstance(action, PassMove):
        action_text = "PASS"
    else:  # Defensive fallback for future action types.
        action_text = repr(action)

    print(f"\nTurn {turn_number}: {state.current_player.name}")
    print(f"Agent: {agent_name}")
    print(state)
    print(f"Action: {action_text}")
