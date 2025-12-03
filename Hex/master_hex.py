from typing import Dict, Iterable, List

from seahorse.game.game_state import GameState
from seahorse.game.master import GameMaster
from seahorse.player.player import Player


class MasterHex(GameMaster):
    """
    Master to play the game Hex

    Attributes:
        name (str): Name of the game
        initial_game_state (GameState): Initial state of the game
        current_game_state (GameState): Current state of the game
        players_iterator (Iterable): An iterable for the players_iterator, ordered according to the playing order.
            If a list is provided, a cyclic iterator is automatically built
        log_level (str): Name of the log file
    """

    def __init__(self, name: str, initial_game_state: GameState, players_iterator: Iterable[Player], log_level: str, port: int = 8080, hostname: str = "localhost", time_limit: int = 60*15) -> None:
        super().__init__(name, initial_game_state, players_iterator, log_level, port, hostname, time_limit)
        
    def compute_winner(self) -> List[Player]:
        """
        Computes the winners of the game based on the scores.

        Seahorse's GameMaster.get_winner() calls this method with no
        arguments, so scores must be read from the current game state.

        Returns:
            List[Player]: List of the players who won the game
        """
        scores: Dict[int, float] = self.current_game_state.get_scores()
        if not scores:
            return []
        max_val = max(scores.values())
        players_id = [pid for pid, val in scores.items() if val == max_val]
        winners = [p for p in self.players if p.get_id() in players_id]
        return winners

    def get_custom_stats(self):
        return [{"name": "coups", "value": self.current_game_state.get_step(), "agent_id":-1}]