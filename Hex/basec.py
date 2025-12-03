# my_player.py
from __future__ import annotations

import math
import random
import time
import heapq
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Any

from player_hex import PlayerHex
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from seahorse.game.light_action import LightAction
from game_state_hex import GameStateHex
from seahorse.utils.custom_exceptions import MethodNotImplementedError


# ============================================================
#       Minimax P2 = T2 + T3 + T9-light (bridges simples)
#   - Alpha–beta + iterative deepening + TT
#   - Quiescence sur coups tactiques
#   - Killer heuristic + History heuristic
#   - LMR léger
#   - Heuristique : distance + connectivité + patterns simples
#   - + micro-solver "mate en 1" + détection de victoire fiable
# ============================================================


class SearchTimeout(Exception):
    """Exception interne pour interrompre proprement la recherche."""
    pass


@dataclass
class TTEntry:
    value: float
    depth: int
    best_action: Optional[LightAction]


def _winner_from_scores(state: GameStateHex) -> Optional[str]:
    """
    Helper global : renvoie le piece_type ("R" ou "B") du gagnant
    en se basant UNIQUEMENT sur l'API officielle de GameStateHex :

        - state.is_done()
        - state.scores : dict[player_id -> score]
        - state.players : liste de PlayerHex (avec .piece_type)
    """
    # is_done() est déjà défini dans GameStateHex via self.scores
    if not state.is_done():
        return None

    scores = state.scores
    for p in state.players:
        if scores.get(p.id, 0.0) == 1.0:
            # PlayerHex a un attribut piece_type
            return getattr(p, "piece_type", None)
    return None


class MinimaxEngine:
    """
    Moteur Minimax pour Hex, perspective fixée au joueur racine (root_piece).

    Pour débugger en étapes P2a/P2b, tu peux activer/désactiver :

      self.USE_QUIESCENCE
      self.USE_KILLER_HISTORY
      self.USE_LMR
      self.USE_PATTERNS
    """

    def __init__(self, root_piece: str):
        self.root_piece: str = root_piece  # "R" ou "B"
        self.tt: Dict[Any, TTEntry] = {}

        # TT et heuristiques de recherche
        self.WIN_SCORE = 10_000.0
        self.LOSS_SCORE = -10_000.0
        self.MAX_DEPTH_HARD = 7  # barrière dure (LMR / quiescence donneront parfois plus en "effectif")

        # Flags Pack P2
        self.USE_QUIESCENCE = True
        self.USE_KILLER_HISTORY = True
        self.USE_LMR = True
        self.USE_PATTERNS = True

        # Heuristique d'évaluation (T2 + T9-light)
        self.W_DIST = 3.0
        self.W_CONN = 1.0
        self.W_BRIDGE = 1.5      # patterns "bridges" simples / deux voisins amis
        self.W_THREAT = 1.0      # empties avec fort voisinage ennemi

        # Killer moves : dict[ply] -> [killer1, killer2]
        self.killer_moves: Dict[int, List[Optional[LightAction]]] = {}

        # History heuristic : dict[(player_piece, (i,j))] -> score
        self.history: Dict[Tuple[str, Tuple[int, int]], float] = {}

    # -----------------------
    # Interface principale
    # -----------------------

    def choose_action(self, state: GameStateHex, time_budget: float) -> LightAction:
        """
        Choisit le meilleur coup avec iterative deepening sous contrainte de temps.
        """
        start_time = time.time()
        deadline = start_time + max(0.05, time_budget * 0.9)  # marge de sécurité

        possible_actions = list(state.get_possible_light_actions())
        if not possible_actions:
            raise MethodNotImplementedError("No possible actions in non-terminal state.")

        # Fallback : coup aléatoire si la search échoue
        best_action: LightAction = random.choice(possible_actions)
        best_value: float = float("-inf")

        depth = 1
        while depth <= self.MAX_DEPTH_HARD:
            try:
                value, action = self._search_root(
                    state, depth, deadline
                )
                if action is not None:
                    best_action = action
                    best_value = value
                depth += 1
            except SearchTimeout:
                break

        # print(f"[DEBUG] depth_reached={depth-1}, value={best_value}")
        return best_action

    # -----------------------
    # Minimax + alpha–beta
    # -----------------------

    def _search_root(
        self,
        state: GameStateHex,
        depth: int,
        deadline: float,
    ) -> Tuple[float, Optional[LightAction]]:
        """
        Racine du Minimax : on détermine aussi le meilleur coup.
        """
        is_max_player = (state.next_player.get_piece_type() == self.root_piece)
        alpha = float("-inf")
        beta = float("+inf")

        possible_actions = list(state.get_possible_light_actions())
        if not possible_actions:
            v = self._evaluate_state(state)
            return v, None

        # Move ordering (centralité + history + killer)
        possible_actions = self._order_moves(state, possible_actions, ply=0, tt_entry=None)

        best_value = float("-inf") if is_max_player else float("+inf")
        best_action: Optional[LightAction] = None

        for idx, action in enumerate(possible_actions):
            self._check_time(deadline)
            child_state = state.apply_action(action)

            value = self._alphabeta(
                child_state,
                depth - 1,
                alpha,
                beta,
                not is_max_player,
                deadline,
                ply=1,
            )

            if is_max_player:
                if value > best_value or best_action is None:
                    best_value = value
                    best_action = action
                alpha = max(alpha, best_value)
            else:
                if value < best_value or best_action is None:
                    best_value = value
                    best_action = action
                beta = min(beta, best_value)

            if alpha >= beta:
                break

        return best_value, best_action

    def _alphabeta(
        self,
        state: GameStateHex,
        depth: int,
        alpha: float,
        beta: float,
        is_max_player: bool,
        deadline: float,
        ply: int,
    ) -> float:
        """
        Minimax récursif avec alpha–beta + TT + killer/history + LMR + quiescence.
        """
        self._check_time(deadline)

        # Terminal ? -> utiliser API officielle via _winner_from_scores
        winner = _winner_from_scores(state)
        if winner is not None:
            if winner == self.root_piece:
                return self.WIN_SCORE
            else:
                return self.LOSS_SCORE

        # Profondeur limite -> quiescence ou eval statique
        if depth <= 0:
            if self.USE_QUIESCENCE:
                return self._quiescence(state, alpha, beta, is_max_player, deadline, ply, q_depth=4)
            else:
                return self._evaluate_state(state)

        # Transposition table
        key = self._state_key(state)
        entry = self.tt.get(key)
        if entry is not None and entry.depth >= depth:
            return entry.value

        possible_actions = list(state.get_possible_light_actions())
        if not possible_actions:
            value = self._evaluate_state(state)
            self.tt[key] = TTEntry(value=value, depth=depth, best_action=None)
            return value

        # Move ordering avec TT, killer, history...
        possible_actions = self._order_moves(state, possible_actions, ply=ply, tt_entry=entry)

        player_to_move = state.next_player.get_piece_type()

        if is_max_player:
            value = float("-inf")
        else:
            value = float("+inf")

        best_action: Optional[LightAction] = None

        # LMR : on réduit la profondeur pour certains coups tardifs non tactiques
        for move_index, action in enumerate(possible_actions):
            self._check_time(deadline)
            child_state = state.apply_action(action)

            child_is_tactical = self._is_tactical_move(state, action, player_to_move)

            # Profondeur pour ce coup (LMR léger)
            next_depth = depth - 1
            do_lmr = (
                self.USE_LMR
                and depth >= 3
                and move_index >= 3
                and not child_is_tactical
            )

            if do_lmr:
                reduced_depth = max(1, depth - 2)
                child_value = self._alphabeta(
                    child_state,
                    reduced_depth,
                    alpha,
                    beta,
                    not is_max_player,
                    deadline,
                    ply=ply + 1,
                )
                # Vérification : si ça a l'air très prometteur, on refait une recherche non réduite
                if (is_max_player and child_value > alpha) or ((not is_max_player) and child_value < beta):
                    child_value = self._alphabeta(
                        child_state,
                        next_depth,
                        alpha,
                        beta,
                        not is_max_player,
                        deadline,
                        ply=ply + 1,
                    )
            else:
                child_value = self._alphabeta(
                    child_state,
                    next_depth,
                    alpha,
                    beta,
                    not is_max_player,
                    deadline,
                    ply=ply + 1,
                )

            # Mise à jour max/min
            if is_max_player:
                if child_value > value or best_action is None:
                    value = child_value
                    best_action = action
                if value > alpha:
                    alpha = value
            else:
                if child_value < value or best_action is None:
                    value = child_value
                    best_action = action
                if value < beta:
                    beta = value

            # Beta-cutoff ? -> killer & history
            if alpha >= beta:
                if self.USE_KILLER_HISTORY:
                    self._register_killer(ply, action)
                    self._update_history(player_to_move, action, depth)
                break

        # Mise à jour TT
        self.tt[key] = TTEntry(value=value, depth=depth, best_action=best_action)
        return value

    # -----------------------
    # Quiescence search (localisée)
    # -----------------------

    def _quiescence(
        self,
        state: GameStateHex,
        alpha: float,
        beta: float,
        is_max_player: bool,
        deadline: float,
        ply: int,
        q_depth: int,
    ) -> float:
        """
        Quiescence search : explore seulement les coups tactiques (connectants/brisants)
        sur une profondeur limitée q_depth.
        """
        self._check_time(deadline)

        # Utiliser la même logique de terminalité que le reste
        winner = _winner_from_scores(state)
        if winner is not None:
            if winner == self.root_piece:
                return self.WIN_SCORE
            else:
                return self.LOSS_SCORE

        stand_pat = self._evaluate_state(state)

        if is_max_player:
            if stand_pat >= beta:
                return stand_pat
            if stand_pat > alpha:
                alpha = stand_pat
        else:
            if stand_pat <= alpha:
                return stand_pat
            if stand_pat < beta:
                beta = stand_pat

        if q_depth <= 0:
            return stand_pat

        player_to_move = state.next_player.get_piece_type()
        possible_actions = list(state.get_possible_light_actions())

        # On ne garde que les coups tactiques
        tactical_actions = [
            a for a in possible_actions
            if self._is_tactical_move(state, a, player_to_move)
        ]
        if not tactical_actions:
            return stand_pat

        # Move ordering sur les tactiques aussi
        tactical_actions = self._order_moves(state, tactical_actions, ply=ply, tt_entry=None)

        if is_max_player:
            value = stand_pat
            for action in tactical_actions:
                self._check_time(deadline)
                child_state = state.apply_action(action)
                child_value = self._quiescence(
                    child_state,
                    alpha,
                    beta,
                    not is_max_player,
                    deadline,
                    ply + 1,
                    q_depth - 1,
                )
                if child_value > value:
                    value = child_value
                if value > alpha:
                    alpha = value
                if alpha >= beta:
                    break
        else:
            value = stand_pat
            for action in tactical_actions:
                self._check_time(deadline)
                child_state = state.apply_action(action)
                child_value = self._quiescence(
                    child_state,
                    alpha,
                    beta,
                    not is_max_player,
                    deadline,
                    ply + 1,
                    q_depth - 1,
                )
                if child_value < value:
                    value = child_value
                if value < beta:
                    beta = value
                if alpha >= beta:
                    break

        return value

    # -----------------------
    # Heuristique (T2 + T9-light)
    #  - distance de connexion
    #  - connectivité
    #  - "bridges" / deux-step connexions simples
    #  - menaces : fortes cases pour l'adversaire
    # -----------------------

    def _evaluate_state(self, state: GameStateHex) -> float:
        """
        h(s) = W_DIST * (d_opp - d_self)
             + W_CONN * (conn_self - conn_opp)
             + W_BRIDGE * (bridge_self - bridge_opp)
             + W_THREAT * (threat_self - threat_opp)
        """
        root = self.root_piece
        opp = "B" if root == "R" else "R"

        d_self = self._shortest_connection_distance(state, root)
        d_opp = self._shortest_connection_distance(state, opp)

        conn_self = self._connectivity_score(state, root)
        conn_opp = self._connectivity_score(state, opp)

        if self.USE_PATTERNS:
            bridge_self, threat_self = self._pattern_scores(state, root)
            bridge_opp, threat_opp = self._pattern_scores(state, opp)
        else:
            bridge_self = bridge_opp = 0.0
            threat_self = threat_opp = 0.0

        dist_term = (d_opp - d_self)
        conn_term = (conn_self - conn_opp)
        bridge_term = (bridge_self - bridge_opp)
        threat_term = (threat_self - threat_opp)

        return (
            self.W_DIST * dist_term
            + self.W_CONN * conn_term
            + self.W_BRIDGE * bridge_term
            + self.W_THREAT * threat_term
        )

    def _shortest_connection_distance(self, state: GameStateHex, piece_type: str) -> float:
        """
        Distance façon greedy :

        - cases amies : coût 0
        - cases vides : coût 1
        - cases ennemies : bloquantes
        """
        rep = state.rep
        env = rep.env
        dim_i, dim_j = rep.dimensions

        INF = 1e9
        dist = [[INF for _ in range(dim_j)] for _ in range(dim_i)]
        pq: List[Tuple[float, Tuple[int, int]]] = []

        objectives: List[Tuple[int, int]] = []

        if piece_type == "R":
            for j in range(dim_j):
                objectives.append((dim_i - 1, j))
                cell = env.get((0, j))
                if cell is None:
                    dist[0][j] = 1.0
                elif cell.piece_type == piece_type:
                    dist[0][j] = 0.0
                else:
                    continue
                heapq.heappush(pq, (dist[0][j], (0, j)))
        else:
            for i in range(dim_i):
                objectives.append((i, dim_j - 1))
                cell = env.get((i, 0))
                if cell is None:
                    dist[i][0] = 1.0
                elif cell.piece_type == piece_type:
                    dist[i][0] = 0.0
                else:
                    continue
                heapq.heappush(pq, (dist[i][0], (i, 0)))

        best = INF
        while pq:
            d, (i, j) = heapq.heappop(pq)
            if d > dist[i][j]:
                continue
            if (i, j) in objectives:
                best = d
                break

            for n_type, (ni, nj) in rep.get_neighbours(i, j).values():
                if n_type == "EMPTY":
                    new_d = d + 1.0
                elif n_type == piece_type:
                    new_d = d
                else:
                    continue

                if 0 <= ni < dim_i and 0 <= nj < dim_j and new_d < dist[ni][nj]:
                    dist[ni][nj] = new_d
                    heapq.heappush(pq, (new_d, (ni, nj)))

        return best

    def _connectivity_score(self, state: GameStateHex, piece_type: str) -> float:
        """
        Score simple de connectivité : nombre de paires de voisins amis.
        """
        rep = state.rep
        env = rep.env
        score = 0.0

        for (i, j), piece in env.items():
            if piece.piece_type != piece_type:
                continue
            for n_type, (ni, nj) in rep.get_neighbours(i, j).values():
                if n_type == piece_type:
                    score += 1.0

        return score

    def _pattern_scores(self, state: GameStateHex, piece_type: str) -> Tuple[float, float]:
        """
        T9-light :
          - bridge_score : nombre de cases vides ayant >= 2 voisins amis.
          - threat_score : nombre de cases vides ayant >= 2 voisins ennemis.

        C'est une approx "safe" de :
          - has_safe_bridge(player)
          - has_two_step_connection(player)
        """
        rep = state.rep
        env = rep.env
        dim_i, dim_j = rep.dimensions

        bridge_score = 0.0
        threat_score = 0.0

        opp = "B" if piece_type == "R" else "R"

        for i in range(dim_i):
            for j in range(dim_j):
                if (i, j) in env:
                    continue  # case occupée
                # case vide
                neigh = rep.get_neighbours(i, j).values()
                friend = sum(1 for n_type, _ in neigh if n_type == piece_type)
                enemy = sum(1 for n_type, _ in neigh if n_type == opp)

                if friend >= 2:
                    bridge_score += 1.0
                if enemy >= 2:
                    threat_score += 1.0

        return bridge_score, threat_score

    # -----------------------
    # Utilitaires : TT, move ordering, time, killer/history, tactiques
    # -----------------------

    def _state_key(self, state: GameStateHex) -> Tuple:
        """
        Clé simple pour TT : (positions triées, joueur au trait).
        """
        rep = state.rep
        env = rep.env
        items = tuple(sorted((pos, piece.piece_type) for pos, piece in env.items()))
        player_to_move = state.next_player.get_piece_type()
        return items, player_to_move

    def _order_moves(
        self,
        state: GameStateHex,
        actions: List[LightAction],
        ply: int,
        tt_entry: Optional[TTEntry] = None,
    ) -> List[LightAction]:
        """
        Move ordering combinant :
          - coup TT
          - killer moves (si activé)
          - history heuristic (si activé)
          - centralité géométrique
          - "tactique" (coups avec >=2 voisins amis / ennemis)
        """
        rep = state.rep
        dim_i, dim_j = rep.dimensions
        center_i = (dim_i - 1) / 2.0
        center_j = (dim_j - 1) / 2.0

        player_to_move = state.next_player.get_piece_type()
        killers = self.killer_moves.get(ply, [None, None]) if self.USE_KILLER_HISTORY else [None, None]

        tt_move = tt_entry.best_action if tt_entry is not None else None

        scored: List[Tuple[float, LightAction]] = []

        for a in actions:
            pos = a.data["position"]
            i, j = pos
            center_dist = abs(i - center_i) + abs(j - center_j)
            score = -center_dist  # plus proche du centre = mieux

            # Bonus TT
            if tt_move is not None and pos == tt_move.data["position"]:
                score += 1000.0

            # Bonus killer
            if self.USE_KILLER_HISTORY:
                for k in killers:
                    if k is not None and pos == k.data["position"]:
                        score += 500.0
                        break

                # Bonus history
                score += self.history.get((player_to_move, pos), 0.0)

            # Bonus tactique léger
            if self._is_tactical_move(state, a, player_to_move):
                score += 50.0

            scored.append((score, a))

        scored.sort(key=lambda x: x[0], reverse=True)
        ordered = [a for _, a in scored]
        return ordered

    def _is_tactical_move(self, state: GameStateHex, action: LightAction, player_piece: str) -> bool:
        """
        Coup tactique = coup qui :
          - connecte potentiellement deux groupes amis (>=2 voisins amis),
          - ou touche fortement la chaîne adverse (>=2 voisins ennemis).
        """
        rep = state.rep
        env = rep.env
        i, j = action.data["position"]

        neigh = rep.get_neighbours(i, j).values()

        friend = 0
        enemy = 0
        for n_type, (ni, nj) in neigh:
            if (ni, nj) in env:
                if env[(ni, nj)].piece_type == player_piece:
                    friend += 1
                else:
                    enemy += 1

        # deux voisins amis : connecte deux zones
        # deux voisins ennemis : casse une chaîne potentielle / double menace
        return friend >= 2 or enemy >= 2

    def _register_killer(self, ply: int, action: LightAction) -> None:
        if not self.USE_KILLER_HISTORY:
            return
        killers = self.killer_moves.setdefault(ply, [None, None])
        pos = action.data["position"]

        for k in killers:
            if k is not None and k.data["position"] == pos:
                return  # déjà enregistré

        # shift : nouveau killer en tête
        killers.insert(0, action)
        if len(killers) > 2:
            killers.pop()

    def _update_history(self, player_piece: str, action: LightAction, depth: int) -> None:
        if not self.USE_KILLER_HISTORY:
            return
        pos = action.data["position"]
        key = (player_piece, pos)
        self.history[key] = self.history.get(key, 0.0) + depth * depth

    def _check_time(self, deadline: float) -> None:
        if time.time() >= deadline:
            raise SearchTimeout()


# ============================================================
#                   Classe MyPlayer (Hex)
# ============================================================

class MyPlayer(PlayerHex):
    """
    Player class for Hex game

    Attributes:
        piece_type (str): piece type of the player "R" for the first player and "B" for the second player
    """

    def __init__(self, piece_type: str, name: str = "MyPlayer"):
        """
        Initialize the PlayerHex instance.

        Args:
            piece_type (str): Type of the player's game piece
            name (str, optional): Name of the player (default is "bob")
        """
        super().__init__(piece_type, name)
        self._engine = MinimaxEngine(root_piece=piece_type)

    def compute_action(self, current_state: GameState, remaining_time: int = 1e9, **kwargs) -> Action:
        """
        Use the minimax algorithm to choose the best action based on the heuristic evaluation of game states.

        Args:
            current_state (GameState): The current game state.

        Returns:
            Action: The best action as determined by minimax.
        """
        if not isinstance(current_state, GameStateHex):
            raise ValueError("MyPlayer only supports GameStateHex.")

        # --------------------------------------------------------
        # 1) Micro-solver "coup gagnant en 1"
        #    On parcourt tous les coups légaux, on applique,
        #    et on utilise l'API officielle GameStateHex (scores/is_done)
        #    pour voir si on gagne immédiatement.
        # --------------------------------------------------------
        possible_light_actions = list(current_state.get_possible_light_actions())
        for la in possible_light_actions:
            child = current_state.apply_action(la)
            winner_piece = _winner_from_scores(child)
            if winner_piece == self.piece_type:
                # On ne réfléchit pas plus : mate en 1 trouvé
                return la

        # --------------------------------------------------------
        # 2) Sinon, on lance Minimax P2 (T2+T3+T9-light)
        # --------------------------------------------------------
        total_sec_est = float(remaining_time)
        total_sec_est = max(1.0, min(total_sec_est, 900.0))  # clamp
        # early/mid/late game : on pourrait raffiner, mais baseline simple
        time_budget = max(0.5, min(total_sec_est / 40.0, 5.0))

        try:
            best_light_action = self._engine.choose_action(current_state, time_budget=time_budget)
        except SearchTimeout:
            # En cas de problème, on joue un coup valide au hasard
            actions = list(current_state.get_possible_actions())
            if not actions:
                raise MethodNotImplementedError("No possible actions at timeout.")
            return random.choice(actions)

        # LightAction est déjà accepté par le framework (cf. random_player_hex)
        return best_light_action  # type: ignore