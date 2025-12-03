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
#      Minimax (alpha-beta) + Iterative deepening + TT
#      Heuristique : distance de connexion + connectivité
# ============================================================


class SearchTimeout(Exception):
    """Exception interne pour interrompre proprement la recherche."""
    pass


@dataclass
class TTEntry:
    value: float
    depth: int
    best_action: Optional[LightAction]


class MinimaxEngine:
    """
    Moteur Minimax générique pour Hex, utilisé par MyPlayer.

    - perspective : piece_type de la racine (self.root_piece)
    - TT : clé = (configuration du board, joueur au trait)
    """

    def __init__(self, root_piece: str):
        self.root_piece: str = root_piece   # "R" ou "B"
        self.tt: Dict[Any, TTEntry] = {}
        self.WIN_SCORE = 10_000.0
        self.LOSS_SCORE = -10_000.0
        self.MAX_DEPTH_HARD = 6            # barrière dure de profondeur (baseline T2)
        # paramètres heuristiques (faciles à tuner plus tard)
        self.W_DIST = 3.0
        self.W_CONN = 1.0
        # progression vers l'objectif (bords)
        self.W_PROG = 2.0

    # -----------------------
    # Interface principale
    # -----------------------

    def choose_action(self, state: GameStateHex, time_budget: float) -> LightAction:
        """
        Choisit le meilleur coup avec iterative deepening sous contrainte de temps.

        Args:
            state: état courant (GameStateHex)
            time_budget: temps alloué en secondes

        Returns:
            LightAction: coup choisi
        """
        start_time = time.time()
        deadline = start_time + max(0.05, time_budget * 0.9)  # petite marge de sécurité

        possible_actions = list(state.get_possible_light_actions())
        if not possible_actions:
            raise MethodNotImplementedError("No possible actions in non-terminal state.")

        # Fallback : coup aléatoire (au cas où la recherche n'aboutit pas)
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
                # On garde le meilleur coup de la dernière profondeur complétée
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
        Noeud racine du Minimax (on garde aussi le meilleur coup).
        """
        is_max_player = (state.next_player.get_piece_type() == self.root_piece)
        alpha = float("-inf")
        beta = float("+inf")

        possible_actions = list(state.get_possible_light_actions())
        if not possible_actions:
            # Position terminale ou très bizarre : on évalue
            v = self._evaluate_state(state)
            return v, None

        # Move ordering (simple : centralité) + TT suggestion
        possible_actions = self._order_moves(state, possible_actions)

        best_value = float("-inf") if is_max_player else float("+inf")
        best_action: Optional[LightAction] = None

        for action in possible_actions:
            self._check_time(deadline)
            child_state = state.apply_action(action)

            value = self._alphabeta(
                child_state,
                depth - 1,
                alpha,
                beta,
                not is_max_player,
                deadline,
            )

            if is_max_player:
                if value > best_value:
                    best_value = value
                    best_action = action
                alpha = max(alpha, best_value)
            else:
                if value < best_value:
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
    ) -> float:
        """Minimax récursif avec alpha–beta + TT + cutoff temps."""
        self._check_time(deadline)

        # Terminal / profondeur limite
        winner = self._detect_winner(state)
        if winner is not None:
            if winner == self.root_piece:
                return self.WIN_SCORE
            else:
                return self.LOSS_SCORE

        if depth <= 0:
            return self._evaluate_state(state)

        # Transposition table
        key = self._state_key(state)
        entry = self.tt.get(key)
        if entry is not None and entry.depth >= depth:
            return entry.value

        possible_actions = list(state.get_possible_light_actions())
        if not possible_actions:
            # Pas de coups → évaluation statique
            value = self._evaluate_state(state)
            self.tt[key] = TTEntry(value=value, depth=depth, best_action=None)
            return value

        possible_actions = self._order_moves(state, possible_actions, tt_entry=entry)

        if is_max_player:
            value = float("-inf")
            best_action = None
            for action in possible_actions:
                self._check_time(deadline)
                child_state = state.apply_action(action)
                child_value = self._alphabeta(
                    child_state,
                    depth - 1,
                    alpha,
                    beta,
                    False,
                    deadline,
                )
                if child_value > value:
                    value = child_value
                    best_action = action
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
        else:
            value = float("+inf")
            best_action = None
            for action in possible_actions:
                self._check_time(deadline)
                child_state = state.apply_action(action)
                child_value = self._alphabeta(
                    child_state,
                    depth - 1,
                    alpha,
                    beta,
                    True,
                    deadline,
                )
                if child_value < value:
                    value = child_value
                    best_action = action
                beta = min(beta, value)
                if alpha >= beta:
                    break

        # Mise à jour TT
        self.tt[key] = TTEntry(value=value, depth=depth, best_action=best_action)
        return value

    # -----------------------
    # Heuristique T2
    #  - distance de connexion (type greedy fourni)
    #  - connectivité (voisins amis)
    # -----------------------

    def _evaluate_state(self, state: GameStateHex) -> float:
        """
        h(s) = W_DIST * (d_opp - d_self) + W_CONN * (conn_self - conn_opp)
        plus c'est grand, mieux c'est pour self.root_piece.
        """
        root = self.root_piece
        opp = "B" if root == "R" else "R"

        d_self = self._shortest_connection_distance(state, root)
        d_opp = self._shortest_connection_distance(state, opp)

        conn_self = self._connectivity_score(state, root)
        conn_opp = self._connectivity_score(state, opp)

        # edge progress term: reward stones that are on/near the target edges
        prog_self = self._edge_progress_score(state, root)
        prog_opp = self._edge_progress_score(state, opp)

        # plus d_self est petit → bon ; plus d_opp est grand → bon
        dist_term = (d_opp - d_self)
        conn_term = (conn_self - conn_opp)

        prog_term = (prog_self - prog_opp)

        return (
            self.W_DIST * dist_term
            + self.W_CONN * conn_term
            + self.W_PROG * prog_term
        )

    def _shortest_connection_distance(self, state: GameStateHex, piece_type: str) -> float:
        """
        Approx. distance de connexion à la greedy (Dijkstra discret) :

        - cases amies : coût 0
        - cases vides : coût 1
        - cases ennemies : bloquantes (ignorées)

        Retourne un nombre (plus petit = meilleure connexion).
        """
        rep = state.rep
        env = rep.env
        dim_i, dim_j = rep.dimensions

        INF = 1e9
        dist = [[INF for _ in range(dim_j)] for _ in range(dim_i)]
        pq: List[Tuple[float, Tuple[int, int]]] = []

        objectives: List[Tuple[int, int]] = []

        if piece_type == "R":
            # Connecte top (i=0) à bottom (i=dim_i-1)
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
            # "B" connecte left (j=0) à right (j=dim_j-1)
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
                    # ennemi : on ne traverse pas
                    continue

                if 0 <= ni < dim_i and 0 <= nj < dim_j and new_d < dist[ni][nj]:
                    dist[ni][nj] = new_d
                    heapq.heappush(pq, (new_d, (ni, nj)))

        return best

    def _edge_progress_score(self, state: GameStateHex, piece_type: str) -> float:
        """
        Simple progress score towards edges / goal direction.

        - For R: bonus for stones on top/bottom and progression proportional to i index.
        - For B: bonus for stones on left/right and progression proportional to j index.
        """
        rep = state.rep
        env = rep.env
        dim_i, dim_j = rep.dimensions
        score = 0.0

        for (i, j), piece in env.items():
            if piece.piece_type != piece_type:
                continue
            if piece_type == "R":
                # touching top/bottom
                if i == 0 or i == dim_i - 1:
                    score += 2.0
                # progression towards bottom
                score += float(i) / max(1, (dim_i - 1))
            else:
                if j == 0 or j == dim_j - 1:
                    score += 2.0
                score += float(j) / max(1, (dim_j - 1))

        return score

    def _connectivity_score(self, state: GameStateHex, piece_type: str) -> float:
        """
        Connectivity score based on connected components (groups) instead of simple neighbour pairs.

        For each group we reward the square of its size (bigger groups are much stronger),
        and provide an extra bonus when a group touches both target sides for that player.
        """
        rep = state.rep
        env = rep.env

        visited = set()
        groups = []

        for (i, j), piece in env.items():
            if piece.piece_type != piece_type:
                continue
            if (i, j) in visited:
                continue

            # BFS/DFS to collect the whole group
            stack = [(i, j)]
            group = []
            visited.add((i, j))
            while stack:
                ci, cj = stack.pop()
                group.append((ci, cj))
                for n_type, (ni, nj) in rep.get_neighbours(ci, cj).values():
                    if n_type == piece_type and (ni, nj) not in visited:
                        visited.add((ni, nj))
                        stack.append((ni, nj))
            groups.append(group)

        dim_i, dim_j = rep.dimensions
        score = 0.0
        for group in groups:
            size = len(group)
            # quadratic reward for group size
            score += float(size * size)

            touches_side1 = False
            touches_side2 = False

            if piece_type == "R":
                for (gi, gj) in group:
                    if gi == 0:
                        touches_side1 = True
                    if gi == dim_i - 1:
                        touches_side2 = True
            else:
                for (gi, gj) in group:
                    if gj == 0:
                        touches_side1 = True
                    if gj == dim_j - 1:
                        touches_side2 = True

            if touches_side1 and touches_side2:
                # big bonus for groups that already span the two sides
                score += 5.0 * float(size)

        return score

    # -----------------------
    # Utilitaires : winner, clé TT, move ordering, temps
    # -----------------------

    def _detect_winner(self, state: GameStateHex) -> Optional[str]:
        """
        Détecte un vainqueur sur la base du board uniquement.
        Retourne "R", "B" ou None.
        """
        rep = state.rep
        env = rep.env
        dim = rep.dimensions[0]

        # DFS pour "R" (top -> bottom)
        visited: set[Tuple[int, int]] = set()

        def dfs_bot(i: int, j: int) -> bool:
            if (i, j) in visited:
                return False
            visited.add((i, j))
            if i == dim - 1:
                return True
            for n_type, (ni, nj) in rep.get_neighbours(i, j).values():
                if n_type == "R" and (ni, nj) not in visited:
                    if dfs_bot(ni, nj):
                        return True
            return False

        for j in range(dim):
            piece = env.get((0, j))
            if piece is not None and piece.get_type() == "R":
                if dfs_bot(0, j):
                    return "R"

        # DFS pour "B" (left -> right)
        visited.clear()

        def dfs_right(i: int, j: int) -> bool:
            if (i, j) in visited:
                return False
            visited.add((i, j))
            if j == dim - 1:
                return True
            for n_type, (ni, nj) in rep.get_neighbours(i, j).values():
                if n_type == "B" and (ni, nj) not in visited:
                    if dfs_right(ni, nj):
                        return True
            return False

        for i in range(dim):
            piece = env.get((i, 0))
            if piece is not None and piece.get_type() == "B":
                if dfs_right(i, 0):
                    return "B"

        return None

    def _state_key(self, state: GameStateHex) -> Tuple:
        """
        Clé simple pour TT : (positions triées, joueur au trait).
        C'est pas ultra optimisé mais suffisant en baseline.
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
        tt_entry: Optional[TTEntry] = None,
    ) -> List[LightAction]:
        """
        Move ordering très simple :
        - d'abord le coup TT (si présent),
        - ensuite tri par proximité du centre (centralité géométrique).
        """
        rep = state.rep
        dim_i, dim_j = rep.dimensions
        center_i = (dim_i - 1) / 2.0
        center_j = (dim_j - 1) / 2.0

        tt_move = tt_entry.best_action if tt_entry is not None else None

        scored: List[Tuple[float, LightAction]] = []
        for a in actions:
            pos = a.data["position"]
            i, j = pos
            # distance de Manhattan au centre
            center_dist = abs(i - center_i) + abs(j - center_j)
            score = -center_dist  # plus proche du centre = mieux
            scored.append((score, a))

        scored.sort(key=lambda x: x[0], reverse=True)
        ordered = [a for _, a in scored]

        # Met le coup TT en tête si présent
        if tt_move is not None:
            for idx, a in enumerate(ordered):
                if a.data["position"] == tt_move.data["position"]:
                    ordered.insert(0, ordered.pop(idx))
                    break

        return ordered

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
            name (str, optional): Name of the player (default is "MyPlayer")
        """
        super().__init__(piece_type, name)
        self._engine = MinimaxEngine(root_piece=piece_type)

    def compute_action(self, current_state: GameState, remaining_time: int = 1e9, **kwargs) -> Action:
        """
        Utilise Minimax (alpha–beta + iterative deepening) pour choisir le meilleur coup.

        Args:
            current_state (GameState): current game state (normalement GameStateHex)
            remaining_time (int): temps restant total (unité dépend du framework, on reste défensif)

        Returns:
            Action: Best action found
        """
        if not isinstance(current_state, GameStateHex):
            # En pratique, ce sera déjà un GameStateHex ; sinon on peut lever une erreur.
            raise ValueError("MyPlayer only supports GameStateHex.")

        # Heuristique simple de budget temps :
        # - on suppose 15 minutes totales ≈ 900 s,
        # - on prend une fraction du temps restant avec plafonds.
        # Si remaining_time est en autre unité, ça reste "sûr" (on clippe).
        total_sec_est = float(remaining_time)
        total_sec_est = max(1.0, min(total_sec_est, 900.0))
        time_budget = max(0.5, min(total_sec_est / 40.0, 5.0))

        return self._engine.choose_action(current_state, time_budget=time_budget)