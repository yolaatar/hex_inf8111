from __future__ import annotations

import math
import random
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Set

from player_hex import PlayerHex
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from seahorse.game.light_action import LightAction
from game_state_hex import GameStateHex
from seahorse.utils.custom_exceptions import MethodNotImplementedError


# ============================================================================
# Debug
# ============================================================================

DEBUG_MODE = True
DEBUG_LOG_FILE = "hex_debug.log"


class DebugLogger:
    def __init__(self, enabled: bool = False, filename: str = "hex_debug1.log", agent_file: Optional[str] = None):
        self.enabled = enabled
        self.filename = filename
        # Nom du fichier agent, par défaut dérivé de __file__ si non fourni
        try:
            default_agent = __file__.split("/")[-1]
        except Exception:
            default_agent = "unknown.py"
        self.agent_file = agent_file or default_agent

    def log(self, msg: str) -> None:
        if not self.enabled:
            return
        try:
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            with open(self.filename, "a", encoding="utf-8") as f:
                # Ajoute le tag file=<agent_file> à chaque ligne
                f.write(f"[{ts}] file={self.agent_file} {msg}\n")
        except Exception:
            # Ne jamais casser la partie à cause du debug.
            pass


debug_logger = DebugLogger(enabled=DEBUG_MODE, filename=DEBUG_LOG_FILE)


# ============================================================================
# Exceptions & TT
# ============================================================================

class SearchTimeout(Exception):
    """Raised when the search exceeds the allocated time."""
    pass


@dataclass
class TTEntry:
    value: float
    depth: int
    best_action: Optional[LightAction]
    flag: str  # "EXACT", "LOWER", "UPPER"


def _winner_from_scores(state: GameStateHex) -> Optional[str]:
    """
    Retourne "R" ou "B" si la partie est terminée, sinon None.
    On suppose que state.scores[p.id] == 1.0 indique le vainqueur.
    """
    if not state.is_done():
        return None
    scores = state.scores
    for p in state.players:
        if scores.get(p.id, 0.0) == 1.0:
            return getattr(p, "piece_type", None)
    return None


# ============================================================================
# Moteur Hex V9 : value froide + défense de “ligne droite”
# ============================================================================
class HexEngineV9:
    """
    Moteur de recherche pour Hex :
    - Minimax avec alpha-bêta + itérative deepening.
    - Value centrée sur :
        * distance de connexion (attaque + défense),
        * connectivité locale,
        * pression de frontière,
        * ponts virtuels (nouveau).
    - Move ordering :
        * ponts (pairs de voisins) + ponts virtuels (nouveau),
        * coups au contact,
        * coups qui coupent le plus court chemin adverse
          quand on est en retard dans la course (d_opp << d_self),
        * coups qui coupent ou consolident des ponts virtuels (nouveau).
    """

    def __init__(self, root_piece: str):
        self.root_piece: str = root_piece
        self.opp_piece: str = "B" if root_piece == "R" else "R"

        # Scores de victoire (valeurs énormes pour trancher).
        self.WIN_SCORE = 10_000.0
        self.LOSS_SCORE = -10_000.0

        # Poids d'évaluation (value "froide")
        self.W_DIST = 8.0       # Différence de distance de connexion (très dominant)
        self.W_CONN = 1.0       # Connectivité locale (faible)
        self.W_FRONTIER = 0.6   # Pression sur la frontière (faible)
        self.W_VB = 0.5         # Ponts virtuels (nouveau)

        self.VB_MIN_STONES = 4          # nb min de pierres sur le plateau pour les activer
        self.VB_MIN_FILL_RATIO = 0.05   # % min du plateau rempli (5 %)

        # Renforcement/affaiblissement fort en fin de partie
        self.NEAR_WIN_BONUS_SELF = 900.0
        self.NEAR_WIN_BONUS_OPP = 1000.0

        # Configuration de profondeur
        self.BASE_MAX_DEPTH = 6
        self.MAX_DEPTH_CAP = 10

        # TT (transposition table) en LRU
        self.tt: OrderedDict[int, TTEntry] = OrderedDict()
        self.TT_MAX_SIZE = 100_000

        # Cache distance (board_hash + piece_type -> distance)
        self._distance_cache: OrderedDict[Tuple[int, str], float] = OrderedDict()
        self.DIST_CACHE_LIMIT = 80_000

        # Cache ponts virtuels (board_hash + piece_type -> (bridges, cut_points))
        self._vb_cache: OrderedDict[
            Tuple[int, str],
            Tuple[List[Tuple[Tuple[int, int], Tuple[int, int], List[Tuple[int, int]]]], Dict[Tuple[int, int], int]]
        ] = OrderedDict()
        self.VB_CACHE_LIMIT = 20_000

        # Zobrist hashing
        self._zobrist_tables: Dict[Tuple[int, int], List[List[Tuple[int, int]]]] = {}
        rng = random.Random(1337)
        self._player_hash = {
            "R": rng.getrandbits(64),
            "B": rng.getrandbits(64),
        }
        
                # Danger de course dans un corridor (lignes droites)
        self.W_RACE = 10.0  # à tuner

        # Ouverture
        self.OPENING_MOVE_LIMIT = 5

                # --------------------------------------------------------------
        # V10.1 : défense avancée contre les lignes droites adverses
        # --------------------------------------------------------------
        # Quand considérer qu'il y a "course" intéressante
        self.RACE_WARNING_MARGIN = 2      # on s'inquiète si d_opp <= d_self + 2
        self.RACE_WARNING_MAXD = 6        # et si d_opp <= 6

        # Poids de coupes sur le chemin minimal adverse
        self.W_CUT_PATH_STRONG = 80.0     # quand la course est franchement perdante
        self.W_CUT_PATH_WEAK = 45.0       # simple alerte, mais on reste prudent

        # Coups dans le corridor (autour du chemin minimal)
        self.W_CUT_CORRIDOR = 18.0        # bonus si coup dans le corridor + au contact de bleu

        # Pénalisation de nos propres "lignes droites" perdantes
        self.W_PENALIZE_STRAIGHT = 20.0   # malus pour étendre une ligne perdante dans le corridor

        # Stats de debug
        self.nodes_searched: int = 0
        self.tt_hits: int = 0
        self.max_ply: int = 0
        self.last_completed_depth: int = 0
        self.last_root_value: float = 0.0
        self.last_best_move: Optional[LightAction] = None
        self.last_elapsed: float = 0.0
        # Pour l'heuristique de stabilité à la racine
        self.last_second_best_value: float = 0.0

        # Infos debug fournies par MyPlayer
        self.debug_game_id: Optional[int] = None
        self.debug_move_index: Optional[int] = None

        # ---------------------------------------------------------
        # Limitation du facteur de branchement
        # BRANCHING_LIMITS[depth] = nb max de coups explorés
        # depth = profondeur restante dans l'alpha-bêta
        # Index 0 non utilisé (feuilles)
        # ---------------------------------------------------------
        self.BRANCHING_LIMITS: List[int] = [
            999,  # depth = 0 (non utilisé)
            6,    # depth = 1
            8,    # depth = 2
            12,   # depth = 3
            16,   # depth = 4
            22,   # depth = 5
            28,   # depth = 6
        ]
        # En pratique, si peu de coups, on ne coupe pas
        self.MIN_ACTIONS_NO_CUT: int = 10

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def choose_action(self, state: GameStateHex, time_budget: float) -> LightAction:
        """
        Recherche itérative avec alpha-bêta, sous contrainte de temps.
        """
        start_time = time.time()
        # Utiliser tout le budget alloué pour ce coup (sans le *0.9)
        deadline = start_time + max(0.05, time_budget)

        possible_actions = list(state.get_possible_light_actions())
        if not possible_actions:
            raise MethodNotImplementedError("No possible actions in non-terminal state.")

        # Stats
        self.nodes_searched = 0
        self.tt_hits = 0
        self.max_ply = 0
        self.last_completed_depth = 0
        self.last_root_value = 0.0
        self.last_best_move = None

        is_max_player = (state.next_player.get_piece_type() == self.root_piece)
        best_action = random.choice(possible_actions)
        best_value = float("-inf") if is_max_player else float("+inf")

        depth_limit = self._adaptive_depth_cap(state, time_budget, len(possible_actions))

        # Mode endgame: quand il reste très peu de cases, pousser profondeur
        # et relâcher fortement la réduction de branching.
        dim_i, dim_j = state.rep.dimensions
        nb_empty_global = dim_i * dim_j - len(state.rep.env)
        saved_branching_limits = None
        force_endgame_mode = False
        # élargir le seuil d'endgame et aussi déclencher si une course est très courte
        d_self_now = self._shortest_connection_distance(state, self.root_piece)
        d_opp_now = self._shortest_connection_distance(state, self.opp_piece)
        if nb_empty_global <= 30 or d_self_now <= 3 or d_opp_now <= 3:
            force_endgame_mode = True
            depth_limit = min(self.MAX_DEPTH_CAP, max(depth_limit, self.MAX_DEPTH_CAP))
            saved_branching_limits = list(self.BRANCHING_LIMITS)
            self.BRANCHING_LIMITS = [999] * len(self.BRANCHING_LIMITS)

        # Soft cap basé sur la phase (nb de cases vides)
        dim_i, dim_j = state.rep.dimensions
        nb_empty = dim_i * dim_j - len(state.rep.env)
        hard_cap = time_budget * 0.98
        # Soft cap agressif: on vise à utiliser presque tout le budget alloué
        if nb_empty > 140:
            soft_ratio = 0.9
        elif nb_empty > 80:
            soft_ratio = 0.9
        else:
            soft_ratio = 0.9
        soft_cap = min(hard_cap, time_budget * soft_ratio)

        depth = 1
        prev_best_action: Optional[LightAction] = None
        stable_count = 0
        while depth <= depth_limit:
            try:
                value, action = self._search_root(state, depth, deadline)
                if action is not None:
                    best_action = action
                    best_value = value
                    self.last_completed_depth = depth
                    self.last_root_value = value
                    self.last_best_move = action
                    # suivi stabilité du coup racine
                    if prev_best_action is not None and action.data["position"] == prev_best_action.data["position"]:
                        stable_count += 1
                    else:
                        stable_count = 0
                    prev_best_action = action

                if DEBUG_MODE:
                    debug_logger.log(
                        "EVENT=iter "
                        f"game_id={self.debug_game_id} move={self.debug_move_index} "
                        f"depth={depth} val={value:.2f} "
                        f"nodes={self.nodes_searched} tt_hits={self.tt_hits}"
                    )
                # Arrêt anticipé si position stable et soft_cap atteint
                elapsed_so_far = time.time() - start_time
                margin = abs(self.last_root_value - self.last_second_best_value)
                margin_large = margin >= 400.0
                best_move_stable = stable_count >= 2
                if elapsed_so_far >= soft_cap and margin_large and best_move_stable:
                    break
                depth += 1
            except SearchTimeout:
                break

        # Restaurer les limites de branching si on a activé le mode endgame
        if force_endgame_mode and saved_branching_limits is not None:
            self.BRANCHING_LIMITS = saved_branching_limits

        self.last_elapsed = time.time() - start_time
        return best_action

    def suggest_opening_action(self, state: GameStateHex, actions: List[LightAction]) -> Optional[LightAction]:
        """
        Simple préférence d'ouverture : cases proches du centre et alignées avec notre axe.
        """
        if len(state.rep.env) >= self.OPENING_MOVE_LIMIT:
            return None

        rep = state.rep
        env = rep.env
        # Phase de jeu : on n'active les ponts virtuels qu'après un certain stade

        dim_i, dim_j = rep.dimensions
        center_i = (dim_i - 1) / 2.0
        center_j = (dim_j - 1) / 2.0

        scored: List[Tuple[float, LightAction]] = []
        for a in actions:
            i, j = a.data["position"]
            if (i, j) in env:
                continue
            center_dist = abs(i - center_i) + abs(j - center_j)
            score = -center_dist

            if self.root_piece == "R":
                axis_align = 1.0 - abs(j - center_j) / (center_j + 1.0)
            else:
                axis_align = 1.0 - abs(i - center_i) / (center_i + 1.0)

            score += 2.0 * axis_align
            scored.append((score, a))

        if not scored:
            return None

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    def find_winning_action(self, state: GameStateHex, piece_type: str) -> Optional[LightAction]:
        """
        Vérifie s’il existe un coup gagnant en un seul coup pour 'piece_type'.
        """
        for la in state.get_possible_light_actions():
            child = state.apply_action(la)
            winner_piece = _winner_from_scores(child)
            if winner_piece == piece_type:
                return la
        return None

    # ------------------------------------------------------------------
    # Search core
    # ------------------------------------------------------------------

    def _search_root(
        self,
        state: GameStateHex,
        depth: int,
        deadline: float,
    ) -> Tuple[float, Optional[LightAction]]:
        self._check_time(deadline)

        is_max_player = (state.next_player.get_piece_type() == self.root_piece)
        alpha = float("-inf")
        beta = float("+inf")

        actions = list(state.get_possible_light_actions())
        if not actions:
            v = self._evaluate_state(state)
            return v, None

        tt_entry = self._get_tt_entry(self._hash_key(state))
        actions = self._order_moves(state, actions, tt_entry)
        # Limite de branchement après tri
        actions = self._limit_branching(actions, depth)

        best_value = float("-inf") if is_max_player else float("+inf")
        best_action: Optional[LightAction] = None
        # suivi du deuxième meilleur pour mesurer la marge de la racine
        second_best_value = float("-inf") if is_max_player else float("+inf")

        for action in actions:
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
                    if best_action is not None:
                        second_best_value = max(second_best_value, best_value)
                    best_value = value
                    best_action = action
                else:
                    second_best_value = max(second_best_value, value)
                alpha = max(alpha, best_value)
            else:
                if value < best_value or best_action is None:
                    if best_action is not None:
                        second_best_value = min(second_best_value, best_value)
                    best_value = value
                    best_action = action
                else:
                    second_best_value = min(second_best_value, value)
                beta = min(beta, best_value)

            if alpha >= beta:
                break

        # Expose le 2e meilleur pour la stabilité
        self.last_second_best_value = second_best_value
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
        self._check_time(deadline)
        self.nodes_searched += 1
        if ply > self.max_ply:
            self.max_ply = ply

        winner = _winner_from_scores(state)
        if winner is not None:
            return self.WIN_SCORE if winner == self.root_piece else self.LOSS_SCORE

        if depth <= 0:
            return self._evaluate_state(state)

        key = self._hash_key(state)
        alpha_orig = alpha
        beta_orig = beta

        entry = self._get_tt_entry(key)
        if entry is not None and entry.depth >= depth:
            self.tt_hits += 1
            if entry.flag == "EXACT":
                return entry.value
            if entry.flag == "LOWER":
                alpha = max(alpha, entry.value)
            elif entry.flag == "UPPER":
                beta = min(beta, entry.value)
            if alpha >= beta:
                return entry.value

        actions = list(state.get_possible_light_actions())
        if not actions:
            value = self._evaluate_state(state)
            self._store_tt(key, TTEntry(value=value, depth=depth, best_action=None, flag="EXACT"))
            return value

        actions = self._order_moves(state, actions, entry)
        # Limite de branchement après tri
        actions = self._limit_branching(actions, depth)

        value = float("-inf") if is_max_player else float("+inf")
        best_action: Optional[LightAction] = None

        for action in actions:
            self._check_time(deadline)
            child_state = state.apply_action(action)
            child_value = self._alphabeta(
                child_state,
                depth - 1,
                alpha,
                beta,
                not is_max_player,
                deadline,
                ply=ply + 1,
            )

            if is_max_player:
                if child_value > value or best_action is None:
                    value = child_value
                    best_action = action
                alpha = max(alpha, value)
            else:
                if child_value < value or best_action is None:
                    value = child_value
                    best_action = action
                beta = min(beta, value)

            if alpha >= beta:
                break

        flag = "EXACT"
        if value <= alpha_orig:
            flag = "UPPER"
        elif value >= beta_orig:
            flag = "LOWER"

        self._store_tt(key, TTEntry(value=value, depth=depth, best_action=best_action, flag=flag))
        return value

    # ------------------------------------------------------------------
    # Évaluation (value froide) + ponts virtuels
    # ------------------------------------------------------------------

    def _evaluate_state(self, state: GameStateHex) -> float:
        """
        Heuristique principale V6+ :
        - d_self / d_opp = nombre de cases vides nécessaires pour connecter (signal dominant).
        - connectivité locale des pierres.
        - pression sur la "frontière" (cases vides voisines).
        - ponts virtuels (connexion virtuelle entre groupes).
        """
        root = self.root_piece
        opp = self.opp_piece

        d_self = self._shortest_connection_distance(state, root)
        d_opp = self._shortest_connection_distance(state, opp)

        conn_self = self._connectivity_score(state, root)
        conn_opp = self._connectivity_score(state, opp)

        front_self, front_opp = self._frontier_scores(state, root, opp)

        vb_self = self._virtual_bridge_score(state, root)
        vb_opp = self._virtual_bridge_score(state, opp)

        dist_term = (d_opp - d_self)
        conn_term = (conn_self - conn_opp)
        frontier_term = (front_self - front_opp)
        vb_term = (vb_self - vb_opp)

        value = (
            self.W_DIST * dist_term
            + self.W_CONN * conn_term
            + self.W_FRONTIER * frontier_term
            + self.W_VB * vb_term
        )

        # === Nouveau : danger de course dans un corridor ===
        race_term = self._corridor_race_eval(state)
        value += self.W_RACE * race_term

        # Bonus/pénalité très forts quand on est proche de connecter
        if d_self <= 2:
            value += self.NEAR_WIN_BONUS_SELF + 150.0 * (2 - d_self)
        if d_opp <= 3:
            value -= self.NEAR_WIN_BONUS_OPP + 180.0 * (3 - d_opp)

        return value
       

    # --- Distance de connexion (0–1 BFS) -------------------------------------

    def _shortest_connection_distance(self, state: GameStateHex, piece_type: str) -> float:
        """
        Nombre minimal de cases vides nécessaires pour connecter nos deux côtés.
        BFS 0/1 avec cache.
        """
        key = (self._board_hash(state), piece_type)
        cached = self._distance_cache.get(key)
        if cached is not None:
            self._distance_cache.move_to_end(key)
            return cached

        value = self._compute_shortest_connection_distance(state, piece_type)
        self._distance_cache[key] = value
        self._distance_cache.move_to_end(key)
        if len(self._distance_cache) > self.DIST_CACHE_LIMIT:
            self._distance_cache.popitem(last=False)
        return value

    def _compute_shortest_connection_distance(self, state: GameStateHex, piece_type: str) -> float:
        rep = state.rep
        env = rep.env
        dim_i, dim_j = rep.dimensions

        INF = dim_i + dim_j + 5
        dist = [[INF for _ in range(dim_j)] for _ in range(dim_i)]
        dq: deque[Tuple[int, int]] = deque()
        goals: Set[Tuple[int, int]] = set()

        # --- Adjacency de ponts virtuels (entre NOS pierres) ---
        # On traitera ces voisins comme des pierres de même couleur à coût 0.
        vb_adj = self._get_vb_adjacency(state, piece_type)

        # R connecte haut-bas, B gauche-droite
        if piece_type == "R":
            for j in range(dim_j):
                goals.add((dim_i - 1, j))
                cell = env.get((0, j))
                if cell is None:
                    dist[0][j] = 1
                    dq.append((0, j))
                elif cell.piece_type == piece_type:
                    dist[0][j] = 0
                    dq.appendleft((0, j))
        else:
            for i in range(dim_i):
                goals.add((i, dim_j - 1))
                cell = env.get((i, 0))
                if cell is None:
                    dist[i][0] = 1
                    dq.append((i, 0))
                elif cell.piece_type == piece_type:
                    dist[i][0] = 0
                    dq.appendleft((i, 0))

        while dq:
            i, j = dq.popleft()
            if (i, j) in goals:
                return dist[i][j]

            # --- Voisins hexagonaux "classiques" ---
            for n_type, (ni, nj) in rep.get_neighbours(i, j).values():
                if not (0 <= ni < dim_i and 0 <= nj < dim_j):
                    continue

                if n_type == piece_type:
                    new_d = dist[i][j]          # traverser une pierre à nous : coût 0
                elif n_type == "EMPTY":
                    new_d = dist[i][j] + 1      # occuper une case vide : coût 1
                else:
                    continue                     # pierre adverse : bloquant

                if new_d < dist[ni][nj]:
                    dist[ni][nj] = new_d
                    if n_type == piece_type:
                        dq.appendleft((ni, nj))
                    else:
                        dq.append((ni, nj))

            # --- Voisins via ponts virtuels (A <-> B) ---
            # Si (i, j) est une pierre à nous qui a des ponts virtuels vers d'autres
            # pierres, on peut "sauter" gratuitement vers ces autres pierres.
            pos = (i, j)
            if pos in vb_adj:
                for (vi, vj) in vb_adj[pos]:
                    # sécurité : on vérifie que c'est bien une pierre à nous
                    cell_v = env.get((vi, vj))
                    if cell_v is None or cell_v.piece_type != piece_type:
                        continue

                    new_d = dist[i][j]  # coût 0 : pont virtuel = connexion "gratuite"
                    if new_d < dist[vi][vj]:
                        dist[vi][vj] = new_d
                        dq.appendleft((vi, vj))

        return INF

    # --- Chemin le plus court : cellules vides critiques ----------------------

    def _shortest_path_empty_cells(self, state: GameStateHex, piece_type: str) -> Set[Tuple[int, int]]:
        """
        Renvoie un ensemble de cases vides appartenant à un des plus courts
        chemins de connexion pour 'piece_type'.

        Version VC-aware :
        - comme avant, mais on ajoute des transitions de coût 0 entre pierres
          reliées par un pont virtuel.
        """
        rep = state.rep
        env = rep.env
        dim_i, dim_j = rep.dimensions

        INF = dim_i + dim_j + 5
        dist = [[INF for _ in range(dim_j)] for _ in range(dim_i)]
        parent: List[List[Optional[Tuple[int, int]]]] = [
            [None for _ in range(dim_j)] for _ in range(dim_i)
        ]
        dq: deque[Tuple[int, int]] = deque()
        goals: Set[Tuple[int, int]] = set()

        # Adjacency de ponts virtuels
        vb_adj = self._get_vb_adjacency(state, piece_type)

        # R connecte haut-bas, B gauche-droite
        if piece_type == "R":
            for j in range(dim_j):
                goals.add((dim_i - 1, j))
                cell = env.get((0, j))
                if cell is None:
                    dist[0][j] = 1
                    dq.append((0, j))
                elif cell.piece_type == piece_type:
                    dist[0][j] = 0
                    dq.appendleft((0, j))
        else:
            for i in range(dim_i):
                goals.add((i, dim_j - 1))
                cell = env.get((i, 0))
                if cell is None:
                    dist[i][0] = 1
                    dq.append((i, 0))
                elif cell.piece_type == piece_type:
                    dist[i][0] = 0
                    dq.appendleft((i, 0))

        best_goal: Optional[Tuple[int, int]] = None

        while dq:
            i, j = dq.popleft()
            if (i, j) in goals:
                best_goal = (i, j)
                break

            # Voisins classiques
            for n_type, (ni, nj) in rep.get_neighbours(i, j).values():
                if not (0 <= ni < dim_i and 0 <= nj < dim_j):
                    continue

                if n_type == piece_type:
                    new_d = dist[i][j]
                elif n_type == "EMPTY":
                    new_d = dist[i][j] + 1
                else:
                    continue

                if new_d < dist[ni][nj]:
                    dist[ni][nj] = new_d
                    parent[ni][nj] = (i, j)
                    if n_type == piece_type:
                        dq.appendleft((ni, nj))
                    else:
                        dq.append((ni, nj))

            # Voisins via ponts virtuels
            pos = (i, j)
            if pos in vb_adj:
                for (vi, vj) in vb_adj[pos]:
                    cell_v = env.get((vi, vj))
                    if cell_v is None or cell_v.piece_type != piece_type:
                        continue

                    new_d = dist[i][j]
                    if new_d < dist[vi][vj]:
                        dist[vi][vj] = new_d
                        parent[vi][vj] = (i, j)
                        dq.appendleft((vi, vj))

        if best_goal is None:
            return set()

        # Reconstruction du chemin
        path: List[Tuple[int, int]] = []
        cur = best_goal
        while cur is not None:
            path.append(cur)
            ci, cj = cur
            cur = parent[ci][cj]

        empties: Set[Tuple[int, int]] = set()
        for pos in path:
            if pos not in env:  # cellule vide sur un plus court chemin
                empties.add(pos)
        return empties

    # --- Connectivité locale --------------------------------------------------

    def _connectivity_score(self, state: GameStateHex, piece_type: str) -> float:
        """
        Score de connectivité :
        - +0.5 par voisin ami,
        - -0.5 par pierre isolée.
        """
        rep = state.rep
        env = rep.env
        score_pairs = 0.0
        isolated_penalty = 0.0

        for (i, j), piece in env.items():
            if piece.piece_type != piece_type:
                continue

            friend_neigh = 0
            for _, (ni, nj) in rep.get_neighbours(i, j).values():
                occ = env.get((ni, nj))
                if occ is not None and occ.piece_type == piece_type:
                    friend_neigh += 1
                    score_pairs += 0.5

            if friend_neigh == 0:
                isolated_penalty += 1.0

        return score_pairs - 0.5 * isolated_penalty

    # --- Frontière / pression -------------------------------------------------

    def _frontier_scores(
        self,
        state: GameStateHex,
        root: str,
        opp: str,
    ) -> Tuple[float, float]:
        """
        Score de "frontière" : cases vides adjacentes à des pierres.
        - On valorise les cases vides qui ont plus de voisins amis que d'ennemis.
        - Séparé pour root et opp.
        """
        rep = state.rep
        env = rep.env
        dim_i, dim_j = rep.dimensions

        front_root = 0.0
        front_opp = 0.0

        for i in range(dim_i):
            for j in range(dim_j):
                if (i, j) in env:
                    continue

                friend_root = 0
                friend_opp = 0
                for _, (ni, nj) in rep.get_neighbours(i, j).values():
                    occ = env.get((ni, nj))
                    if occ is None:
                        continue
                    if occ.piece_type == root:
                        friend_root += 1
                    elif occ.piece_type == opp:
                        friend_opp += 1

                if friend_root + friend_opp == 0:
                    continue

                if friend_root > friend_opp:
                    front_root += friend_root - 0.5 * friend_opp
                elif friend_opp > friend_root:
                    front_opp += friend_opp - 0.5 * friend_root

        return front_root, front_opp

    # --- Ponts virtuels -------------------------------------------------------

    def _find_virtual_bridges(
        self,
        state: GameStateHex,
        piece_type: str,
    ) -> Tuple[
        List[Tuple[Tuple[int, int], Tuple[int, int], List[Tuple[int, int]]]],
        Dict[Tuple[int, int], int],
    ]:
        """
        Détection de ponts virtuels pour 'piece_type'.

        Un pont virtuel est ici défini comme deux pierres A et B de même couleur
        ayant exactement deux voisins communs X et Y, tels que X et Y ne sont
        pas occupés par l'adversaire. L'adversaire doit jouer sur X ET Y pour
        couper la connexion virtuelle entre A et B.

        Retourne :
        - bridges: liste de (A, B, [X, Y])
        - cut_points: dict position -> nombre de ponts virtuels passant par cette case
        """
        board_key = (self._board_hash(state), piece_type)
        cached = self._vb_cache.get(board_key)
        if cached is not None:
            self._vb_cache.move_to_end(board_key)
            return cached

        rep = state.rep
        env = rep.env

        # Phase de jeu : on n'active les ponts virtuels qu'après un certain stade
        num_stones = len(env)
        dim_i, dim_j = rep.dimensions
        board_size = dim_i * dim_j
        fill_ratio = num_stones / board_size if board_size > 0 else 0.0

        if num_stones < self.VB_MIN_STONES or fill_ratio < self.VB_MIN_FILL_RATIO:
            # Trop tôt dans la partie : pas de ponts virtuels
            return [], {}

        positions: List[Tuple[int, int]] = [
            pos for pos, cell in env.items() if cell.piece_type == piece_type
        ]

        bridges: List[Tuple[Tuple[int, int], Tuple[int, int], List[Tuple[int, int]]]] = []
        cut_points: Dict[Tuple[int, int], int] = {}

        # Pour éviter les doublons A-B / B-A
        visited_pairs: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()

        for a in positions:
            ai, aj = a
            neigh_a = {nb for _, nb in rep.get_neighbours(ai, aj).values()}

            candidate_bs: Set[Tuple[int, int]] = set()
            for nb in neigh_a:
                ni, nj = nb
                for _, nb2 in rep.get_neighbours(ni, nj).values():
                    candidate_bs.add(nb2)

            for b in candidate_bs:
                if b == a:
                    continue

                cell_b = env.get(b)
                if cell_b is None or cell_b.piece_type != piece_type:
                    continue

                # Canonicalisation du couple (a, b) pour éviter doublons
                pair = (a, b) if a < b else (b, a)
                if pair in visited_pairs:
                    continue

                bi, bj = b
                neigh_b = {nb for _, nb in rep.get_neighbours(bi, bj).values()}
                common = neigh_a & neigh_b

                # Pont virtuel "classique" : exactement 2 voisins communs
                if len(common) != 2:
                    continue

                cuts: List[Tuple[int, int]] = []
                valid_bridge = True
                for c in common:
                    occ = env.get(c)
                    # Si occupé par l'adversaire, ce n'est plus un vrai pont virtuel utilisable
                    if occ is not None and occ.piece_type != piece_type:
                        valid_bridge = False
                        break
                    cuts.append(c)

                if not valid_bridge:
                    continue

                visited_pairs.add(pair)
                bridges.append((pair[0], pair[1], cuts))
                for c in cuts:
                    cut_points[c] = cut_points.get(c, 0) + 1

        # Mise en cache (LRU)
        self._vb_cache[board_key] = (bridges, cut_points)
        self._vb_cache.move_to_end(board_key)
        if len(self._vb_cache) > self.VB_CACHE_LIMIT:
            self._vb_cache.popitem(last=False)

        return bridges, cut_points
    
    def _find_broken_virtual_bridges(
        self,
        state: GameStateHex,
        piece_type: str,
    ) -> Dict[Tuple[int, int], int]:
        """
        Détection de *ponts virtuels cassés* pour 'piece_type'.

        Cas visé :
        - Deux pierres A et B de même couleur.
        - Elles ont exactement deux voisins communs C1 et C2.
        - Parmi {C1, C2} :
            * l'un est occupé par l'adversaire,
            * l'autre est encore vide.

        La case vide est une *case de réparation* : y jouer connecte de
        manière très solide A et B (réponse standard après que l'adversaire
        ait "attaqué" le pont).

        Retourne :
        - rescue_points: dict position -> nombre de motifs de ce type
          pour lesquels cette case est une réparation.
        """
        rep = state.rep
        env = rep.env
        opp_piece = "B" if piece_type == "R" else "R"

        # Liste de nos pierres
        positions: List[Tuple[int, int]] = [
            pos for pos, cell in env.items() if cell.piece_type == piece_type
        ]

        rescue_points: Dict[Tuple[int, int], int] = {}
        visited_pairs: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()

        for a in positions:
            ai, aj = a
            neigh_a = {nb for _, nb in rep.get_neighbours(ai, aj).values()}

            candidate_bs: Set[Tuple[int, int]] = set()
            for nb in neigh_a:
                ni, nj = nb
                for _, nb2 in rep.get_neighbours(ni, nj).values():
                    candidate_bs.add(nb2)

            for b in candidate_bs:
                if b == a:
                    continue

                cell_b = env.get(b)
                if cell_b is None or cell_b.piece_type != piece_type:
                    continue

                pair = (a, b) if a < b else (b, a)
                if pair in visited_pairs:
                    continue
                visited_pairs.add(pair)

                bi, bj = b
                neigh_b = {nb for _, nb in rep.get_neighbours(bi, bj).values()}
                common = neigh_a & neigh_b

                # Motif "diamond" uniquement
                if len(common) != 2:
                    continue

                c1, c2 = tuple(common)
                occ1 = env.get(c1)
                occ2 = env.get(c2)

                def is_enemy(occ):
                    return occ is not None and occ.piece_type == opp_piece

                def is_friend(occ):
                    return occ is not None and occ.piece_type == piece_type

                def is_empty(pos):
                    return pos not in env

                # Cas 1 : C1 ennemi, C2 vide -> C2 est la réparation
                if is_enemy(occ1) and not is_enemy(occ2):
                    if is_empty(c2) and not is_friend(occ2):
                        rescue_points[c2] = rescue_points.get(c2, 0) + 1

                # Cas 2 : C2 ennemi, C1 vide -> C1 est la réparation
                if is_enemy(occ2) and not is_enemy(occ1):
                    if is_empty(c1) and not is_friend(occ1):
                        rescue_points[c1] = rescue_points.get(c1, 0) + 1

        return rescue_points
    def _get_vb_adjacency(
        self,
        state: GameStateHex,
        piece_type: str,
    ) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
        """
        Construit une adjacency de ponts virtuels pour 'piece_type'.

        Pour chaque pont virtuel (A, B, cuts), on ajoute une arête non orientée
        A <-> B dans un petit graphe interne.

        Ensuite, dans la distance de connexion, on traitera ces arêtes comme
        des voisins supplémentaires de coût 0 (comme une pierre déjà connectée).
        """
        bridges, _ = self._find_virtual_bridges(state, piece_type)
        vb_adj: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}

        for a, b, _cuts in bridges:
            vb_adj.setdefault(a, []).append(b)
            vb_adj.setdefault(b, []).append(a)

        return vb_adj

    def _virtual_bridge_score(self, state: GameStateHex, piece_type: str) -> float:
        """
        Score de ponts virtuels pour 'piece_type'.

        Idée :
        - +1.5 par pont virtuel détecté.
        - +0.5 supplémentaire par point de coupure déjà occupé par nous
          (pont "renforcé").
        """
        bridges, cut_points = self._find_virtual_bridges(state, piece_type)
        env = state.rep.env

        score = 1.5 * len(bridges)

        for pos, cnt in cut_points.items():
            cell = env.get(pos)
            # Si on possède déjà un des points de coupure, c'est encore plus solide
            if cell is not None and cell.piece_type == piece_type:
                score += 0.5 * cnt

        return score
    
        # ------------------------------------------------------------------
    # V10 : analyse de "course" dans un corridor (lignes droites)
    # ------------------------------------------------------------------

    def _corridor_reach_costs(
        self,
        state: GameStateHex,
        piece_type: str,
        corridor: Set[Tuple[int, int]],
    ) -> Dict[Tuple[int, int], int]:
        """
        BFS dans un corridor : coût minimal (en nombre de coups) pour
        contrôler chaque case du corridor pour 'piece_type'.

        Règles :
        - traverser une pierre amie : coût 0,
        - jouer sur une case vide : +1,
        - pierres adverses : bloquantes (on ne passe pas à travers).
        """
        rep = state.rep
        env = rep.env

        INF = 10**9
        costs: Dict[Tuple[int, int], int] = {pos: INF for pos in corridor}
        dq: deque[Tuple[int, int]] = deque()

        # sources : toutes nos pierres présentes dans le corridor
        for pos in corridor:
            cell = env.get(pos)
            if cell is not None and cell.piece_type == piece_type:
                costs[pos] = 0
                dq.append(pos)

        # Si aucune pierre de cette couleur dans le corridor, on laisse INF
        if not dq:
            return costs

        while dq:
            i, j = dq.popleft()
            cur = costs[(i, j)]

            for _, (ni, nj) in rep.get_neighbours(i, j).values():
                npos = (ni, nj)
                if npos not in corridor:
                    continue

                occ = env.get(npos)
                if occ is not None and occ.piece_type == piece_type:
                    new_cost = cur
                elif occ is None:
                    new_cost = cur + 1
                else:
                    # pierre adverse : mur
                    continue

                if new_cost < costs[npos]:
                    costs[npos] = new_cost
                    if occ is not None and occ.piece_type == piece_type:
                        dq.appendleft(npos)
                    else:
                        dq.append(npos)

        return costs

    def _analyze_race_in_corridor(
        self,
        state: GameStateHex,
        player_piece: str,
    ) -> Tuple[Optional[Set[Tuple[int, int]]], Optional[Set[Tuple[int, int]]], float]:
        """
        Analyse d'une "course" locale dans un corridor autour du plus court
        chemin adverse.

        Retourne :
        - corridor : ensemble de cases impliquées dans la course (ou None),
        - opp_path : cases vides d'un plus court chemin adverse (ou None),
        - race_score : >0 si le joueur au trait est globalement plus rapide,
                       <0 si l'adversaire est plus rapide sur ce chemin.
        """
        opp_piece = "B" if player_piece == "R" else "R"

        d_player = self._shortest_connection_distance(state, player_piece)
        d_opp = self._shortest_connection_distance(state, opp_piece)

        # 🔥 Nouveau critère : on commence à s'intéresser au corridor
        # dès que la ligne adverse devient "raisonnablement courte".
        # - ancien : d_opp <= d_player + 1 and d_opp <= 6
        # - nouveau : dès que d_opp est petit (<= 8) OU pas beaucoup plus long que nous.
        if not (d_opp <= 8 or d_opp <= d_player + 2):
            return None, None, 0.0

        opp_path = self._shortest_path_empty_cells(state, opp_piece)
        if not opp_path:
            return None, None, 0.0

        rep = state.rep
        dim_i, dim_j = rep.dimensions

        # Corridor = chemin critique adverse + anneau de voisins
        # Corridor = chemin critique adverse + 2 anneaux de voisins
        corridor: Set[Tuple[int, int]] = set(opp_path)
        frontier: Set[Tuple[int, int]] = set(opp_path)

        # On étend le corridor sur 2 "couches" de voisins
        for _ in range(2):
            new_frontier: Set[Tuple[int, int]] = set()
            for (i, j) in frontier:
                for _, (ni, nj) in rep.get_neighbours(i, j).values():
                    if 0 <= ni < dim_i and 0 <= nj < dim_j and (ni, nj) not in corridor:
                        corridor.add((ni, nj))
                        new_frontier.add((ni, nj))
            frontier = new_frontier

        # Coûts de prise de contrôle de chaque case pour les deux joueurs
        costs_player = self._corridor_reach_costs(state, player_piece, corridor)
        costs_opp = self._corridor_reach_costs(state, opp_piece, corridor)

        INF = 10**9
        score = 0.0
        for pos in opp_path:
            cp = costs_player.get(pos, INF)
            co = costs_opp.get(pos, INF)
            if co < cp:
                score -= 1.0  # l'adversaire est plus rapide sur cette case
            elif cp < co:
                score += 1.0  # nous sommes plus rapides

        return corridor, opp_path, score
    
    def _corridor_race_eval(self, state: GameStateHex) -> float:
        """
        Évalue la course dans un corridor critique du point de vue de root_piece.

        Si l'adversaire est globalement plus rapide sur son chemin critique,
        on renvoie une grosse valeur négative (danger). Si on est plus rapide,
        petite valeur positive.
        """
        root = self.root_piece
        opp  = self.opp_piece

        # On regarde la course "adversaire vs root", donc player_piece = root
        corridor, opp_path, race_score = self._analyze_race_in_corridor(state, root)

        if corridor is None or not opp_path:
            return 0.0

        # race_score > 0 : root est plus rapide sur ce chemin
        # race_score < 0 : l'adversaire est plus rapide
        # On amplifie fortement le cas défensif
        if race_score < 0:
            # Danger : on met une pénalité proportionnelle
            return 3.0 * race_score  # négatif
        elif race_score > 0:
            # Bonus plus doux si on domine la course
            return 1.0 * race_score
        else:
            return 0.0

    # ------------------------------------------------------------------
    # Zobrist, TT, move ordering
    # ------------------------------------------------------------------

    def _hash_key(self, state: GameStateHex) -> int:
        board_hash = self._board_hash(state)
        player_hash = self._player_hash[state.next_player.get_piece_type()]
        return board_hash ^ player_hash

    def _board_hash(self, state: GameStateHex) -> int:
        rep = state.rep
        env = rep.env
        dim_i, dim_j = rep.dimensions
        self._ensure_zobrist(dim_i, dim_j)
        table = self._zobrist_tables[(dim_i, dim_j)]

        h = 0
        for (i, j), piece in env.items():
            idx = 0 if piece.piece_type == "R" else 1
            h ^= table[i][j][idx]
        return h

    def _ensure_zobrist(self, dim_i: int, dim_j: int) -> None:
        key = (dim_i, dim_j)
        if key in self._zobrist_tables:
            return
        rng = random.Random(hash(key) ^ 0x9E3779B185EBCA87)
        table: List[List[Tuple[int, int]]] = []
        for i in range(dim_i):
            row: List[Tuple[int, int]] = []
            for j in range(dim_j):
                row.append((rng.getrandbits(64), rng.getrandbits(64)))
            table.append(row)
        self._zobrist_tables[key] = table

    def _get_tt_entry(self, key: int) -> Optional[TTEntry]:
        entry = self.tt.get(key)
        if entry is not None:
            self.tt.move_to_end(key)
        return entry

    def _store_tt(self, key: int, entry: TTEntry) -> None:
        self.tt[key] = entry
        self.tt.move_to_end(key)
        if len(self.tt) > self.TT_MAX_SIZE:
            self.tt.popitem(last=False)

    def _order_moves(
        self,
        state: GameStateHex,
        actions: List[LightAction],
        tt_entry: Optional[TTEntry] = None,
    ) -> List[LightAction]:
        """
        Move ordering :
        - Coup TT en premier.
        - Cases proches du centre.
        - Cases adjacentes à nos pierres (attaque) ou à celles de l’adversaire (défense).
        - Priorité aux :
            * coups qui complètent nos ponts (un pion sur deux),
            * coups qui cassent des ponts adverses,
            * coups qui coupent le plus court chemin adverse
              quand la course d_opp vs d_player est critique,
            * coups qui coupent des ponts virtuels adverses,
            * coups qui consolident / complètent nos ponts virtuels,
            * V10 : coupes solides dans le corridor si la ligne droite
                    adverse est (ou devient) dangereuse.
        - Légère préférence pour avancer dans l’axe de connexion.
        """
        rep = state.rep
        env = state.rep.env
        dim_i, dim_j = state.rep.dimensions
        center_i = (dim_i - 1) / 2.0
        center_j = (dim_j - 1) / 2.0

        player_piece = state.next_player.get_piece_type()
        opp_piece = "B" if player_piece == "R" else "R"

        # --- Distances globales ---
        d_player = self._shortest_connection_distance(state, player_piece)
        d_opp = self._shortest_connection_distance(state, opp_piece)

        # Alerte "course dangereuse" un peu plus tôt que V10
        race_warning = (
            d_opp <= d_player + self.RACE_WARNING_MARGIN
            and d_opp <= self.RACE_WARNING_MAXD
        )

        # Course franchement critique (V10)
        race_critical = (d_opp + 1 < d_player) and (d_opp <= 4)

        # Chemin minimal adverse (cases vides critiques)
        opp_path_empties: Set[Tuple[int, int]] = set()
        if race_warning:  # on le calcule dès l'alerte
            opp_path_empties = self._shortest_path_empty_cells(state, opp_piece)

        # --- V10.1 : analyse de course dans un corridor (ligne droite adverse) ---
        corridor: Optional[Set[Tuple[int, int]]] = None
        opp_path: Optional[Set[Tuple[int, int]]] = None
        race_score: float = 0.0

        empties = dim_i * dim_j - len(env)
        if empties < dim_i * dim_j * 0.9:
            corridor, opp_path, race_score = self._analyze_race_in_corridor(state, player_piece)

        race_losing_for_player = (corridor is not None and race_score <= -1.0)
        strong_defense_mode = (corridor is not None and race_score < 0.0)

        # Ponts virtuels "normaux" (entiers)
        _, player_cut_points = self._find_virtual_bridges(state, player_piece)
        _, opp_cut_points = self._find_virtual_bridges(state, opp_piece)

        # Ponts virtuels cassés : cases où jouer répare un pont attaqué
        broken_rescue_points = self._find_broken_virtual_bridges(state, player_piece)

        tt_pos = tt_entry.best_action.data["position"] if (tt_entry and tt_entry.best_action) else None

        scored: List[Tuple[float, LightAction]] = []

        for action in actions:
            i, j = action.data["position"]
            pos = (i, j)
            score = 0.0

            # TT move prioritaire
            if tt_pos is not None and pos == tt_pos:
                score += 10_000.0

            # Proximité du centre
            center_dist = abs(i - center_i) + abs(j - center_j)
            score -= center_dist

            # Analyse locale des voisins
            friend_adj = 0
            enemy_adj = 0
            for _, (ni, nj) in rep.get_neighbours(i, j).values():
                occ = env.get((ni, nj))
                if occ is None:
                    continue
                if occ.piece_type == player_piece:
                    friend_adj += 1
                elif occ.piece_type == opp_piece:
                    enemy_adj += 1

            # Pions adjacents : attaque & présence
            if friend_adj > 0:
                score += 2.0 * friend_adj
            # Adjacent à des pierres adverses : potentiel de coupe / défense locale
            if enemy_adj > 0:
                score += 1.8 * enemy_adj

            # Potentiel de ponts (pairs de voisins)
            friend_pairs = friend_adj * (friend_adj - 1) / 2.0
            enemy_pairs = enemy_adj * (enemy_adj - 1) / 2.0

            # Compléter nos ponts (un pion sur deux)
            if friend_pairs > 0:
                score += 2.2 * friend_pairs

            # Couper les ponts adverses
            if enemy_pairs > 0:
                score += 3.0 * enemy_pairs

            on_opp_min_path = (pos in opp_path_empties)

            # 1) Coups qui coupent directement le chemin minimal adverse
            if on_opp_min_path:
                if race_losing_for_player:
                    # VRAIMENT prioritaire (cas de ta partie)
                    score += self.W_CUT_PATH_STRONG
                elif race_warning:
                    score += self.W_CUT_PATH_WEAK

            # 2) Ponts virtuels : couper ceux de l'adversaire
            if pos in opp_cut_points:
                score += 8.0 * opp_cut_points[pos]

            # 3) Ponts virtuels : consolider / compléter les nôtres
            if pos in player_cut_points:
                score += 4.0 * player_cut_points[pos]

            # 4) Ponts virtuels cassés : jouer ici répare un pont attaqué
            if pos in broken_rescue_points:
                score += 30.0 * broken_rescue_points[pos]

            # --------------------------------------------------------------
            # V10.1 : gestion des lignes droites / courses perdantes
            # --------------------------------------------------------------
            in_corridor = corridor is not None and pos in corridor

            # Prolonger *notre* ligne dans un corridor où on perd la course →
            # malus (on évite les coups comme ton (7,3) ou (5,3) dans la partie).
            is_line_extension = (
                in_corridor
                and friend_adj >= 2
                and (opp_path is None or pos not in opp_path)
            )
            if race_losing_for_player and is_line_extension:
                score -= self.W_PENALIZE_STRAIGHT

            # Au contraire : si on perd la course, on valorise fortement
            # les coups dans le corridor qui touchent Bleu.
            if race_losing_for_player and in_corridor and enemy_adj > 0:
                score += self.W_CUT_CORRIDOR * (1 + min(enemy_adj, 2) / 2.0)

            # Et si, en plus, c'est une case du chemin minimal (cas déjà couvert
            # par on_opp_min_path), elle prendra naturellement la tête.

            # ------------------------------------------------------------------
            # V10 : gestion fine du corridor / ligne droite adverse
            # ------------------------------------------------------------------
            in_corridor = corridor is not None and pos in corridor

            # Combien de cases du chemin critique adverse sont voisines de ce coup ?
            cut_neighbors = 0
            if opp_path is not None:
                for _, (ni, nj) in rep.get_neighbours(i, j).values():
                    if (ni, nj) in opp_path:
                        cut_neighbors += 1

            # 1) Bonus pour les coupes **directes** sur le chemin critique
            if opp_path is not None and pos in opp_path:
                # Base : on coupe sa route
                base = 12.0
                # Si la course locale est déjà défavorable, on sur-renforce ce coup
                if strong_defense_mode:
                    base += 18.0
                # On valorise aussi la solidité locale de la coupe
                solidity = friend_adj + 0.5 * enemy_adj
                score += base + 2.0 * solidity

            # 2) Bonus pour les coups qui touchent plusieurs cases du chemin critique
            if in_corridor and cut_neighbors > 0:
                # on aime particulièrement les coups qui "contrôlent" plusieurs hex du chemin
                score += 6.0 * cut_neighbors

            # 3) Pénalisation des extensions de notre propre ligne perdante
            #    (quand on est en mode défense : course locale défavorable)
            is_line_extension = (
                in_corridor
                and friend_adj >= 2       # on prolonge surtout notre chaîne
                and enemy_adj == 0        # mais sans vraiment le couper
                and (opp_path is None or pos not in opp_path)
            )
            if strong_defense_mode and is_line_extension:
                # On décourage fortement les "lignes parallèles" inutiles
                score -= 25.0

            # 4) Si on est clairement perdant sur la course locale, on booste encore
            #    les coups directement sur le chemin adverse.
            if race_losing_for_player and opp_path is not None and pos in opp_path:
                score += 15.0

            # Petit bonus si avance dans l’axe de connexion du joueur au trait
            if player_piece == "R":
                axis_progress = i / (dim_i - 1) if dim_i > 1 else 0.0
            else:
                axis_progress = j / (dim_j - 1) if dim_j > 1 else 0.0
            score += 0.5 * axis_progress

            scored.append((score, action))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [a for _, a in scored]

    def _limit_branching(
        self,
        actions: List[LightAction],
        depth: int,
    ) -> List[LightAction]:
        """
        Réduit le nombre de coups explorés à cette profondeur.

        - Les actions doivent être déjà triées (meilleurs d'abord).
        - depth = profondeur restante dans la recherche.
        """
        n = len(actions)
        if n <= self.MIN_ACTIONS_NO_CUT:
            return actions

        # Clamp de la profondeur dans la table
        d = max(0, min(depth, len(self.BRANCHING_LIMITS) - 1))
        max_children = self.BRANCHING_LIMITS[d]

        if max_children >= n:
            return actions
        return actions[:max_children]

    # ------------------------------------------------------------------
    # Time & depth
    # ------------------------------------------------------------------

    def _check_time(self, deadline: float) -> None:
        if time.time() >= deadline:
            raise SearchTimeout()

    def _adaptive_depth_cap(self, state: GameStateHex, time_budget: float, branching: int) -> int:
        """
        Profondeur max approximative en fonction du temps et du branching.
        On vise un peu plus profond que V4/V5 grâce à une value plus légère.
        """
        dim_i, dim_j = state.rep.dimensions
        empties = dim_i * dim_j - len(state.rep.env)
        slack = time_budget

        depth = self.BASE_MAX_DEPTH

        # Plus de temps & peu de coups -> plus de profondeur
        if slack > 2.0 and branching < 60:
            depth += 1
        if slack > 3.5 and branching < 40:
            depth += 1
        if slack > 5.0 and branching < 25:
            depth += 1

        # Début de partie (énormément d'empties) -> réduire un peu
        if empties > (dim_i * dim_j * 0.6):
            depth -= 1

        return max(3, min(depth, self.MAX_DEPTH_CAP))

# ============================================================================
# MyPlayer : wrapper autour du moteur
# ============================================================================

_GLOBAL_GAME_COUNTER = 0


class MyPlayer(PlayerHex):
    def __init__(self, piece_type: str, name: str = "MyPlayerHexV6"):
        super().__init__(piece_type, name)
        self._engine = HexEngineV9(root_piece=piece_type)

        global _GLOBAL_GAME_COUNTER
        _GLOBAL_GAME_COUNTER += 1
        self._debug_game_id = _GLOBAL_GAME_COUNTER

        # --- Time manager (global budget) ---
        # Budget total par partie (15 minutes)
        self._total_budget_seconds: float = 15.0 * 60.0
        # Limite dure par coup (peut être imposée par le framework)
        self._per_move_cap: float = 8.0  # autoriser plus de temps par coup si le framework le tolère
        # Fraction max du temps restant utilisable pour ce coup (plus conservateur mais plus lent)
        self._time_fraction: float = 0.9
        # Horodatage de début de partie (côté de ce joueur)
        self._game_start_ts: float = time.time()

        if DEBUG_MODE:
            debug_logger.log(
                f"EVENT=new_player game_id={self._debug_game_id} piece={piece_type}"
            )

    def compute_action(self, current_state: GameState, remaining_time: int = 1e9, **kwargs) -> Action:
        if not isinstance(current_state, GameStateHex):
            raise ValueError("MyPlayer only supports GameStateHex.")

        possible_light_actions = list(current_state.get_possible_light_actions())
        if not possible_light_actions:
            raise MethodNotImplementedError("No possible actions available.")

        move_index = len(current_state.rep.env)  # 0-based: nombre de coups déjà joués

        # 1. Coup gagnant immédiat
        winning_move = self._engine.find_winning_action(current_state, self.piece_type)
        if winning_move is not None:
            if DEBUG_MODE:
                pos = winning_move.data["position"]
                debug_logger.log(
                    "EVENT=immediate_win "
                    f"game_id={self._debug_game_id} move={move_index} "
                    f"piece={self.piece_type} pos={pos}"
                )
            return winning_move

        # 2. Bloquer un coup gagnant adverse
        opp_piece = "B" if self.piece_type == "R" else "R"
        opponent_win = self._engine.find_winning_action(current_state, opp_piece)
        if opponent_win is not None:
            target_pos = opponent_win.data["position"]
            for la in possible_light_actions:
                if la.data["position"] == target_pos:
                    if DEBUG_MODE:
                        debug_logger.log(
                            "EVENT=block_opponent_win "
                            f"game_id={self._debug_game_id} move={move_index} "
                            f"piece={self.piece_type} pos={target_pos}"
                        )
                    return la

        # 3. Heuristique d’ouverture
        opening_hint = self._engine.suggest_opening_action(current_state, possible_light_actions)
        if opening_hint is not None:
            if DEBUG_MODE:
                debug_logger.log(
                    "EVENT=opening_hint "
                    f"game_id={self._debug_game_id} move={move_index} "
                    f"piece={self.piece_type} pos={opening_hint.data['position']}"
                )
            return opening_hint

        # 4. Allocation du temps pour la recherche (Time Manager global)
        dim_i, dim_j = current_state.rep.dimensions
        total_cells = dim_i * dim_j
        empties = total_cells - len(current_state.rep.env)
        # moves_left_est: approx half of empties, with a minimum to avoid huge ratios
        moves_left_est = max(8, empties // 2)

        # Temps global restant: préférer remaining_time si fourni par le framework
        if isinstance(remaining_time, (int, float)) and remaining_time < 1e8:
            time_remaining_global = max(0.0, float(remaining_time))
        else:
            time_elapsed = time.time() - self._game_start_ts
            time_remaining_global = max(0.0, self._total_budget_seconds - time_elapsed)

        # Base per move
        base = time_remaining_global / max(1, moves_left_est)

        # Phase-based multiplier (more greedy early to raise the curve)
        if empties > 140:
            phase_mult = 1.0   # opening: spend more to grow depth
        elif empties > 80:
            phase_mult = 1.2   # midgame: push further
        elif empties > 30:
            phase_mult = 1.4   # late: strong allocation
        else:
            phase_mult = 1.9   # tight endgame: maximal allocation

        # Bonus for tactical crunch
        d_self = self._engine._shortest_connection_distance(current_state, self.piece_type)
        d_opp = self._engine._shortest_connection_distance(current_state, "B" if self.piece_type == "R" else "R")
        if d_self <= 3 or d_opp <= 3:
            phase_mult *= 1.4

        # Final per-move time budget with clamp (0.6s to 6.0s), then respect per-move cap
        time_budget = base * phase_mult
        time_budget = max(0.6, min(time_budget, 6.0))
        time_budget = min(time_budget, self._per_move_cap)

        if DEBUG_MODE:
            debug_logger.log(
                "EVENT=time_budget "
                f"game_id={self._debug_game_id} move={move_index} "
                f"global_remaining={time_remaining_global:.1f} moves_left_est={moves_left_est} "
                f"budget={time_budget:.2f} empties={empties} d_self={d_self:.1f} d_opp={d_opp:.1f}"
            )

        # Infos debug pour le moteur
        self._engine.debug_game_id = self._debug_game_id
        self._engine.debug_move_index = move_index

        try:
            best_light_action = self._engine.choose_action(current_state, time_budget=time_budget)
        except SearchTimeout:
            # En cas de gros timeout au mauvais moment, on joue un coup au hasard
            fallback = random.choice(possible_light_actions)
            if DEBUG_MODE:
                debug_logger.log(
                    "EVENT=timeout_fallback "
                    f"game_id={self._debug_game_id} move={move_index} "
                    f"piece={self.piece_type} pos={fallback.data['position']}"
                )
            return fallback

        if DEBUG_MODE and self._engine.last_best_move is not None:
            pos = self._engine.last_best_move.data["position"]
            debug_logger.log(
                "EVENT=search_choice "
                f"game_id={self._debug_game_id} move={move_index} "
                f"piece={self.piece_type} pos={pos} "
                f"depth={self._engine.last_completed_depth} "
                f"val={self._engine.last_root_value:.2f} "
                f"nodes={self._engine.nodes_searched} "
                f"tt_hits={self._engine.tt_hits} "
                f"max_ply={self._engine.max_ply} "
                f"elapsed={self._engine.last_elapsed:.3f}"
            )

        return best_light_action  # type: ignore