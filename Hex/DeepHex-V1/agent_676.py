from __future__ import annotations
import math
import time
import random
from dataclasses import dataclass
from typing import Dict, Tuple, List, Iterable, Optional, Set

from player_hex import PlayerHex
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from seahorse.game.light_action import LightAction
from game_state_hex import GameStateHex
from seahorse.utils.custom_exceptions import MethodNotImplementedError


# ================================
# MCTS (PUCT) + RAVE (AMAF) + Heuristique connectivité/bridges
# ================================

@dataclass
class _ChildEdge:
    action: LightAction
    child_key: Tuple
    prior: float = 0.0  # prior PUCT (>=0)


class _Node:
    __slots__ = (
        "key", "N", "W",
        "children", "untried_actions",
        "player_to_move",
        "rave"  # Dict[pos -> [AMAF_N, AMAF_W]]
    )
    def __init__(self, key, player_to_move, untried_actions: List[LightAction]):
        self.key = key
        self.N = 0
        self.W = 0.0
        self.children: Dict[Tuple, _ChildEdge] = {}
        self.untried_actions: List[LightAction] = list(untried_actions)
        self.player_to_move = player_to_move
        self.rave: Dict[Tuple[int, int], List[float]] = {}  # position -> [N, W]

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0


class MyPlayer(PlayerHex):

    _MAX_NODES = 200_000
    _MAX_VALUE_CACHE = 300_000
    _MAX_DIST_CACHE = 200_000

    def __init__(
        self,
        piece_type: str,
        name: str = "MCTS_RAVE_HeuristicPlus",
        *,
        c_puct: float = 1.6,
        base_time: float = 1.0,
        debug: bool = False,
        center_opening: bool = True,
        root_dirichlet: float = 0.0,
        c_puct_init: float = 1.25,
        c_puct_base: float = 19652.0,
        rave_beta: float = 3000.0
    ):
        super().__init__(piece_type, name)
        self._c_puct = float(c_puct)
        self._c_init = float(c_puct_init)
        self._c_base = float(c_puct_base)
        self._base_time = float(base_time)
        self._debug = debug
        self._center_opening = center_opening
        self._root_dirichlet = float(root_dirichlet)
        self._rave_beta = float(rave_beta)

        self._nodes: Dict[Tuple, _Node] = {}
        self._value_cache: Dict[Tuple, float] = {}
        self._dist_cache: Dict[Tuple, Tuple[int, int]] = {}

        # Zobrist hashing (déterministe)
        self._zobrist: Dict[Tuple[int, int, str], int] = {}
        self._zobrist_to_move = {
            "R": 0x9E3779B185EBCA87,
            "B": 0xC2B2AE3D27D4EB4F
        }
        random.seed(0xC0FFEE)
        self._init_zobrist(max_dim=30)  # ajuster si board > 30

    # -------------- Public API --------------

    def compute_action(self, current_state: GameState, remaining_time: int = 1e9, **kwargs) -> Action:
        state: GameStateHex = current_state
        board = state.get_rep()
        env = board.get_env()
        dim = board.get_dimensions()[0]
        ply = len(env)

        legal = list(state.generate_possible_light_actions())
        if not legal:
            raise MethodNotImplementedError("No legal actions available.")
        if len(legal) == 1:
            return legal[0]

        if self._center_opening and ply == 0 and self.piece_type == "R":
            center = (dim // 2, dim // 2)
            return LightAction({"piece": "R", "position": center})

        # Time management
        phase = self._phase(dim, ply)
        phase_boost = {"opening": 0.8, "mid": 1.0, "end": 1.25}[phase]
        budget_soft = self._base_time * phase_boost
        budget_rt = max(0.06, min(budget_soft, float(remaining_time) * 0.03))
        deadline = time.time() + budget_rt

        root_player = self.piece_type
        root_key = self._state_key(state)
        root_node = self._get_or_create_node(state, root_key)

        D_R, D_B = self._distance_pair(state)
        d_self = D_R if root_player == "R" else D_B
        d_opp  = D_B if root_player == "R" else D_R

        # Tactiques immédiates
        if d_self == 0:
            return self._best_prior_fallback(state, root_player)
        win_now = self._find_win_in_one(state, root_player)
        if win_now is not None:
            return win_now
        block = self._block_opponent_win_in_one(state, root_player)
        if block is not None:
            return block

        if self._root_dirichlet > 0 and root_node.untried_actions:
            self._inject_root_noise(state, root_node, root_player, alpha=0.3, eps=0.15)

        iters = 0
        while time.time() < deadline:
            self._search_iteration(state, root_key, root_player)
            iters += 1

        # Décision: enfant le plus visité
        best_action = None
        best_visits = -1
        for child_key, edge in root_node.children.items():
            cn = self._nodes.get(child_key)
            if cn and cn.N > best_visits:
                best_visits = cn.N
                best_action = edge.action

        if best_action is None:
            best_action = self._best_prior_fallback(state, root_player)

        if self._debug:
            try:
                print("[MCTS-RAVE DEBUG]", {
                    "ply": ply, "phase": phase, "iters": iters,
                    "d_self": d_self, "d_opp": d_opp,
                    "best": best_action.data.get("position")
                }, flush=True)
            except Exception:
                pass

        return best_action

    # -------------- MCTS + RAVE internals --------------

    def _search_iteration(self, root_state: GameStateHex, root_key: Tuple, root_player: str) -> float:
        path_nodes: List[_Node] = []
        path_edges_pos: List[Tuple[int, int]] = []  # positions jouées le long du chemin (AMAF)
        node = self._nodes[root_key]
        state = root_state
        key = root_key
        path_nodes.append(node)

        # Selection / Expansion
        while True:
            if node.untried_actions:
                action = self._select_expansion_action(state, node, root_player)
                pos = tuple(action.data["position"])
                next_state = state.apply_action(action)
                next_key = self._state_key(next_state)
                child_node = self._get_or_create_node(next_state, next_key)

                prior = max(0.0, self._move_prior(state, action, root_player))
                prior += 1e-6 * random.random()  # jitter léger
                node.children[next_key] = _ChildEdge(action=action, child_key=next_key, prior=prior)

                # AMAF: consigne l'action jouée
                path_edges_pos.append(pos)

                state, key, node = next_state, next_key, child_node
                path_nodes.append(node)
                break
            else:
                if not node.children:
                    break  # terminal
                child_key, action = self._puct_rave_best_child(node)
                pos = tuple(action.data["position"])
                path_edges_pos.append(pos)
                state = state.apply_action(action)
                key = child_key
                node = self._nodes[key]
                path_nodes.append(node)

        # Evaluation (pas de rollout complet -> value heuristique)
        value = self._evaluate_state(state, root_player)

        # Backprop Q
        for n in path_nodes:
            n.N += 1
            n.W += value

        # Backprop AMAF (par action) : pour chaque nœud ancêtre,
        # met à jour rave[pos] si ce coup est légal/visité depuis ce nœud.
        # On utilise l'information disponible: positions des untried + enfants.
        for idx, n in enumerate(path_nodes):
            # Build set des positions possibles depuis ce nœud (connues)
            legal_positions_from_node: Set[Tuple[int, int]] = set()
            for a in n.untried_actions:
                legal_positions_from_node.add(tuple(a.data["position"]))
            for _ck, e in n.children.items():
                legal_positions_from_node.add(tuple(e.action.data["position"]))

            for pos in path_edges_pos:
                if pos in legal_positions_from_node:
                    stats = n.rave.get(pos)
                    if stats is None:
                        n.rave[pos] = [1.0, value]
                    else:
                        stats[0] += 1.0
                        stats[1] += value

        return value

    def _puct_rave_best_child(self, node: _Node) -> Tuple[Tuple, LightAction]:
        best_key = None
        best_action = None
        best_score = -1e18
        N_parent = max(1, node.N)
        sqrt_Np = math.sqrt(N_parent)
        c_now = self._effective_cpuct(N_parent)

        for child_key, edge in node.children.items():
            child = self._nodes.get(child_key)
            N = 0 if child is None else child.N
            Q = 0.0 if (child is None or child.N == 0) else (child.W / child.N)

            # AMAF_Q pour cette action (depuis ce nœud)
            pos = tuple(edge.action.data["position"])
            rstats = node.rave.get(pos)
            if rstats and rstats[0] > 0:
                AMAF_Q = rstats[1] / rstats[0]
                rave_N = rstats[0]
            else:
                AMAF_Q = 0.0
                rave_N = 0.0

            # β dynamique (mélange Q et AMAF_Q)
            # Variante: dépendre de N_parent et du nombre AMAF pour l'action
            beta = math.sqrt(self._rave_beta / (3.0 * (N_parent + 1e-9) + self._rave_beta))
            if rave_N <= 0:
                beta = 0.0

            mixed = (1.0 - beta) * Q + beta * AMAF_Q
            P = max(0.0, edge.prior)
            U = c_now * P * (sqrt_Np / (1.0 + N))
            score = mixed + U
            if score > best_score:
                best_score = score
                best_key = child_key
                best_action = edge.action
        return best_key, best_action

    def _effective_cpuct(self, N_parent: int) -> float:
        # AlphaZero-like schedule
        return self._c_init + math.log((N_parent + self._c_base + 1.0) / self._c_base)

    # -------------- Heuristique d'évaluation (distances + connectivité + ponts) --------------

    def _evaluate_state(self, state: GameStateHex, root_player: str) -> float:
        key = self._state_key(state)
        v = self._value_cache.get(key)
        if v is not None:
            return v

        dim = state.get_rep().get_dimensions()[0]
        D_R, D_B = self._distance_pair(state)

        # Connexité (approx) + motif "pont"
        conn_R = self._component_connectivity(state, "R")
        conn_B = self._component_connectivity(state, "B")
        bridge_R = self._bridge_bonus(state, "R")
        bridge_B = self._bridge_bonus(state, "B")

        if D_R == 0 and D_B > 0:
            val_R = 1.0
        elif D_B == 0 and D_R > 0:
            val_R = -1.0
        elif D_R == 0 and D_B == 0:
            val_R = 0.0
        else:
            Z = float(2 * dim)
            alpha = 1.85
            beta_conn = 0.40
            beta_bridge = 0.30
            base = math.tanh(alpha * (D_B - D_R) / Z)
            base += beta_conn * (conn_R - conn_B) / max(1.0, dim)
            base += beta_bridge * (bridge_R - bridge_B)
            val_R = max(-1.0, min(1.0, base))

        val = val_R if root_player == "R" else -val_R

        if len(self._value_cache) > self._MAX_VALUE_CACHE:
            self._value_cache.clear()
        self._value_cache[key] = val
        return val

    def _component_connectivity(self, state: GameStateHex, color: str) -> float:
        rep = state.get_rep()
        env = rep.get_env()
        dim = rep.get_dimensions()[0]
        visited = set()
        count = 0
        neigh = [(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1)]

        # somme des tailles des composantes qui touchent au moins un bord
        for (i, j), p in env.items():
            if p.get_type() == color and (i == 0 or j == 0 or i == dim-1 or j == dim-1):
                if (i, j) in visited:
                    continue
                stack = [(i, j)]
                comp_size = 0
                touches_border = False
                while stack:
                    ci, cj = stack.pop()
                    if (ci, cj) in visited:
                        continue
                    visited.add((ci, cj))
                    comp_size += 1
                    if ci == 0 or cj == 0 or ci == dim-1 or cj == dim-1:
                        touches_border = True
                    for di, dj in neigh:
                        ni, nj = ci + di, cj + dj
                        if 0 <= ni < dim and 0 <= nj < dim:
                            p2 = env.get((ni, nj))
                            if p2 is not None and p2.get_type() == color and (ni, nj) not in visited:
                                stack.append((ni, nj))
                if touches_border:
                    count += comp_size
        return float(count)

    def _bridge_bonus(self, state: GameStateHex, color: str) -> float:
        rep = state.get_rep()
        env = rep.get_env()
        dim = rep.get_dimensions()[0]
        bonus = 0.0
        # motif simple: C - . - C sur diagonale (pont potentiel si on remplit le .)
        for (i, j), p in env.items():
            if p.get_type() != color:
                continue
            for di, dj in [(1, -1), (-1, 1)]:
                ni, nj = i + di, j + dj
                ni2, nj2 = i + 2*di, j + 2*dj
                if 0 <= ni < dim and 0 <= nj < dim and 0 <= ni2 < dim and 0 <= nj2 < dim:
                    mid = env.get((ni, nj))
                    end = env.get((ni2, nj2))
                    if mid is None and (end is not None and end.get_type() == color):
                        bonus += 1.0
        return bonus

    # -------------- Sélection / Expansion --------------

    def _select_expansion_action(self, state: GameStateHex, node: _Node, root_player: str) -> LightAction:
        actions = node.untried_actions
        if not actions:
            raise MethodNotImplementedError("No untried actions on expansion.")

        dim = state.get_rep().get_dimensions()[0]

        # progressive widening
        K = max(6, 2 + node.N // 18)

        # frontier filter
        frontier = self._frontier_actions(state, actions)
        pool = frontier if len(frontier) >= min(len(actions), 8) else actions

        scored = []
        for a in pool:
            p = self._move_prior(state, a, root_player)
            # tie-break centre
            p -= 0.0008 * self._center_distance(a, dim)
            p = max(-1e6, min(1e6, p))
            scored.append((p, a))
        scored.sort(key=lambda t: -t[0])
        top = scored[:max(1, min(K, len(scored)))]

        # sampling softmax (temp=1)
        mx = max(s for s, _ in top)
        exps = [math.exp(s - mx) for s, _ in top]
        Z = sum(exps)
        if Z <= 0 or not math.isfinite(Z):
            chosen = top[0][1]
        else:
            r = random.random() * Z
            acc = 0.0
            chosen = top[0][1]
            for (e, (_, a)) in zip(exps, top):
                acc += e
                if r <= acc:
                    chosen = a
                    break

        node.untried_actions.remove(chosen)
        return chosen

    def _move_prior(self, state: GameStateHex, action: LightAction, my_color: str) -> float:
        """Prior heuristique Δdistance + motifs + biais de front."""
        dR0, dB0 = self._zero_one_bfs_pair(state)
        d_self0 = dR0 if my_color == "R" else dB0
        d_opp0  = dB0 if my_color == "R" else dR0

        ns = state.apply_action(action)
        dR1, dB1 = self._zero_one_bfs_pair(ns)
        d_self1 = dR1 if my_color == "R" else dB1
        d_opp1  = dB1 if my_color == "R" else dR1

        delta_self = d_self0 - d_self1
        delta_opp  = d_opp1 - d_opp0

        tpl = self._template_bonus(state, action, my_color)
        fbias = self._front_bias(state, action)

        prior = 0.62 * delta_self + 0.88 * delta_opp + tpl + 0.25 * fbias
        return prior

    def _template_bonus(self, state: GameStateHex, action: LightAction, my_color: str) -> float:
        rep = state.get_rep()
        dim = rep.get_dimensions()[0]
        (i, j) = action.data["position"]
        opp = "B" if my_color == "R" else "R"
        env = rep.get_env()
        neigh = [(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1)]

        def cell_type(c):
            p = env.get(c)
            return None if p is None else p.get_type()

        ally_adj = 0
        opp_adj = 0
        for di, dj in neigh:
            ni, nj = i+di, j+dj
            if 0 <= ni < dim and 0 <= nj < dim:
                t = cell_type((ni, nj))
                if t == my_color:
                    ally_adj += 1
                elif t == opp:
                    opp_adj += 1

        bonus = 0.0
        if ally_adj >= 2:
            bonus += 0.36
        if opp_adj >= 2:
            bonus += 0.46
        return bonus

    def _front_bias(self, state: GameStateHex, action: LightAction) -> float:
        rep = state.get_rep()
        dim = rep.get_dimensions()[0]
        (i, j) = action.data["position"]
        env = rep.get_env()
        acc = 0
        for (pi, pj), _p in env.items():
            d = abs(pi - i) + abs(pj - j)
            if d <= 2:
                acc += 1
        return min(acc, 4) * 0.12

    # -------------- Root helpers (noise, fallback) --------------

    def _best_prior_fallback(self, state: GameStateHex, my_color: str) -> LightAction:
        actions = list(state.generate_possible_light_actions())
        if not actions:
            raise MethodNotImplementedError("No legal actions available.")
        dim = state.get_rep().get_dimensions()[0]
        actions.sort(key=lambda a: self._center_distance(a, dim))  # tie-break centre
        actions.sort(key=lambda a: -self._move_prior(state, a, my_color))
        return actions[0]

    def _inject_root_noise(self, state: GameStateHex, node: _Node, my_color: str, *, alpha: float, eps: float):
        cands: List[Tuple[LightAction, float]] = []
        for a in state.generate_possible_light_actions():
            cands.append((a, self._move_prior(state, a, my_color)))
        if not cands:
            return
        K = len(cands)
        noise = self._dirichlet(alpha, K)
        base = [max(0.0, p) for (_, p) in cands]
        s = sum(base) or 1.0
        base = [b / s for b in base]
        mixed = [(1 - eps) * b + eps * n for b, n in zip(base, noise)]

        # applique sur les enfants existants si présents
        pos2idx = {tuple(a.data["position"]): idx for idx, (a, _) in enumerate(cands)}
        for ck, edge in node.children.items():
            pos = tuple(edge.action.data["position"])
            idx = pos2idx.get(pos)
            if idx is not None:
                edge.prior = mixed[idx]

    # -------------- Tactiques (win/block) --------------

    def _find_win_in_one(self, state: GameStateHex, my_color: str) -> Optional[LightAction]:
        dR0, dB0 = self._zero_one_bfs_pair(state)
        target_before = dR0 if my_color == "R" else dB0
        if target_before <= 0:
            return None
        for a in state.generate_possible_light_actions():
            ns = state.apply_action(a)
            dR1, dB1 = self._zero_one_bfs_pair(ns)
            if (my_color == "R" and dR1 == 0) or (my_color == "B" and dB1 == 0):
                return a
        return None

    def _block_opponent_win_in_one(self, state: GameStateHex, my_color: str) -> Optional[LightAction]:
        dR0, dB0 = self._zero_one_bfs_pair(state)
        base_opp = dB0 if my_color == "R" else dR0

        best = None
        best_delta = 0
        best_delta_self = -1e9
        for a in state.generate_possible_light_actions():
            ns = state.apply_action(a)
            dR1, dB1 = self._zero_one_bfs_pair(ns)
            d_opp1 = dB1 if my_color == "R" else dR1
            d_self0 = dR0 if my_color == "R" else dB0
            d_self1 = dR1 if my_color == "R" else dB1
            dO = d_opp1 - base_opp
            dS = d_self0 - d_self1
            if dO > best_delta or (dO == best_delta and dS > best_delta_self):
                best = a
                best_delta = dO
                best_delta_self = dS

        if best is not None and best_delta > 0:
            return best
        return None

    # -------------- Distances 0-1 BFS + cache --------------

    def _distance_pair(self, state: GameStateHex) -> Tuple[int, int]:
        key = self._state_key(state)
        cached = self._dist_cache.get(key)
        if cached is not None:
            return cached
        D_R, D_B = self._zero_one_bfs_pair(state)
        if len(self._dist_cache) > self._MAX_DIST_CACHE:
            self._dist_cache.clear()
        self._dist_cache[key] = (D_R, D_B)
        return D_R, D_B

    def _zero_one_bfs_pair(self, state: GameStateHex) -> Tuple[int, int]:
        rep = state.get_rep()
        env = rep.get_env()
        dim = rep.get_dimensions()[0]
        INF = 10**9
        neigh = [(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1)]

        def cost_for(color: str, cell: Tuple[int, int]) -> int:
            p = env.get(cell)
            if p is None:
                return 1  # vide => pierre manquante potentielle
            t = p.get_type()
            if t == color:
                return 0  # déjà à moi
            return INF     # adversaire = bloquant

        def bfs_for(color: str) -> int:
            dist = [[INF] * dim for _ in range(dim)]
            from collections import deque
            dq = deque()
            if color == "R":
                for j in range(dim):
                    c = (0, j)
                    w = cost_for(color, c)
                    if w >= INF:
                        continue
                    dist[0][j] = w
                    (dq.appendleft if w == 0 else dq.append)((0, j))
                def goal(i, j): return i == dim - 1
            else:
                for i in range(dim):
                    c = (i, 0)
                    w = cost_for(color, c)
                    if w >= INF:
                        continue
                    dist[i][0] = w
                    (dq.appendleft if w == 0 else dq.append)((i, 0))
                def goal(i, j): return j == dim - 1

            while dq:
                i, j = dq.popleft()
                if goal(i, j):
                    return dist[i][j]
                d0 = dist[i][j]
                for di, dj in neigh:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < dim and 0 <= nj < dim:
                        w = cost_for(color, (ni, nj))
                        if w >= INF:
                            continue
                        nd = d0 + w
                        if nd < dist[ni][nj]:
                            dist[ni][nj] = nd
                            (dq.appendleft if w == 0 else dq.append)((ni, nj))
            return INF

        return bfs_for("R"), bfs_for("B")

    # -------------- Nodes / Keys / Frontier --------------

    def _get_or_create_node(self, state: GameStateHex, key: Tuple) -> _Node:
        node = self._nodes.get(key)
        if node is not None:
            return node

        untried = list(state.generate_possible_light_actions())
        dim = state.get_rep().get_dimensions()[0]
        untried.sort(key=lambda a: (self._center_distance(a, dim),))
        env = state.get_rep().get_env()
        player_to_move = "R" if (len(env) % 2 == 0) else "B"

        node = _Node(key, player_to_move, untried)
        if len(self._nodes) > self._MAX_NODES:
            # purge douce: vide entièrement (simple mais sûr)
            self._nodes.clear()
        self._nodes[key] = node
        return node

    def _init_zobrist(self, max_dim: int = 30):
        for i in range(max_dim):
            for j in range(max_dim):
                self._zobrist[(i, j, "R")] = random.getrandbits(64)
                self._zobrist[(i, j, "B")] = random.getrandbits(64)

    def _state_key(self, state: GameStateHex) -> Tuple:
        rep = state.get_rep()
        env = rep.get_env()
        dim = rep.get_dimensions()[0]
        h = 0
        for (i, j), p in env.items():
            t = p.get_type()
            if t == "R" or t == "B":
                # si dim > max_dim init, on fold modulo
                ii = i if (i, j, t) in self._zobrist else (i % 30)
                jj = j if (i, j, t) in self._zobrist else (j % 30)
                h ^= self._zobrist.get((ii, jj, t), 0)
        # joueur au trait
        to_move = "R" if (len(env) % 2 == 0) else "B"
        h ^= self._zobrist_to_move[to_move]
        # inclure dimension pour réduire collisions
        return (h, dim, 0 if to_move == "R" else 1)

    def _center_distance(self, action: LightAction, dim: int) -> float:
        i, j = action.data["position"]
        c = (dim - 1) / 2.0
        return abs(i - c) + abs(j - c)

    def _frontier_actions(self, state: GameStateHex, actions: Iterable[LightAction]) -> List[LightAction]:
        rep = state.get_rep()
        env = rep.get_env()
        if not env:
            return list(actions)
        stones = list(env.keys())
        res = []
        for a in actions:
            (i, j) = a.data["position"]
            near = False
            for (pi, pj) in stones:
                if abs(pi - i) + abs(pj - j) <= 2:
                    near = True
                    break
            if near:
                res.append(a)
        return res if res else list(actions)

    # -------------- Utilitaires divers --------------

    def _dirichlet(self, alpha: float, K: int) -> List[float]:
        xs = [random.gammavariate(alpha, 1.0) for _ in range(K)]
        s = sum(xs) or 1.0
        return [x / s for x in xs]

    def _phase(self, dim: int, ply: int) -> str:
        total = dim * dim
        ratio = ply / max(1, total)
        if ratio < 0.22:
            return "opening"
        if ratio < 0.70:
            return "mid"
        return "end"