import argparse
import asyncio
import os
from os.path import basename, splitext, dirname
import sys

from board_hex import BoardHex
from player_hex import PlayerHex
from master_hex import MasterHex
from game_state_hex import GameStateHex

from seahorse.player.proxies import InteractivePlayerProxy, LocalPlayerProxy, RemotePlayerProxy
from seahorse.utils.gui_client import GUIClient
from seahorse.utils.recorders import StateRecorder
from seahorse.game.game_layout.board import Piece
from seahorse.utils.custom_exceptions import PlayerDuplicateError

from loguru import logger
from argparse import RawTextHelpFormatter

def play(player1, player2, log_level, port, address, gui, record, gui_path, start_with: int = 0) :

    time_limit = 60*15
    list_players = [player1, player2]
    init_scores = {player1.get_id(): 0, player2.get_id(): 0}
    dim = [14, 14]
    env = {}
    
    init_rep = BoardHex(env=env, dim=dim)
    next_player = player1 if start_with == 0 else player2
    initial_game_state = GameStateHex(
        scores=init_scores, next_player=next_player, players=list_players, rep=init_rep, step=0)
    try:
        master = MasterHex(
            name="Hex", initial_game_state=initial_game_state, players_iterator=list_players, log_level=log_level, port=port,
            hostname=address, time_limit=time_limit
        )
    except PlayerDuplicateError:
        return None

    listeners = [GUIClient(path=gui_path)]*gui
    if record :
        listeners.append(StateRecorder())

    master.record_game(listeners=listeners)

    # Collect results to return for aggregation
    result = {}
    # winner may be set by the master (list of Player)
    winners = getattr(master, "winner", None)
    if winners is None:
        result["winner_ids"] = []
    else:
        try:
            result["winner_ids"] = [w.get_id() for w in winners]
        except Exception:
            # single winner
            try:
                result["winner_ids"] = [winners.get_id()]
            except Exception:
                result["winner_ids"] = []

    result["scores"] = master.current_game_state.get_scores()
    result["steps"] = master.current_game_state.get_step()
    result["custom_stats"] = master.get_custom_stats() if hasattr(master, "get_custom_stats") else []
    return result


if __name__=="__main__":

    parser = argparse.ArgumentParser(
                        prog="main_hex.py",
                        description="Description of the different execution modes:",
                        epilog=r'''
  ___           _                    
 / __| ___ __ _| |_  ___ _ _ ___ ___ 
 \__ \/ -_) _` | ' \/ _ \ '_(_-</ -_)
 |___/\___\__,_|_||_\___/_| /__/\___|
                                     ''',
                        formatter_class=RawTextHelpFormatter)
    parser.add_argument("-t","--type",
                        required=True,
                        type=str, 
                        choices=["local", "host_game", "connect", "human_vs_computer", "human_vs_human"],
                        help="\nThe execution mode you want.\n" 
                             +" - local: Runs everything on you machine\n"
                             +" - host_game: Runs a single player on your machine and waits for an opponent to connect with the 'connect' node.\n\t      You must provide an external ip for the -a argument (use 'ipconfig').\n"
                             +" - connect: Runs a single player and connects to a distant game launched with the 'host' at the hostname specified with '-a'.\n"
                             +" - human_vs_computer: Launches a GUI locally for you to challenge your player.\n"
                             +" - human_vs_human: Launches a GUI locally for you to experiment the game's mechanics.\n"
                             +"\n"
                        )
    parser.add_argument("-a","--address",required=False, default="localhost",help="\nThe external ip of the machine that hosts the GameMaster.\n\n")
    parser.add_argument("-p","--port",required=False,type=int, default=16001, help="The port of the machine that hosts the GameMaster.\n\n")
    parser.add_argument("-g","--no-gui",action='store_false',default=True, help="Headless mode\n\n")
    parser.add_argument("-n","--trials",required=False,type=int, default=1, help="Number of games to play in sequence (local mode only).\n\n")
    parser.add_argument("-r","--record",action="store_true",default=False, help="Stores the succesive game states in a json file.\n\n")
    parser.add_argument("-l","--log",required=False,choices=["DEBUG","INFO"], default="INFO",help="\nSets the logging level.")
    parser.add_argument("players_list",nargs="*", help='The players')

    args=parser.parse_args()

    type = vars(args).get("type")
    address = vars(args).get("address")
    port = vars(args).get("port")
    gui = vars(args).get("no_gui")
    record = vars(args).get("record")
    log_level = vars(args).get("log")
    list_players = vars(args).get("players_list")

    gui_path = os.path.join(dirname(os.path.abspath(__file__)),'GUI','index.html')

    if type == "local" :
        trials = vars(args).get("trials") or 1
        if trials > 1 and gui:
            logger.info("Disabling GUI for batch runs (--trials > 1).")
            gui = False

        # import player modules once
        folder = dirname(list_players[0])
        sys.path.append(folder)
        player1_module = __import__(splitext(basename(list_players[0]))[0], fromlist=[None])
        folder = dirname(list_players[1])
        sys.path.append(folder)
        player2_module = __import__(splitext(basename(list_players[1]))[0], fromlist=[None])

        stats = {"player1_wins": 0, "player2_wins": 0, "draws": 0, "total_steps": 0, "scores": []}
        for i in range(trials):
            # Alternate sides every other game: player1 gets R on even i, B on odd i
            if i % 2 == 0:
                p1_piece, p2_piece = "R", "B"
            else:
                p1_piece, p2_piece = "B", "R"

            player1 = player1_module.MyPlayer(p1_piece, name=splitext(basename(list_players[0]))[0]+f"_1_{i}")
            player2 = player2_module.MyPlayer(p2_piece, name=splitext(basename(list_players[1]))[0]+f"_2_{i}")
            # alternate who starts: 0 -> player1 starts, 1 -> player2 starts
            start_with = i % 2
            starter_name = basename(list_players[start_with])
            print(f"\n--- Game {i+1}/{trials}: {starter_name} starts ({'player1' if start_with==0 else 'player2'}) ---")
            res = play(player1=player1, player2=player2, log_level=log_level, port=port, address=address, gui=gui, record=record, gui_path=gui_path, start_with=start_with)
            if res is None:
                continue
            stats["total_steps"] += res.get("steps", 0)
            stats["scores"].append(res.get("scores", {}))

            # determine winner relative to this game's players
            winner_ids = res.get("winner_ids", [])
            if len(winner_ids) == 0:
                stats["draws"] += 1
            else:
                # player1.get_id() pertains to this game's instance
                if player1.get_id() in winner_ids and player2.get_id() in winner_ids:
                    stats["draws"] += 1
                elif player1.get_id() in winner_ids:
                    stats["player1_wins"] += 1
                elif player2.get_id() in winner_ids:
                    stats["player2_wins"] += 1
                else:
                    # Unknown winner id: count as draw
                    stats["draws"] += 1

            # Print cumulative score after this game
            p1_label = basename(list_players[0])
            p2_label = basename(list_players[1])
            played = i+1
            print(f"Current score after {played} game(s): {p1_label} {stats['player1_wins']} - {stats['player2_wins']} {p2_label} (draws: {stats['draws']})")

        # Report aggregated results
        print("\n=== Aggregated results ===")
        print(f"Games played: {trials}")
        print(f"{basename(list_players[0])} wins: {stats['player1_wins']}")
        print(f"{basename(list_players[1])} wins: {stats['player2_wins']}")
        print(f"Draws: {stats['draws']}")
        avg_steps = stats['total_steps'] / trials if trials>0 else 0
        print(f"Average steps: {avg_steps}")
        sys.exit(0)
    elif type == "host_game" :
        folder = dirname(list_players[0])
        sys.path.append(folder)
        player1_class = __import__(splitext(basename(list_players[0]))[0], fromlist=[None])
        player1 = LocalPlayerProxy(player1_class.MyPlayer("R", name=splitext(basename(list_players[0]))[0]+"_local"),gs=GameStateHex)
        player2 = RemotePlayerProxy(mimics=PlayerHex,piece_type="B",name="_remote")
        if address=='localhost':
            logger.warning('Using `localhost` with `host_game` mode, if both players are on different machines')
            logger.warning('use ipconfig/ifconfig to get your external ip and specity the ip with -a')
        play(player1=player1, player2=player2, log_level=log_level, port=port, address=address, gui=0, record=record, gui_path=gui_path)
    elif type == "connect" :
        folder = dirname(list_players[0])
        sys.path.append(folder)
        player2_class = __import__(splitext(basename(list_players[0]))[0], fromlist=[None])
        player2 = LocalPlayerProxy(player2_class.MyPlayer("B", name="_remote"),gs=GameStateHex)
        if address=='localhost':
            logger.warning('Using `localhost` with `connect` mode, if both players are on different machines')
            logger.warning('use ipconfig/ifconfig to get your external ip and specity the ip with -a')
        asyncio.new_event_loop().run_until_complete(player2.listen(keep_alive=True,master_address=f"http://{address}:{port}"))
    elif type == "human_vs_computer" :
        folder = dirname(list_players[0])
        sys.path.append(folder)
        player1_class = __import__(splitext(basename(list_players[0]))[0], fromlist=[None])
        player1 = InteractivePlayerProxy(PlayerHex("R", name="bob"),gui_path=gui_path,gs=GameStateHex)
        player2 = LocalPlayerProxy(player1_class.MyPlayer("B", name=splitext(basename(list_players[0]))[0]),gs=GameStateHex)
        play(player1=player1, player2=player2, log_level=log_level, port=port, address=address, gui=False, record=record, gui_path=gui_path)
    elif type == "human_vs_human" :
        player1 = InteractivePlayerProxy(PlayerHex("R", name="bob"),gui_path=gui_path,gs=GameStateHex)
        player2 = InteractivePlayerProxy(PlayerHex("B", name="alice"))
        player2.share_sid(player1)
        play(player1=player1, player2=player2, log_level=log_level, port=port, address=address, gui=False, record=record, gui_path=gui_path)
        
