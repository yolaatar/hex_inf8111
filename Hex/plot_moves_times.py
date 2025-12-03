# scripts/plot_move_times.py
import re, sys, os
import matplotlib.pyplot as plt

# Args: log_path [--save-dir DIR] [--no-show]
log_path = sys.argv[1] if len(sys.argv) > 1 else "hex_debug.log"
save_dir = None
no_show = False
for i, arg in enumerate(sys.argv[2:]):
    if arg == "--no-show":
        no_show = True
    elif arg == "--save-dir" and (i+3) <= len(sys.argv):
        save_dir = sys.argv[i+3]
        os.makedirs(save_dir, exist_ok=True)

# Segment by game: start at EVENT=new_player, end at EVENT=immediate_win
pat_new = re.compile(r"EVENT=new_player.*game_id=(\d+).*piece=([RB]).*")
pat_end = re.compile(r"EVENT=immediate_win.*game_id=(\d+).*piece=([RB]).*")
# Capture optional agent file tag added by logger: file=<name>
pat_file = re.compile(r"file=([^\s]+)")
pat_choice = re.compile(r"EVENT=search_choice.*game_id=(\d+).*move=(\d+).*piece=([RB]).*elapsed=([0-9.]+)")

games = []
current = None

with open(log_path, "r") as f:
    for line in f:
        # Try to get agent file name from the line
        mfile = pat_file.search(line)
        file_tag = mfile.group(1) if mfile else None
        # Start of a game
        mnew = pat_new.search(line)
        if mnew:
            gid = int(mnew.group(1))
            starter_piece = mnew.group(2)
            if current and current['events']:
                games.append(current)
            current = {'id': gid, 'events': [], 'agents': {}}  # agents per side
            # Record starter's agent file if available
            if file_tag:
                current['agents'][starter_piece] = file_tag
            continue
        # Search choice events
        mchoice = pat_choice.search(line)
        if mchoice:
            gid = int(mchoice.group(1))
            move = int(mchoice.group(2))
            piece = mchoice.group(3)
            elapsed = float(mchoice.group(4))
            if current is None or current['id'] != gid:
                current = {'id': gid, 'events': [], 'agents': {}}
            current['events'].append((move, piece, elapsed))
            # Record agent per side from file tag if present
            if file_tag and piece not in current['agents']:
                current['agents'][piece] = file_tag
            continue
        # End of a game
        mend = pat_end.search(line)
        if mend:
            gid = int(mend.group(1))
            if current and current['id'] == gid:
                # capture winner piece if present in line
                mwpiece = mend.group(2) if mend and len(mend.groups()) >= 2 else None
                current['winner_piece'] = mwpiece
                games.append(current)
                current = None

# Append last if not closed
if current and current['events']:
    games.append(current)

if not games:
    print("Aucune partie détectée (EVENT=new_player / EVENT=immediate_win manquants).")
    sys.exit(0)

def plot_one(ax, g, title_suffix=""):
    moves_R, times_R = [], []
    moves_B, times_B = [], []
    for move, piece, elapsed in g['events']:
        if piece == 'R':
            moves_R.append(move); times_R.append(elapsed)
        else:
            moves_B.append(move); times_B.append(elapsed)
    # Determine labels using agent file names if available
    agent_R = g.get('agents', {}).get('R', 'Agent R')
    agent_B = g.get('agents', {}).get('B', 'Agent B')
    label_R = f"{agent_R} (R)"
    label_B = f"{agent_B} (B)"
    # Totaux par bot (minutes)
    total_R_min = (sum(times_R) / 60.0) if times_R else 0.0
    total_B_min = (sum(times_B) / 60.0) if times_B else 0.0

    if moves_R:
        ax.plot(moves_R, times_R, marker='o', linestyle='-', color='#d62728', label=label_R)
    if moves_B:
        ax.plot(moves_B, times_B, marker='s', linestyle='-', color='#1f77b4', label=label_B)
    if moves_R and moves_B:
        ax.scatter(moves_R, times_R, color='#d62728', s=24, alpha=0.7)
        ax.scatter(moves_B, times_B, color='#1f77b4', s=24, alpha=0.7)
    # Winner string using agent file + side if known
    winner_piece = g.get('winner_piece')
    if winner_piece == 'R':
        winner_str = f"Winner: {agent_R} (R)"
    elif winner_piece == 'B':
        winner_str = f"Winner: {agent_B} (B)"
    else:
        winner_str = "Winner: unknown"

    ax.set_title(
        f"Per-move decision time — Game {title_suffix} (id={g['id']})\n"
        f"Total: {agent_R} (R) {total_R_min:.1f} min | {agent_B} (B) {total_B_min:.1f} min — {winner_str}"
    )
    ax.set_xlabel('Move number')
    ax.set_ylabel('Elapsed (s)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

# If many games, prefer saving individual images for readability
if len(games) > 2:
    if save_dir is None:
        save_dir = os.path.join(os.getcwd(), "plots")
        os.makedirs(save_dir, exist_ok=True)
    print(f"Nombre de parties: {len(games)}. Sauvegarde des figures dans {save_dir}.")
    for idx, g in enumerate(games, start=1):
        fig, ax = plt.subplots(figsize=(10,4))
        plot_one(ax, g, title_suffix=str(idx))
        fig.tight_layout()
        out = os.path.join(save_dir, f"game_{idx}_id_{g['id']}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"→ {out}")
    if not no_show:
        print("Affichage désactivé pour éviter une fenêtre trop longue. Utilise --no-show pour tailler le bruit.")
else:
    # Plot combined subplots if 1-2 games
    n = len(games)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(10, 4*n), sharex=False)
    if n == 1:
        axes = [axes]
    for idx, g in enumerate(games, start=1):
        plot_one(axes[idx-1], g, title_suffix=str(idx))
    fig.tight_layout()
    if save_dir:
        out = os.path.join(save_dir, "games_combined.png")
        fig.savefig(out, dpi=150)
        print(f"→ {out}")
    if not no_show:
        plt.show()