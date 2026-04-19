import matplotlib.pyplot as plt
import numpy as np
import re
import os
import json
import sys
import argparse
from collections import defaultdict
from pathlib import Path

# Use the established MAGE_DIR path
MAGE_DIR = "/home/raven/Fendley/MagicAI/mage"
DEFAULT_WINRATE_FILE = "/home/raven/Fendley/MagicAI/MageZero/runs/2026-04-14_18-06-53/WinRates_gen1.txt"
OUTPUT_IMAGE = "winrate_matrix.png"

def parse_winrates(file_path):
    """
    Parses win rate data supporting old .txt, new .txt, and .json formats.
    
    Old format: WR with MonoRAggro vs BantRhythm: 0.9375 in 16 games
    New format: 2026-04-14 15:54:33 | WR: 50.00% (50/100) | deckA (mcts, budget 300) vs deckB (mcts, budget 300) | ...
    JSON format: {"timestamp": "...", "wins": 18, "games": 32, "player_a": {"deck": "MonoRAggro", ...}, "player_b": {"deck": "MonoGLandfall", ...}, ...}
    """
    
    # Regex for old format
    old_pattern = re.compile(r"WR with (.*) vs (.*): (.*) in (\d+) games")
    # Regex for new format (focusing on wins/games and deck names)
    new_pattern = re.compile(r".* \| WR: .*% \((\d+)/(\d+)\) \| (.*) \(.*\) vs (.*) \(.*\) \| .*")
    
    matchups = defaultdict(lambda: {"wins": 0, "games": 0})
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return {}

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Try JSON format
            if line.startswith('{'):
                try:
                    data = json.loads(line)
                    wins = data.get("wins")
                    games = data.get("games")
                    player_a = data.get("player_a", {})
                    player_b = data.get("player_b", {})
                    deckA = player_a.get("deck")
                    deckB = player_b.get("deck")
                    
                    if wins is not None and games is not None and deckA and deckB:
                        matchups[(deckA, deckB)]["wins"] += wins
                        matchups[(deckA, deckB)]["games"] += games
                        continue
                except json.JSONDecodeError:
                    pass

            # Try new text format
            new_match = new_pattern.match(line)
            if new_match:
                wins = int(new_match.group(1))
                games = int(new_match.group(2))
                deckA = new_match.group(3).strip()
                deckB = new_match.group(4).strip()
                matchups[(deckA, deckB)]["wins"] += wins
                matchups[(deckA, deckB)]["games"] += games
                continue
                
            # Try old text format
            old_match = old_pattern.match(line)
            if old_match:
                deckA = old_match.group(1).strip()
                deckB = old_match.group(2).strip()
                wr_str = old_match.group(3).strip()
                games_str = old_match.group(4)
                
                if wr_str.lower() == "nan" or not games_str:
                    continue
                    
                wr = float(wr_str)
                games = int(games_str)
                wins = int(round(wr * games))
                matchups[(deckA, deckB)]["wins"] += wins
                matchups[(deckA, deckB)]["games"] += games
                continue

    return matchups

def plot_heatmap(matchups, output_dir):
    """Generates a heatmap of win rates between different decks."""
    if not matchups:
        print("No matchups found to plot.")
        return

    # Extract unique deck names
    decks = sorted(set([d for pair in matchups.keys() for d in pair]))
    deck_to_idx = {deck: i for i, deck in enumerate(decks)}
    n = len(decks)
    
    # Initialize matrix with NaN
    matrix = np.full((n, n), np.nan)
    
    for (deckA, deckB), data in matchups.items():
        if data["games"] > 0:
            idxA = deck_to_idx[deckA]
            idxB = deck_to_idx[deckB]
            win_rate = data["wins"] / data["games"]
            matrix[idxA][idxB] = win_rate
            
            # If the reverse matchup isn't recorded, we can infer it
            if np.isnan(matrix[idxB][idxA]):
                 matrix[idxB][idxA] = 1.0 - win_rate

    # Create the plot
    fig, ax = plt.subplots(figsize=(max(8, n*0.8), max(6, n*0.6)))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(decks, rotation=45, ha="right")
    ax.set_yticklabels(decks)
    
    # Add text annotations for each cell
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}",
                        ha="center", va="center", 
                        color="black" if 0.3 < val < 0.7 else "white")

    ax.set_title(f"{Path(output_dir).name} Deck Win Rates Matrix (Row vs Column)")
    ax.set_xlabel("Opponent Deck")
    ax.set_ylabel("Primary Deck")
    
    plt.colorbar(im, label="Win Rate")
    fig.tight_layout()
    
    output_path = os.path.join(output_dir, OUTPUT_IMAGE)
    plt.savefig(output_path)
    print(f"Saved heatmap to {output_path}")

    # Also plot average winrate per deck
    plot_average_winrates(matchups, decks, output_dir)

def plot_average_winrates(matchups, decks, output_dir):
    """Plots a bar chart of the average win rate for each deck."""
    avg_wrs = []
    for deck in decks:
        total_wins = 0
        total_games = 0
        for (deckA, deckB), data in matchups.items():
            if deckA == deck:
                total_wins += data["wins"]
                total_games += data["games"]
            elif deckB == deck:
                total_wins += (data["games"] - data["wins"])
                total_games += data["games"]
        
        if total_games > 0:
            avg_wrs.append((deck, total_wins / total_games))
    
    if not avg_wrs:
        return

    avg_wrs.sort(key=lambda x: x[1], reverse=True)
    names, values = zip(*avg_wrs)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, values, color='skyblue')
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    plt.ylim(0, 1.0)
    plt.ylabel("Average Win Rate")
    plt.title(f"{Path(output_dir).name} Overall Deck Performance")
    plt.xticks(rotation=45, ha="right")
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    output_path = os.path.join(output_dir, "average_winrates.png")
    plt.savefig(output_path)
    print(f"Saved average win rates plot to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Graph deck win rates from result files.")
    parser.add_argument("input", nargs="?", default=DEFAULT_WINRATE_FILE, help="Path to the winrate data file.")
    parser.add_argument("--output-dir", "-o", default="./", help="Directory to save the generated images.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Created output directory: {args.output_dir}")

    data = parse_winrates(args.input)
    plot_heatmap(data, args.output_dir)

if __name__ == "__main__":
    main()

