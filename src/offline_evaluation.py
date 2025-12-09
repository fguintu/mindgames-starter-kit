"""
Offline evaluation script for ExpectimaxMafiaAgent.
Evaluates the agent against random opponents on SecretMafia-v0.
"""
import os
import re
import random
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

import textarena as ta
from myagent import ExpectimaxMafiaAgent

# ===========================================
# CONFIGURATION
# ===========================================
NUM_EPISODES = 100
EVAL_ENV_IDS = [("SecretMafia-v0", 6)]  # (env-id, num_players)
FILE_NAME = "eval_summary.csv"
VERBOSE = False  # Set True to see game outcomes


# ===========================================
# SIMPLE RANDOM AGENT (opponent)
# ===========================================
class RandomAgent:
    """Simple random agent for evaluation."""
    
    def __call__(self, observation: str) -> str:
        targets = re.findall(r"\[(\d+)\]", observation)
        if targets:
            return f"[{random.choice(targets)}]"
        return random.choice([
            "I'm not sure who to trust.",
            "We need to find the Mafia.",
            "Let's think carefully.",
        ])


# ===========================================
# INITIALIZE MODEL
# ===========================================
print("Loading ExpectimaxMafiaAgent...")
model = ExpectimaxMafiaAgent(
    model_name="microsoft/Phi-3-mini-4k-instruct",
    verbose=VERBOSE,
    quantize=False,
)
print("Model loaded.\n")


def run_game(env_id: str, num_players: int, model, opponent_class) -> dict:
    """Play one episode and return per-episode stats for the *model* player."""
    env = ta.make(env_id)
    env.reset(num_players=num_players)
    
    # Randomly assign model to a player slot
    model_pid = np.random.randint(0, num_players)
    
    # Create fresh opponents for each game
    opponents = {i: opponent_class() for i in range(num_players) if i != model_pid}
    
    # Reset model state for new game
    model.reset()
    
    done = False
    turn_count = 0
    while not done:
        pid, obs = env.get_observation()
        turn_count += 1
        
        if pid == model_pid:
            action = model(obs)
        else:
            action = opponents[pid](obs)
        
        done, _ = env.step(action=action)
    
    rewards, game_info = env.close()
    
    # Get model's role for analysis
    model_role = game_info[model_pid].get("role", "Unknown")
    model_won = rewards[model_pid] > 0
    
    if VERBOSE:
        print(f"  Game finished: {model_role} -> {'WIN' if model_won else 'LOSS'}")
    
    return {
        "model_reward": rewards[model_pid],
        "opponent_reward": np.mean([rewards[i] for i in range(num_players) if i != model_pid]),
        "invalid_move": bool(game_info[model_pid]["invalid_move"]),
        "turn_count": turn_count,
        "model_role": model_role,
    }


# ===========================================
# RUN EVALUATION
# ===========================================
results = defaultdict(list)
role_stats = defaultdict(lambda: {"wins": 0, "losses": 0, "games": 0})

outer_bar = tqdm(EVAL_ENV_IDS, desc="Environments")
for env_id, num_players in outer_bar:
    
    stats = dict(
        wins=0,
        losses=0,
        draws=0,
        total_reward_model=0.0,
        total_reward_opponent=0.0,
        total_invalid_moves=0,
        total_turns=0,
    )
    
    inner_bar = tqdm(range(NUM_EPISODES), desc=f"Evaluating {env_id}", leave=False)
    for ep in inner_bar:
        outcome = run_game(env_id, num_players, model, RandomAgent)
        
        # W/L/D
        if outcome["model_reward"] > outcome["opponent_reward"]:
            stats["wins"] += 1
            role_stats[outcome["model_role"]]["wins"] += 1
        elif outcome["model_reward"] < outcome["opponent_reward"]:
            stats["losses"] += 1
            role_stats[outcome["model_role"]]["losses"] += 1
        else:
            stats["draws"] += 1
        
        role_stats[outcome["model_role"]]["games"] += 1
        
        # Accumulate metrics
        stats["total_reward_model"] += outcome["model_reward"]
        stats["total_reward_opponent"] += outcome["opponent_reward"]
        stats["total_invalid_moves"] += int(outcome["invalid_move"])
        stats["total_turns"] += outcome["turn_count"]
        
        # Live progress bar
        games_done = ep + 1
        inner_bar.set_postfix({
            "Win%": f"{stats['wins'] / games_done:.1%}",
            "Loss%": f"{stats['losses'] / games_done:.1%}",
            "Inv%": f"{stats['total_invalid_moves'] / games_done:.1%}",
        })
    
    # Write per-environment summary
    results["env_id"].append(env_id)
    results["win_rate"].append(stats["wins"] / NUM_EPISODES)
    results["loss_rate"].append(stats["losses"] / NUM_EPISODES)
    results["draw_rate"].append(stats["draws"] / NUM_EPISODES)
    results["invalid_rate"].append(stats["total_invalid_moves"] / NUM_EPISODES)
    results["avg_turns"].append(stats["total_turns"] / NUM_EPISODES)
    results["avg_model_reward"].append(stats["total_reward_model"] / NUM_EPISODES)
    results["avg_opponent_reward"].append(stats["total_reward_opponent"] / NUM_EPISODES)

# ===========================================
# RESULTS
# ===========================================
df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("EVALUATION SUMMARY")
print("=" * 60)
print(df.to_markdown(index=False, floatfmt=".3f"))

# Role breakdown
print("\n" + "=" * 60)

# Save summary to text file
os.makedirs("eval_results", exist_ok=True)
with open("eval_results/eval_summary.txt", "w") as f:
    f.write("EVALUATION SUMMARY\n")
    f.write("=" * 60 + "\n")
    f.write(df.to_markdown(index=False, floatfmt=".3f"))
    f.write("\n\n")
print("WIN RATE BY ROLE")
print("=" * 60)
for role, stats in role_stats.items():
    if stats["games"] > 0:
        win_rate = stats["wins"] / stats["games"]
        print(f"{role}: {stats['wins']}/{stats['games']} ({win_rate:.1%})")

# Save to CSV
os.makedirs("eval_results", exist_ok=True)
df.to_csv(f"random_eval_results/{FILE_NAME}", index=False)
print(f"\nSaved -> random_eval_results/{FILE_NAME}")