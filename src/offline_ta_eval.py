"""
Offline evaluation script for ExpectimaxMafiaAgent.
Evaluates the agent against local HuggingFace LLM opponents on SecretMafia-v0.
Optimized for Google Colab with GPU.

Requirements:
    pip install textarena transformers torch accelerate pandas tqdm bitsandbytes
"""
import os
import re
import random
import warnings
from collections import defaultdict

# Suppress noisy warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message=".*pad_token_id.*")
warnings.filterwarnings("ignore", message=".*Setting `pad_token_id`.*")

import numpy as np
import pandas as pd
from tqdm import tqdm

import textarena as ta
from myagent import ExpectimaxMafiaAgent

# ===========================================
# CONFIGURATION
# ===========================================
NUM_EPISODES = 100
EVAL_ENV_IDS = [("SecretMafia-v0", 6)]
FILE_NAME = "eval_summary.csv"
VERBOSE = True

# Model configuration
AGENT_MODEL = "microsoft/Phi-3-mini-4k-instruct"
OPPONENT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# ===========================================
# INITIALIZE MODELS
# ===========================================
print("=" * 60)
print("Loading models...")
print("=" * 60)

print(f"\n[1/2] Loading agent: {AGENT_MODEL}")
model = ExpectimaxMafiaAgent(
    model_name=AGENT_MODEL,
    verbose=VERBOSE,
    quantize=False,
)
print("Agent loaded.\n")

print(f"[2/2] Loading opponent: {OPPONENT_MODEL}")
print("    Using 8-bit quantization for faster inference...")
shared_opponent = ta.agents.HFLocalAgent(
    model_name=OPPONENT_MODEL,
    max_new_tokens=64,
    quantize=True,
    hf_kwargs={},  # Prevents NoneType error
)
print("Opponent loaded.\n")

print("=" * 60)
print(f"Agent: {AGENT_MODEL}")
print(f"Opponent: {OPPONENT_MODEL}")
print(f"Episodes: {NUM_EPISODES}")
print("=" * 60 + "\n")


def run_game(env_id: str, num_players: int, model) -> dict:
    """Play one episode and return per-episode stats for the *model* player."""
    env = ta.make(env_id)
    env.reset(num_players=num_players)
    
    # Randomly assign model to a player slot
    model_pid = np.random.randint(0, num_players)
    
    # Reuse shared opponent for all other players (stateless, so this is fine)
    opponents = {i: shared_opponent for i in range(num_players) if i != model_pid}
    
    # Reset model state for new game
    model.reset()
    
    done = False
    turn_count = 0
    while not done:
        pid, obs = env.get_observation()
        turn_count += 1
        
        if VERBOSE and turn_count % 10 == 0:
            print(f"    Turn {turn_count}...", end="\r")
        
        if pid == model_pid:
            action = model(obs)
        else:
            action = opponents[pid](obs)
        
        done, _ = env.step(action=action)
    
    rewards, game_info = env.close()
    
    model_role = game_info[model_pid].get("role", "Unknown")
    model_won = rewards[model_pid] > 0
    
    if VERBOSE:
        print(f"  Game finished: {model_role} -> {'WIN' if model_won else 'LOSS'} ({turn_count} turns)")
    
    return {
        "model_reward": rewards[model_pid],
        "opponent_reward": np.mean([rewards[i] for i in range(num_players) if i != model_pid]),
        "invalid_move": bool(game_info[model_pid]["invalid_move"]),
        "turn_count": turn_count,
        "model_role": model_role,
    }


def output_results(env_id, stats, role_stats, games_completed):
    """Output current results to console and CSV."""
    if games_completed == 0:
        print("\nNo games completed. No results to show.")
        return
    
    results = {
        "env_id": [env_id],
        "games": [games_completed],
        "win_rate": [stats["wins"] / games_completed],
        "loss_rate": [stats["losses"] / games_completed],
        "draw_rate": [stats["draws"] / games_completed],
        "invalid_rate": [stats["total_invalid_moves"] / games_completed],
        "avg_turns": [stats["total_turns"] / games_completed],
        "avg_model_reward": [stats["total_reward_model"] / games_completed],
        "avg_opponent_reward": [stats["total_reward_opponent"] / games_completed],
    }
    
    df = pd.DataFrame(results)
    
    print("\n" + "=" * 60)
    print(f"EVALUATION SUMMARY ({games_completed} games completed)")
    print(f"Agent: {AGENT_MODEL}")
    print(f"Opponent: {OPPONENT_MODEL}")
    print("=" * 60)
    print(df.to_markdown(index=False, floatfmt=".3f"))
    
    print("\n" + "=" * 60)
    print("WIN RATE BY ROLE")
    print("=" * 60)
    for role, rstats in role_stats.items():
        if rstats["games"] > 0:
            win_rate = rstats["wins"] / rstats["games"]
            print(f"{role}: {rstats['wins']}/{rstats['games']} ({win_rate:.1%})")
    
    os.makedirs("eval_results", exist_ok=True)
    df.to_csv(f"eval_results/{FILE_NAME}", index=False)
    print(f"\nSaved -> eval_results/{FILE_NAME}")


# ===========================================
# RUN EVALUATION
# ===========================================
role_stats = defaultdict(lambda: {"wins": 0, "losses": 0, "games": 0})
games_completed = 0
current_env_id = EVAL_ENV_IDS[0][0]

stats = dict(
    wins=0,
    losses=0,
    draws=0,
    total_reward_model=0.0,
    total_reward_opponent=0.0,
    total_invalid_moves=0,
    total_turns=0,
)

try:
    outer_bar = tqdm(EVAL_ENV_IDS, desc="Environments")
    for env_id, num_players in outer_bar:
        current_env_id = env_id
        
        stats = dict(
            wins=0, losses=0, draws=0,
            total_reward_model=0.0, total_reward_opponent=0.0,
            total_invalid_moves=0, total_turns=0,
        )
        games_completed = 0
        
        inner_bar = tqdm(range(NUM_EPISODES), desc=f"Evaluating {env_id}", leave=False)
        for ep in inner_bar:
            try:
                outcome = run_game(env_id, num_players, model)
            except Exception as e:
                print(f"\n  Game {ep} failed: {e}")
                continue
            
            games_completed += 1
            
            if outcome["model_reward"] > outcome["opponent_reward"]:
                stats["wins"] += 1
                role_stats[outcome["model_role"]]["wins"] += 1
            elif outcome["model_reward"] < outcome["opponent_reward"]:
                stats["losses"] += 1
                role_stats[outcome["model_role"]]["losses"] += 1
            else:
                stats["draws"] += 1
            
            role_stats[outcome["model_role"]]["games"] += 1
            
            stats["total_reward_model"] += outcome["model_reward"]
            stats["total_reward_opponent"] += outcome["opponent_reward"]
            stats["total_invalid_moves"] += int(outcome["invalid_move"])
            stats["total_turns"] += outcome["turn_count"]
            
            inner_bar.set_postfix({
                "Win%": f"{stats['wins'] / games_completed:.1%}",
                "Loss%": f"{stats['losses'] / games_completed:.1%}",
                "Inv%": f"{stats['total_invalid_moves'] / games_completed:.1%}",
            })

except KeyboardInterrupt:
    print("\n\nEvaluation interrupted. Outputting partial results...")

finally:
    output_results(current_env_id, stats, role_stats, games_completed)