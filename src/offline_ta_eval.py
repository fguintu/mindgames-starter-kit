"""
Offline evaluation script for ExpectimaxMafiaAgent.
Evaluates the agent against OpenRouter LLM opponents on SecretMafia-v0.
"""
import os
import re
import random
from collections import defaultdict
from dotenv import load_dotenv

import numpy as np
import pandas as pd
from tqdm import tqdm

import textarena as ta
from myagent import ExpectimaxMafiaAgent

# Load environment variables from .env file
load_dotenv()

# Verify API key is set
if not os.getenv("OPENROUTER_API_KEY"):
    raise ValueError("OPENROUTER_API_KEY not found. Create a .env file with: OPENROUTER_API_KEY=your_key_here")

# ===========================================
# CONFIGURATION
# ===========================================
NUM_EPISODES = 100
EVAL_ENV_IDS = [("SecretMafia-v0", 6)]  # (env-id, num_players)
FILE_NAME = "eval_summary.csv"
VERBOSE = False  # Set True to see game outcomes

# OpenRouter opponent model
OPPONENT_MODEL = "google/gemini-2.0-flash-lite-001"


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
print(f"Opponent: {OPPONENT_MODEL}\n")


def run_game(env_id: str, num_players: int, model) -> dict:
    """Play one episode and return per-episode stats for the *model* player."""
    env = ta.make(env_id)
    env.reset(num_players=num_players)
    
    # Randomly assign model to a player slot
    model_pid = np.random.randint(0, num_players)
    
    # Create OpenRouter opponents for other players
    opponents = {}
    for i in range(num_players):
        if i != model_pid:
            opponents[i] = ta.agents.OpenRouterAgent(model_name=OPPONENT_MODEL)
    
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


def output_results(env_id, stats, role_stats, games_completed):
    """Output current results to console and CSV."""
    if games_completed == 0:
        print("\nNo games completed. No results to show.")
        return
    
    # Build results from stats
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
    print(f"Opponent: {OPPONENT_MODEL}")
    print("=" * 60)
    print(df.to_markdown(index=False, floatfmt=".3f"))
    
    # Role breakdown
    print("\n" + "=" * 60)
    print("WIN RATE BY ROLE")
    print("=" * 60)
    for role, rstats in role_stats.items():
        if rstats["games"] > 0:
            win_rate = rstats["wins"] / rstats["games"]
            print(f"{role}: {rstats['wins']}/{rstats['games']} ({win_rate:.1%})")
    
    # Save to CSV
    os.makedirs("eval_results", exist_ok=True)
    df.to_csv(f"eval_results/{FILE_NAME}", index=False)
    print(f"\nSaved -> eval_results/{FILE_NAME}")


# ===========================================
# RUN EVALUATION
# ===========================================
role_stats = defaultdict(lambda: {"wins": 0, "losses": 0, "games": 0})
games_completed = 0
current_env_id = EVAL_ENV_IDS[0][0]  # Track current environment

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
        
        # Reset stats for each environment
        stats = dict(
            wins=0,
            losses=0,
            draws=0,
            total_reward_model=0.0,
            total_reward_opponent=0.0,
            total_invalid_moves=0,
            total_turns=0,
        )
        games_completed = 0
        
        inner_bar = tqdm(range(NUM_EPISODES), desc=f"Evaluating {env_id}", leave=False)
        for ep in inner_bar:
            try:
                outcome = run_game(env_id, num_players, model)
            except Exception as e:
                error_msg = str(e)
                if "402" in error_msg or "credits" in error_msg.lower():
                    print(f"\n\n⚠️  OpenRouter credits exhausted. Stopping evaluation early.")
                    raise KeyboardInterrupt  # Use this to trigger the finally block
                else:
                    print(f"\n  Game {ep} failed with error: {e}")
                    continue  # Skip this game and try the next one
            
            games_completed += 1
            
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
            inner_bar.set_postfix({
                "Win%": f"{stats['wins'] / games_completed:.1%}",
                "Loss%": f"{stats['losses'] / games_completed:.1%}",
                "Inv%": f"{stats['total_invalid_moves'] / games_completed:.1%}",
            })

except KeyboardInterrupt:
    print("\n\nEvaluation interrupted. Outputting partial results...")

finally:
    # Always output whatever results we have
    output_results(current_env_id, stats, role_stats, games_completed)