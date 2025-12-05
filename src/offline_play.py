"""
This script allows you to test ExpectimaxMafiaAgent against other agents.
Uses the textarena SecretMafia-v0 environment.

Options:
    - Set USE_HUMAN = True to play manually against AI agents
    - Adjust NUM_PLAYERS (SecretMafia requires 6 players)
"""

import textarena as ta 
import random
import re
from agent import HumanAgent
from myagent import ExpectimaxMafiaAgent
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Verify API key is set
if not os.getenv("OPENROUTER_API_KEY"):
    raise ValueError("OPENROUTER_API_KEY not found. Create a .env file with: OPENROUTER_API_KEY=your_key_here")

# ===========================================
# CONFIGURATION
# ===========================================
USE_HUMAN = False  # Set True to play as human instead of ExpectimaxMafiaAgent
NUM_PLAYERS = 6    # SecretMafia requires 6 players
VERBOSE = True     # Print agent reasoning

# Your agent's player slot (0 to NUM_PLAYERS-1)
YOUR_PLAYER_ID = 0


# ===========================================
# SIMPLE RANDOM AGENT (fallback if ta.agents.RandomAgent unavailable)
# ===========================================
class SimpleRandomAgent:
    """Simple random agent for testing when ta.agents.RandomAgent is unavailable."""
    
    def __call__(self, observation: str) -> str:
        # Extract valid targets from observation
        targets = re.findall(r"\[(\d+)\]", observation)
        if targets:
            return f"[{random.choice(targets)}]"
        # For discussion phase
        return random.choice([
            "I'm not sure who to trust.",
            "We need to find the Mafia.",
            "Let's think carefully.",
            "Has anyone noticed anything suspicious?",
        ])


# ===========================================
# INITIALIZE AGENTS
# ===========================================

# Create your agent
if USE_HUMAN:
    your_agent = HumanAgent()
else:
    print("Loading ExpectimaxMafiaAgent...")
    print("(First run may take a moment to download the model)\n")
    your_agent = ExpectimaxMafiaAgent(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        verbose=VERBOSE,
        quantize=False,  # Set True if low on VRAM
        exploration_rate=0.1,
        lie_probability=0.4,
    )

# Create all agents - your agent + opponents
agents = {}
agents[YOUR_PLAYER_ID] = your_agent

# Fill remaining slots with opponent agents
for i in range(NUM_PLAYERS):
    if i not in agents:
        # Try different agent options
        try:
            # Option 1: Use OpenRouter API agents (requires API key in environment)
            agents[i] = ta.agents.OpenRouterAgent(model_name="google/gemini-2.0-flash-lite-001")
            
            # Option 2: Use textarena's RandomAgent if available
            # agents[i] = ta.agents.RandomAgent()
        except AttributeError:
            # Option 3: Use our simple random agent as fallback
            agents[i] = SimpleRandomAgent()

# ===========================================
# RUN GAME
# ===========================================

# Initialize the environment
env = ta.make(env_id="SecretMafia-v0")
env.reset(num_players=NUM_PLAYERS)

print("=" * 60)
print("SECRETMAFIA GAME START")
print(f"Your agent (ExpectimaxMafiaAgent) is Player {YOUR_PLAYER_ID}")
print("=" * 60)

# Main game loop
done = False
turn = 0

while not done:
    turn += 1
    player_id, observation = env.get_observation()
    
    if VERBOSE:
        print(f"\n--- Turn {turn} | Player {player_id} ---")
        # Show truncated observation
        obs_short = observation[:300] + "..." if len(observation) > 300 else observation
        print(f"Observation: {obs_short}")
    
    action = agents[player_id](observation)
    
    if VERBOSE:
        print(f"Action: {action}")
    
    done, step_info = env.step(action=action)

# Game finished
rewards, game_info = env.close()

print("\n" + "=" * 60)
print("GAME OVER")
print("=" * 60)
print(f"Rewards: {rewards}")
print(f"Game Info: {game_info}")

# Check if your agent won
your_reward = rewards.get(YOUR_PLAYER_ID, 0)
if your_reward > 0:
    print(f"\n🎉 YOUR AGENT (Player {YOUR_PLAYER_ID}) WON!")
else:
    print(f"\n💀 Your agent (Player {YOUR_PLAYER_ID}) lost.")