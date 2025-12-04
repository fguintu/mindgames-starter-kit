"""
ExpectimaxMafiaAgent: Advanced Mafia Agent with LLM + Expectimax Decision-Making

Implements the approach from CS 557 Final Project Proposal:
- LLM-powered natural language generation and reasoning
- Identity-detection reinforcement learning (IDRL) concepts via belief state tracking
- Suspicion scoring based on behavioral heuristics
- Expectimax-style action selection for multi-agent reasoning
- Strategic deception and tactical lying
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import Agent, STANDARD_GAME_PROMPT
from envs.SecretMafia.env import Phase
import re
import random
from typing import Dict, List, Optional, Set
from collections import defaultdict


class ExpectimaxMafiaAgent(Agent):
    """
    Advanced Mafia agent combining LLM reasoning with expectimax decision-making,
    belief state tracking, suspicion scoring, and strategic deception.
    """
    
    # Suspicion heuristic weights
    CONTRADICTION_PENALTY = 0.15
    OVER_ACCUSATION_PENALTY = 0.08
    VOTE_SWITCH_PENALTY = 0.06
    DEFEND_MAFIA_PENALTY = 0.25
    
    # Utility weights for expectimax
    DOCTOR_KILL_UTILITY = 2.0
    DETECTIVE_KILL_UTILITY = 2.5
    VILLAGER_KILL_UTILITY = 1.0
    KNOWN_MAFIA_VOTE_UTILITY = 3.0
    
    def __init__(self, 
                 model_name: str = "microsoft/Phi-3-mini-4k-instruct",
                 device: str = "auto",
                 quantize: bool = False,
                 max_new_tokens: int = 64,
                 exploration_rate: float = 0.12,
                 lie_probability: float = 0.35,
                 verbose: bool = False,
                 hf_kwargs: dict = None):
        """
        Initialize the ExpectimaxMafiaAgent with LLM.
        
        Args:
            model_name: HuggingFace model name (default: Phi-3-mini-4k-instruct)
            device: Device for inference ("auto", "cuda", "cpu")
            quantize: Whether to use 8-bit quantization (saves VRAM)
            max_new_tokens: Maximum tokens to generate
            exploration_rate: Probability of random action for unpredictability
            lie_probability: Base probability of deceptive statements as Mafia
            verbose: Whether to print debug information
            hf_kwargs: Additional kwargs for HuggingFace model loading
        """
        super().__init__()
        
        # Initialize LLM
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
            import torch
        except ImportError:
            raise ImportError("Install: pip install transformers torch accelerate")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if hf_kwargs is None:
            hf_kwargs = {}
        
        hf_kwargs.setdefault('torch_dtype', torch.float16)
        hf_kwargs.setdefault('trust_remote_code', True)
        
        if quantize:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, load_in_8bit=True, device_map=device, **hf_kwargs
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, device_map=device, **hf_kwargs
            )
        
        self.pipeline = pipeline(
            'text-generation',
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens
        )
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        
        # Expectimax parameters
        self.exploration_rate = exploration_rate
        self.lie_probability = lie_probability
        self.verbose = verbose
        
        # Core game state
        self.player_id: Optional[int] = None
        self.role: Optional[str] = None
        self.team: Optional[str] = None
        self.teammates: List[int] = []
        self.num_players: int = 6
        self.alive_players: Set[int] = set()
        self.current_phase: Optional[Phase] = None
        self.day_number: int = 1
        
        # Belief state: P(role | observations) for each player
        self.belief_state: Dict[int, Dict[str, float]] = {}
        
        # Suspicion scores (higher = more suspicious of being Mafia)
        self.suspicion_scores: Dict[int, float] = {}
        
        # Behavioral tracking
        self.player_statements: Dict[int, List[str]] = defaultdict(list)
        self.player_vote_history: Dict[int, List[int]] = defaultdict(list)
        self.player_accusations: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.player_defenses: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.statement_count: Dict[int, int] = defaultdict(int)
        self.last_vote: Dict[int, Optional[int]] = {}
        
        # Known information
        self.known_roles: Dict[int, str] = {}
        self.confirmed_village: Set[int] = set()
        self.confirmed_mafia: Set[int] = set()
        
        # Game history
        self.eliminated_players: List[int] = []
        self.statements_this_day: int = 0
        
        self.initialized = False
        self.turn_count = 0

    def _log(self, message: str):
        """Print debug message if verbose mode enabled."""
        if self.verbose:
            print(f"[Agent {self.player_id}|{self.role}] {message}")

    def _generate_llm_response(self, prompt: str) -> str:
        """Generate response using the LLM with proper chat formatting."""
        try:
            messages = [{"role": "user", "content": prompt}]
            
            if hasattr(self.tokenizer, 'apply_chat_template'):
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                formatted_prompt = prompt
            
            response = self.pipeline(
                formatted_prompt,
                num_return_sequences=1,
                return_full_text=False,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                max_new_tokens=self.max_new_tokens
            )
            result = response[0]['generated_text'].strip()
            result = result.split('\n')[0].strip()
            return result
        except Exception as e:
            self._log(f"LLM error: {e}")
            return ""

    def _build_strategic_prompt(self, observation: str, action_type: str) -> str:
        """Build a strategic prompt for the LLM."""
        alive_list = sorted(self.alive_players)
        
        # Suspicion summary
        suspect_info = []
        for pid in alive_list:
            if pid != self.player_id:
                if pid in self.confirmed_mafia:
                    suspect_info.append(f"Player {pid} is CONFIRMED MAFIA")
                elif pid in self.confirmed_village:
                    suspect_info.append(f"Player {pid} is confirmed innocent")
        
        suspect_str = ". ".join(suspect_info) if suspect_info else "No confirmed info yet"
        
        # Role context
        if self.role == "Mafia":
            role_context = f"You are secretly Mafia (teammates: {self.teammates}). Pretend to be a villager."
        elif self.role == "Detective":
            role_context = f"You are Detective. {suspect_str}."
        elif self.role == "Doctor":
            role_context = "You are Doctor. Help village but hide your role."
        else:
            role_context = "You are Villager. Find the Mafia."
        
        # Action-specific prompts
        if action_type == "discussion":
            full_prompt = f"""You are Player {self.player_id} in Mafia game. {role_context}

Write a single short sentence (max 10 words) to say to the group. Output ONLY the sentence, nothing else."""
        
        elif action_type == "vote":
            valid_targets = [p for p in alive_list if p != self.player_id]
            if self.role == "Mafia":
                valid_targets = [p for p in valid_targets if p not in self.teammates]
            recommended = self._expectimax_select(valid_targets, is_night=False) if valid_targets else 0
            full_prompt = f"""You must vote [{recommended}]. Output ONLY: [{recommended}]"""
        
        else:  # night
            full_prompt = f"""Choose a player number. Output ONLY the number in brackets, example: [3]"""
        
        return full_prompt

    def _initialize_from_observation(self, observation: str):
        """Parse initial game setup from first observation."""
        player_match = re.search(r"You are Player (\d+)", observation)
        if player_match:
            self.player_id = int(player_match.group(1))
        
        role_match = re.search(r"Your role: (\w+)", observation)
        if role_match:
            self.role = role_match.group(1)
        
        team_match = re.search(r"Team: (\w+)", observation)
        if team_match:
            self.team = team_match.group(1)
        
        teammates_match = re.search(r"Your teammates are: ([^.]+)\.", observation)
        if teammates_match:
            teammate_str = teammates_match.group(1)
            self.teammates = [int(x) for x in re.findall(r"Player (\d+)", teammate_str)]
            self.teammates = [t for t in self.teammates if t != self.player_id]
        
        players_match = re.search(r"Players: ([^\n]+)", observation)
        if players_match:
            player_count = len(re.findall(r"Player \d+", players_match.group(1)))
            if player_count > 0:
                self.num_players = player_count
        
        self.alive_players = set(range(self.num_players))
        self._initialize_belief_state()
        
        self._log(f"Initialized: {self.num_players} players, role={self.role}, teammates={self.teammates}")
        self.initialized = True

    def _initialize_belief_state(self):
        """Initialize belief state with prior probabilities."""
        num_mafia = max(1, round(self.num_players * 0.25))
        mafia_prior = num_mafia / self.num_players
        
        for pid in range(self.num_players):
            if pid == self.player_id:
                self.belief_state[pid] = {
                    "Mafia": 1.0 if self.role == "Mafia" else 0.0,
                    "Doctor": 1.0 if self.role == "Doctor" else 0.0,
                    "Detective": 1.0 if self.role == "Detective" else 0.0,
                    "Villager": 1.0 if self.role == "Villager" else 0.0
                }
                self.suspicion_scores[pid] = 0.0
            elif pid in self.teammates:
                self.belief_state[pid] = {"Mafia": 1.0, "Doctor": 0.0, "Detective": 0.0, "Villager": 0.0}
                self.suspicion_scores[pid] = 0.0
                self.confirmed_mafia.add(pid)
            else:
                self.belief_state[pid] = {
                    "Mafia": mafia_prior,
                    "Doctor": 1 / self.num_players,
                    "Detective": 1 / self.num_players,
                    "Villager": 1.0 - mafia_prior - 2 / self.num_players
                }
                self.suspicion_scores[pid] = 0.5

    def _update_game_state(self, observation: str):
        """Update game state from observation text."""
        self._detect_phase(observation)
        self._track_eliminations(observation)
        self._parse_statements(observation)
        self._handle_investigation_results(observation)
        
        if "Day breaks" in observation:
            self.day_number += 1
            self.statements_this_day = 0

    def _detect_phase(self, observation: str):
        """Detect current game phase from observation."""
        obs_lower = observation.lower()
        
        if "night has fallen" in obs_lower or "mafia, agree on a victim" in obs_lower:
            self.current_phase = Phase.NIGHT_MAFIA
        elif "choose one player to protect" in obs_lower:
            self.current_phase = Phase.NIGHT_DOCTOR
        elif "choose one player to investigate" in obs_lower:
            self.current_phase = Phase.NIGHT_DETECTIVE
        elif "voting phase" in obs_lower:
            self.current_phase = Phase.DAY_VOTING
        elif "day breaks" in obs_lower or "discuss" in obs_lower:
            self.current_phase = Phase.DAY_DISCUSSION

    def _track_eliminations(self, observation: str):
        """Track player eliminations."""
        patterns = [r"Player (\d+) was eliminated", r"Player (\d+) was killed", r"Player (\d+) has been eliminated"]
        
        for pattern in patterns:
            for pid_str in re.findall(pattern, observation):
                pid = int(pid_str)
                if pid in self.alive_players:
                    self.alive_players.discard(pid)
                    self.eliminated_players.append(pid)
                    self._log(f"Player {pid} eliminated")

    def _parse_statements(self, observation: str):
        """Parse player statements for behavioral analysis."""
        matches = re.findall(r"\[Player (\d+)\]:\s*(.+?)(?=\[Player \d+\]:|$)", observation, re.DOTALL)
        
        for pid_str, statement in matches:
            pid = int(pid_str)
            statement = statement.strip()
            if statement and pid != self.player_id:
                self.player_statements[pid].append(statement)
                self.statement_count[pid] += 1
                self._analyze_statement(pid, statement)

    def _analyze_statement(self, player_id: int, statement: str):
        """Analyze statement for suspicious behavior."""
        if player_id in self.teammates or player_id == self.player_id:
            return
        
        statement_lower = statement.lower()
        suspicion_delta = 0.0
        
        # Track accusations
        for acc in re.findall(r"(?:suspect|accuse|vote|suspicious)\s*(?:player\s*)?(\d+)", statement_lower):
            target = int(acc)
            if target < self.num_players:
                self.player_accusations[player_id][target] += 1
        
        # Track defenses
        for def_m in re.findall(r"(?:trust|innocent|defend|not mafia)\s*(?:player\s*)?(\d+)", statement_lower):
            target = int(def_m)
            if target < self.num_players:
                self.player_defenses[player_id][target] += 1
        
        # Over-accusation penalty
        if len(self.player_accusations[player_id]) > 2:
            suspicion_delta += self.OVER_ACCUSATION_PENALTY
        
        # Defending Mafia penalty
        for target in self.player_defenses[player_id]:
            if target in self.confirmed_mafia:
                suspicion_delta += self.DEFEND_MAFIA_PENALTY
        
        old_score = self.suspicion_scores.get(player_id, 0.5)
        self.suspicion_scores[player_id] = min(1.0, max(0.0, old_score + suspicion_delta))

    def _handle_investigation_results(self, observation: str):
        """Handle detective investigation results."""
        if self.role != "Detective":
            return
        
        mafia_match = re.search(r"Player (\d+) IS a Mafia member", observation)
        not_mafia_match = re.search(r"Player (\d+) IS NOT a Mafia member", observation)
        
        if mafia_match:
            target = int(mafia_match.group(1))
            self.confirmed_mafia.add(target)
            self.belief_state[target] = {"Mafia": 1.0, "Doctor": 0.0, "Detective": 0.0, "Villager": 0.0}
            self.suspicion_scores[target] = 1.0
            self._log(f"CONFIRMED: Player {target} is Mafia!")
        
        elif not_mafia_match:
            target = int(not_mafia_match.group(1))
            self.confirmed_village.add(target)
            self.belief_state[target]["Mafia"] = 0.0
            self.suspicion_scores[target] = max(0.0, self.suspicion_scores.get(target, 0.5) - 0.3)
            self._log(f"CONFIRMED: Player {target} is NOT Mafia")

    def _get_valid_targets(self, observation: str) -> List[int]:
        """Extract valid targets from observation."""
        targets = re.findall(r"\[(\d+)\]", observation)
        return [int(t) for t in targets if int(t) in self.alive_players and int(t) != self.player_id]

    def _expectimax_select(self, candidates: List[int], is_night: bool = False) -> int:
        """Expectimax-style action selection."""
        if not candidates:
            return -1
        
        best_action = candidates[0]
        best_utility = float('-inf')
        
        for target in candidates:
            utility = self._compute_utility(target, is_night)
            utility += random.gauss(0, 0.03)
            
            if utility > best_utility:
                best_utility = utility
                best_action = target
        
        self._log(f"Expectimax selected {best_action} with utility {best_utility:.3f}")
        return best_action

    def _compute_utility(self, target: int, is_night: bool) -> float:
        """Compute expected utility of targeting a player."""
        beliefs = self.belief_state.get(target, {})
        suspicion = self.suspicion_scores.get(target, 0.5)
        
        if self.role == "Mafia":
            if target in self.teammates:
                return -100.0
            if is_night:
                return (beliefs.get("Doctor", 0) * self.DOCTOR_KILL_UTILITY +
                        beliefs.get("Detective", 0) * self.DETECTIVE_KILL_UTILITY +
                        beliefs.get("Villager", 0) * self.VILLAGER_KILL_UTILITY)
            else:
                return suspicion * 0.8
        
        elif self.role == "Doctor" and is_night:
            return 1.0 - beliefs.get("Mafia", 0.33) + beliefs.get("Detective", 0) * 0.5
        
        elif self.role == "Detective":
            if is_night:
                if target in self.known_roles:
                    return -100.0
                return 1.0 - abs(suspicion - 0.55) * 2
            else:
                return self.KNOWN_MAFIA_VOTE_UTILITY if target in self.confirmed_mafia else suspicion
        
        else:  # Villager
            return self.KNOWN_MAFIA_VOTE_UTILITY if target in self.confirmed_mafia else suspicion * 1.2

    def _is_night_phase(self, observation: str) -> bool:
        """Check if current phase is night."""
        obs_lower = observation.lower()
        return any(x in obs_lower for x in [
            "night has fallen", "night phase - choose", "mafia, agree on a victim",
            "choose one player to protect:", "choose one player to investigate:"
        ])

    def _is_voting_phase(self, observation: str) -> bool:
        """Check if current phase is voting."""
        obs_lower = observation.lower()
        return "voting phase" in obs_lower and "submit" in obs_lower

    def _is_discussion_phase(self, observation: str) -> bool:
        """Check if current phase is day discussion."""
        if self._is_voting_phase(observation):
            return False
        obs_lower = observation.lower()
        return "day breaks" in obs_lower or "discuss" in obs_lower

    def __call__(self, observation: str) -> str:
        """Process observation and return action."""
        self.turn_count += 1
        
        if not self.initialized:
            self._initialize_from_observation(observation)
        
        self._update_game_state(observation)
        
        if self._is_night_phase(observation):
            self._log("Detected NIGHT phase")
            return self._night_action(observation)
        elif self._is_voting_phase(observation):
            self._log("Detected VOTING phase")
            return self._voting_action(observation)
        elif self._is_discussion_phase(observation):
            self._log("Detected DISCUSSION phase")
            return self._discussion_action(observation)
        else:
            self._log("No action phase detected, providing generic response")
            return self._discussion_action(observation)

    def _night_action(self, observation: str) -> str:
        """Handle night phase actions."""
        valid_targets = self._get_valid_targets(observation)
        
        if not valid_targets:
            excluded = {self.player_id} | set(self.teammates)
            valid_targets = [p for p in self.alive_players if p not in excluded]
        
        if not valid_targets:
            return "[0]"
        
        if random.random() < self.exploration_rate:
            target = random.choice(valid_targets)
            self._log(f"Night action (random): targeting {target}")
        else:
            target = self._expectimax_select(valid_targets, is_night=True)
            self._log(f"Night action (expectimax): targeting {target}")
        
        return f"[{target}]"

    def _voting_action(self, observation: str) -> str:
        """Handle day voting phase."""
        valid_targets = self._get_valid_targets(observation)
        
        if not valid_targets:
            valid_targets = [p for p in self.alive_players if p != self.player_id]
            if self.role == "Mafia":
                valid_targets = [p for p in valid_targets if p not in self.teammates]
        
        if not valid_targets:
            return "[0]"
        
        # Vote for confirmed Mafia immediately
        confirmed_alive = [p for p in self.confirmed_mafia if p in valid_targets]
        if confirmed_alive and self.role != "Mafia":
            target = confirmed_alive[0]
            self._log(f"Voting for confirmed Mafia: Player {target}")
            return f"[{target}]"
        
        target = self._expectimax_select(valid_targets, is_night=False)
        self._log(f"Voting for Player {target}")
        return f"[{target}]"

    def _discussion_action(self, observation: str) -> str:
        """Handle day discussion phase with LLM."""
        self.statements_this_day += 1
        
        prompt = self._build_strategic_prompt(observation, "discussion")
        response = self._generate_llm_response(prompt)
        self._log(f"LLM response: {response}")
        
        if len(response) > 100:
            response = response[:100]
        
        if not response or len(response) < 3:
            if self.role == "Detective" and self.confirmed_mafia:
                alive_mafia = [p for p in self.confirmed_mafia if p in self.alive_players]
                if alive_mafia:
                    response = f"Player {alive_mafia[0]} is Mafia. Vote them out."
                else:
                    response = "We need to find the remaining Mafia."
            elif self.role == "Mafia":
                suspects = [p for p in self.alive_players if p != self.player_id and p not in self.teammates]
                if suspects:
                    response = f"I think Player {suspects[0]} is suspicious."
                else:
                    response = "Let us think carefully about this."
            else:
                response = "We should look for suspicious behavior."
        
        return response

    def get_belief_summary(self) -> str:
        """Get summary of current belief state."""
        lines = ["=== Belief State ==="]
        for pid in sorted(self.alive_players):
            beliefs = self.belief_state.get(pid, {})
            suspicion = self.suspicion_scores.get(pid, 0.5)
            mafia_prob = beliefs.get("Mafia", 0)
            status = "CONFIRMED MAFIA" if pid in self.confirmed_mafia else \
                     "CONFIRMED VILLAGE" if pid in self.confirmed_village else ""
            lines.append(f"Player {pid}: P(Mafia)={mafia_prob:.2f}, Suspicion={suspicion:.2f} {status}")
        return "\n".join(lines)


def test_agent():
    """Test the agent."""
    print("Loading ExpectimaxMafiaAgent with LLM...")
    print("(First run may take a moment to download the model)\n")
    
    agent = ExpectimaxMafiaAgent(
        model_name="microsoft/Phi-3-mini-4k-instruct",
        verbose=True,
        quantize=False
    )
    
    print("=" * 60)
    print("TEST 1: First observation (Welcome + Night phase)")
    print("=" * 60)
    
    first_obs = """Welcome to Secret Mafia! You are Player 2.
Your role: Detective
Team: Village

Players: Player 0, Player 1, Player 2, Player 3, Player 4, Player 5

Night phase - choose one player to investigate: [0], [1], [3], [4], [5]"""
    
    response = agent(first_obs)
    print(f"Response: {response}")
    
    print("\n" + "=" * 60)
    print("TEST 2: Investigation result")
    print("=" * 60)
    
    result_obs = """Player 3 IS a Mafia member."""
    response = agent(result_obs)
    print(f"Response: {response}")
    print(f"Confirmed Mafia: {agent.confirmed_mafia}")
    
    print("\n" + "=" * 60)
    print("TEST 3: Day Discussion")
    print("=" * 60)
    
    day_obs = """Day breaks. Discuss for 3 rounds, then a vote will follow.
[Player 0]: I think we should be careful today."""
    
    response = agent(day_obs)
    print(f"Response: {response}")
    
    print("\n" + "=" * 60)
    print("TEST 4: Voting Phase (should vote [3])")
    print("=" * 60)
    
    vote_obs = """Voting phase - submit one vote in format [X]. Valid: [0], [1], [3], [4], [5]"""
    
    response = agent(vote_obs)
    print(f"Response: {response}")
    
    vote_match = re.search(r"\[(\d+)\]", response)
    if vote_match and vote_match.group(1) == '3':
        print("✓ Correctly voted for confirmed Mafia")
    else:
        print("✗ Did not vote for confirmed Mafia")
    
    print("\n" + "=" * 60)
    print("TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_agent()