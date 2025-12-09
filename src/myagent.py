"""
ExpectimaxMafiaAgent: Advanced Mafia Agent with LLM + Expectimax Decision-Making

Implements the approach from CS 557 Final Project Proposal:
- LLM-powered natural language generation and reasoning
- Identity-detection reinforcement learning (IDRL) concepts via belief state tracking
- Suspicion scoring based on behavioral heuristics
- Expectimax-style action selection for multi-agent reasoning
- Strategic deception and tactical lying

Key features:
- Detective delays reveal until vote is guaranteed
- Tracks investigated players to avoid wasted investigations
- Doctor protects vocal/claiming players (prioritizes Detective claimants)
- Vote pattern analysis increases suspicion for anti-consensus votes
- Villagers trust and follow Detective accusations
- LLM-based statement analysis and discussion generation
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import Agent
from envs.SecretMafia.env import Phase
import re, random
from typing import Dict, List, Optional, Set
from collections import defaultdict


class ExpectimaxMafiaAgent(Agent):
    
    CONTRADICTION_PENALTY = 0.15
    OVER_ACCUSATION_PENALTY = 0.08
    VOTE_AGAINST_CONSENSUS_PENALTY = 0.20
    DEFEND_MAFIA_PENALTY = 0.25
    
    DOCTOR_KILL_UTILITY = 2.0
    DETECTIVE_KILL_UTILITY = 2.5
    VILLAGER_KILL_UTILITY = 1.0
    KNOWN_MAFIA_VOTE_UTILITY = 3.0
    
    def __init__(self, model_name: str = "microsoft/Phi-3-mini-4k-instruct",
                 device: str = "auto", quantize: bool = False,
                 max_new_tokens: int = 64, exploration_rate: float = 0.12,
                 lie_probability: float = 0.35, verbose: bool = False, hf_kwargs: dict = None):
        super().__init__()
        
        from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
        import torch
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
        hf_kwargs = hf_kwargs or {}
        hf_kwargs.setdefault('torch_dtype', torch.float16)
        hf_kwargs.setdefault('trust_remote_code', False)
        
        if quantize:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map=device, **hf_kwargs)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, **hf_kwargs)
        
        self.pipeline = pipeline('text-generation', model=self.model, tokenizer=self.tokenizer, max_new_tokens=max_new_tokens)
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.exploration_rate = exploration_rate
        self.lie_probability = lie_probability
        self.verbose = verbose
        
        self.reset()

    def reset(self):
        self.player_id: Optional[int] = None
        self.role: Optional[str] = None
        self.team: Optional[str] = None
        self.teammates: List[int] = []
        self.num_players: int = 6
        self.alive_players: Set[int] = set()
        self.current_phase: Optional[Phase] = None
        self.day_number: int = 1
        
        self.belief_state: Dict[int, Dict[str, float]] = {}
        self.suspicion_scores: Dict[int, float] = {}
        
        self.player_statements: Dict[int, List[str]] = defaultdict(list)
        self.player_vote_history: Dict[int, List[int]] = defaultdict(list)
        self.player_accusations: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.player_defenses: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.statement_count: Dict[int, int] = defaultdict(int)
        
        self.confirmed_village: Set[int] = set()
        self.confirmed_mafia: Set[int] = set()
        self.investigated_players: Set[int] = set()
        
        self.claimed_detective: Optional[int] = None
        self.claimed_doctor: Optional[int] = None
        self.detective_accusations: Dict[int, int] = {}
        
        self.daily_votes: Dict[int, Dict[int, int]] = defaultdict(dict)
        self.vocal_players: Set[int] = set()
        
        self.eliminated_players: List[int] = []
        self.statements_this_day: int = 0
        self.initialized = False
        self.turn_count = 0

    def _log(self, msg: str):
        if self.verbose:
            print(f"[P{self.player_id}|{self.role}] {msg}")

    def _generate_llm_response(self, prompt: str, temperature: float = 0.6) -> str:
        """Generate a response using the LLM pipeline."""
        try:
            messages = [{"role": "user", "content": prompt}]
            if hasattr(self.tokenizer, 'apply_chat_template'):
                formatted = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                formatted = prompt
            
            response = self.pipeline(
                formatted,
                num_return_sequences=1,
                return_full_text=False,
                do_sample=True,
                temperature=temperature,
                top_p=0.9
            )
            return response[0]['generated_text'].strip()
        except Exception as e:
            self._log(f"LLM error: {e}")
            return ""

    def _analyze_statement_llm(self, speaker_id: int, statement: str):
        """Use LLM to analyze a player's statement for accusations, defenses, and role claims."""
        if speaker_id in self.teammates or speaker_id == self.player_id:
            return
        
        prompt = f"""Analyze this Mafia game statement from Player {speaker_id}:
"{statement}"

Extract ONLY what is explicitly stated. Respond in this exact format:
ACCUSES: [player numbers they accuse/suspect, or NONE]
DEFENDS: [player numbers they defend/trust, or NONE]
CLAIMS_ROLE: [Detective/Doctor/Villager/Mafia, or NONE]
CLAIMS_INVESTIGATED: [if they claim to have investigated someone, that player number, or NONE]
CONFIDENCE: [LOW/MEDIUM/HIGH - how confident are their claims]

Example response:
ACCUSES: 3, 5
DEFENDS: 1
CLAIMS_ROLE: Detective
CLAIMS_INVESTIGATED: 3
CONFIDENCE: HIGH"""

        response = self._generate_llm_response(prompt, temperature=0.3)
        self._parse_analysis_response(speaker_id, response)

    def _parse_analysis_response(self, speaker_id: int, response: str):
        """Parse the LLM analysis response and update game state."""
        response_lower = response.lower()
        
        # Parse accusations
        accuses_match = re.search(r"accuses:\s*([^\n]+)", response_lower)
        if accuses_match and "none" not in accuses_match.group(1):
            for num in re.findall(r"(\d+)", accuses_match.group(1)):
                target = int(num)
                if target < self.num_players and target != speaker_id:
                    self.player_accusations[speaker_id][target] += 1
        
        # Parse defenses
        defends_match = re.search(r"defends:\s*([^\n]+)", response_lower)
        if defends_match and "none" not in defends_match.group(1):
            for num in re.findall(r"(\d+)", defends_match.group(1)):
                target = int(num)
                if target < self.num_players:
                    self.player_defenses[speaker_id][target] += 1
        
        # Parse role claims
        role_match = re.search(r"claims_role:\s*(\w+)", response_lower)
        if role_match and "none" not in role_match.group(1):
            claimed_role = role_match.group(1).capitalize()
            if claimed_role == "Detective":
                self.claimed_detective = speaker_id
                self.belief_state[speaker_id]["Detective"] = 0.8
                self._log(f"LLM detected: Player {speaker_id} claims Detective")
        
        # Parse investigation claims
        investigated_match = re.search(r"claims_investigated:\s*(\d+)", response_lower)
        confidence_match = re.search(r"confidence:\s*(\w+)", response_lower)
        confidence = confidence_match.group(1) if confidence_match else "medium"
        
        if investigated_match:
            accused = int(investigated_match.group(1))
            if accused in self.alive_players and accused != self.player_id:
                self.detective_accusations[speaker_id] = accused
                
                if self.claimed_detective is None:
                    self.claimed_detective = speaker_id
                    self.belief_state[speaker_id]["Detective"] = 0.7
                
                old = self.suspicion_scores.get(accused, 0.5)
                boost = 0.5 if confidence == "high" else 0.35 if confidence == "medium" else 0.2
                if self.role in ["Villager", "Doctor"]:
                    self.suspicion_scores[accused] = min(1.0, old + boost)
                else:
                    self.suspicion_scores[accused] = min(1.0, old + boost * 0.6)
                self._log(f"LLM detected: Player {speaker_id} claims Player {accused} is Mafia (conf={confidence})")
        
        self._update_suspicion_from_analysis(speaker_id)

    def _update_suspicion_from_analysis(self, player_id: int):
        """Update suspicion scores based on analyzed behavior."""
        suspicion_delta = 0.0
        
        if len(self.player_accusations[player_id]) > 2:
            suspicion_delta += self.OVER_ACCUSATION_PENALTY
        
        for target in self.player_defenses[player_id]:
            if target in self.confirmed_mafia:
                suspicion_delta += self.DEFEND_MAFIA_PENALTY
                self._log(f"Player {player_id} defended confirmed Mafia {target}!")
        
        old = self.suspicion_scores.get(player_id, 0.5)
        self.suspicion_scores[player_id] = min(1.0, max(0.0, old + suspicion_delta))

    def _initialize_from_observation(self, observation: str):
        m = re.search(r"You are Player (\d+)", observation)
        if m: self.player_id = int(m.group(1))
        
        m = re.search(r"Your role: (\w+)", observation)
        if m: self.role = m.group(1)
        
        m = re.search(r"Team: (\w+)", observation)
        if m: self.team = m.group(1)
        
        m = re.search(r"Your teammates are: ([^.]+)\.", observation)
        if m:
            self.teammates = [int(x) for x in re.findall(r"Player (\d+)", m.group(1))]
            self.teammates = [t for t in self.teammates if t != self.player_id]
        
        m = re.search(r"Players: ([^\n]+)", observation)
        if m:
            count = len(re.findall(r"Player \d+", m.group(1)))
            if count > 0: self.num_players = count
        
        self.alive_players = set(range(self.num_players))
        self._initialize_belief_state()
        self._log(f"Init: {self.num_players}p, role={self.role}, teammates={self.teammates}")
        self.initialized = True

    def _initialize_belief_state(self):
        num_mafia = max(1, round(self.num_players * 0.25))
        mafia_prior = num_mafia / self.num_players
        
        for pid in range(self.num_players):
            if pid == self.player_id:
                self.belief_state[pid] = {"Mafia": 1.0 if self.role == "Mafia" else 0.0,
                                          "Doctor": 1.0 if self.role == "Doctor" else 0.0,
                                          "Detective": 1.0 if self.role == "Detective" else 0.0,
                                          "Villager": 1.0 if self.role == "Villager" else 0.0}
                self.suspicion_scores[pid] = 0.0
            elif pid in self.teammates:
                self.belief_state[pid] = {"Mafia": 1.0, "Doctor": 0.0, "Detective": 0.0, "Villager": 0.0}
                self.suspicion_scores[pid] = 0.0
                self.confirmed_mafia.add(pid)
            else:
                self.belief_state[pid] = {"Mafia": mafia_prior, "Doctor": 1/self.num_players,
                                          "Detective": 1/self.num_players,
                                          "Villager": 1.0 - mafia_prior - 2/self.num_players}
                self.suspicion_scores[pid] = 0.5

    def _update_game_state(self, observation: str):
        self._detect_phase(observation)
        self._track_eliminations(observation)
        self._parse_statements(observation)
        self._parse_votes(observation)
        self._handle_investigation_results(observation)
        
        if "Day breaks" in observation:
            self.day_number += 1
            self.statements_this_day = 0

    def _detect_phase(self, observation: str):
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
        patterns = [r"Player (\d+) was eliminated", r"Player (\d+) was killed", r"Player (\d+) has been eliminated"]
        for pattern in patterns:
            for pid_str in re.findall(pattern, observation):
                pid = int(pid_str)
                if pid in self.alive_players:
                    self.alive_players.discard(pid)
                    self.eliminated_players.append(pid)
                    self._log(f"Player {pid} eliminated")

    def _parse_statements(self, observation: str):
        matches = re.findall(r"\[Player (\d+)\]:\s*(.+?)(?=\[Player \d+\]:|$)", observation, re.DOTALL)
        for pid_str, statement in matches:
            pid = int(pid_str)
            statement = statement.strip()
            if statement and pid != self.player_id:
                self.player_statements[pid].append(statement)
                self.statement_count[pid] += 1
                
                if self.statement_count[pid] >= 2:
                    self.vocal_players.add(pid)
                
                self._analyze_statement_llm(pid, statement)

    def _parse_votes(self, observation: str):
        """Parse vote actions to track voting patterns."""
        vote_pattern = r"\[Player (\d+)\]:.*?\[(\d+)\]"
        for voter_str, target_str in re.findall(vote_pattern, observation):
            voter, target = int(voter_str), int(target_str)
            if voter != self.player_id:
                self.player_vote_history[voter].append(target)
                self.daily_votes[self.day_number][voter] = target
                
                if target not in self.confirmed_mafia and self.confirmed_mafia:
                    old = self.suspicion_scores.get(voter, 0.5)
                    self.suspicion_scores[voter] = min(1.0, old + self.VOTE_AGAINST_CONSENSUS_PENALTY)
                    self._log(f"Player {voter} voted {target} instead of confirmed Mafia - suspicious!")

    def _handle_investigation_results(self, observation: str):
        if self.role != "Detective":
            return
        
        mafia_match = re.search(r"Player (\d+) IS a Mafia member", observation)
        not_mafia_match = re.search(r"Player (\d+) IS NOT a Mafia member", observation)
        
        if mafia_match:
            target = int(mafia_match.group(1))
            self.confirmed_mafia.add(target)
            self.investigated_players.add(target)
            self.belief_state[target] = {"Mafia": 1.0, "Doctor": 0.0, "Detective": 0.0, "Villager": 0.0}
            self.suspicion_scores[target] = 1.0
            self._log(f"CONFIRMED: Player {target} is Mafia!")
        
        elif not_mafia_match:
            target = int(not_mafia_match.group(1))
            self.confirmed_village.add(target)
            self.investigated_players.add(target)
            self.belief_state[target]["Mafia"] = 0.0
            self.suspicion_scores[target] = max(0.0, self.suspicion_scores.get(target, 0.5) - 0.3)
            self._log(f"CONFIRMED: Player {target} is NOT Mafia")

    def _get_valid_targets(self, observation: str) -> List[int]:
        targets = re.findall(r"\[(\d+)\]", observation)
        return [int(t) for t in targets if int(t) in self.alive_players and int(t) != self.player_id]

    def _expectimax_select(self, candidates: List[int], is_night: bool = False) -> int:
        if not candidates:
            return -1
        
        best_action, best_utility = candidates[0], float('-inf')
        for target in candidates:
            utility = self._compute_utility(target, is_night) + random.gauss(0, 0.03)
            if utility > best_utility:
                best_utility = utility
                best_action = target
        
        self._log(f"Expectimax: {best_action} (utility={best_utility:.3f})")
        return best_action

    def _compute_utility(self, target: int, is_night: bool) -> float:
        beliefs = self.belief_state.get(target, {})
        suspicion = self.suspicion_scores.get(target, 0.5)
        
        if self.role == "Mafia":
            if target in self.teammates:
                return -100.0
            if is_night:
                utility = beliefs.get("Villager", 0) * self.VILLAGER_KILL_UTILITY
                if self.claimed_detective == target:
                    utility += 3.0
                if target in self.vocal_players:
                    utility += 0.5
                return utility
            else:
                return suspicion * 0.8
        
        elif self.role == "Doctor":
            if is_night:
                if target in self.confirmed_mafia:
                    return -100.0
                
                utility = beliefs.get("Detective", 0.17) * 2.0 + beliefs.get("Doctor", 0.17) * 0.5
                
                if self.claimed_detective == target:
                    utility += 3.0
                    self._log(f"Protecting claimed Detective: Player {target}")
                
                if target in self.vocal_players:
                    utility += 0.8
                
                if target in self.confirmed_village:
                    utility += 0.5
                
                utility -= suspicion * 0.2
                
                for accused in self.player_accusations.get(target, {}):
                    if accused in self.confirmed_mafia:
                        utility += 0.4
                
                return utility
            else:
                if target in self.confirmed_mafia:
                    return self.KNOWN_MAFIA_VOTE_UTILITY
                return suspicion * 1.2 + beliefs.get("Mafia", 0.33)
        
        elif self.role == "Detective":
            if is_night:
                if target in self.investigated_players:
                    return -100.0
                if target in self.confirmed_mafia or target in self.confirmed_village:
                    return -100.0
                return suspicion * 1.5 + 0.2
            else:
                if target in self.confirmed_mafia:
                    return self.KNOWN_MAFIA_VOTE_UTILITY
                return suspicion * 1.5 + beliefs.get("Mafia", 0.33)
        
        else:  # Villager
            if target in self.confirmed_mafia:
                return self.KNOWN_MAFIA_VOTE_UTILITY
            
            for accuser, accused in self.detective_accusations.items():
                if accused == target and accuser not in self.confirmed_mafia:
                    return 2.5
            
            utility = suspicion * 1.5 + beliefs.get("Mafia", 0.33) * 0.8
            
            for defended, count in self.player_defenses.get(target, {}).items():
                if defended in self.confirmed_mafia:
                    utility += 0.5 * count
            
            return utility

    def _is_night_phase(self, obs: str) -> bool:
        obs_lower = obs.lower()
        return any(x in obs_lower for x in ["night has fallen", "night phase - choose",
                   "mafia, agree on a victim", "choose one player to protect:", "choose one player to investigate:"])

    def _is_voting_phase(self, obs: str) -> bool:
        return "voting phase" in obs.lower() and "submit" in obs.lower()

    def _is_discussion_phase(self, obs: str) -> bool:
        if self._is_voting_phase(obs):
            return False
        obs_lower = obs.lower()
        return "day breaks" in obs_lower or "discuss" in obs_lower

    def __call__(self, observation: str) -> str:
        self.turn_count += 1
        
        if not self.initialized:
            self._initialize_from_observation(observation)
        
        self._update_game_state(observation)
        
        if self._is_night_phase(observation):
            return self._night_action(observation)
        elif self._is_voting_phase(observation):
            return self._voting_action(observation)
        elif self._is_discussion_phase(observation):
            return self._discussion_action(observation)
        else:
            return self._discussion_action(observation)

    def _night_action(self, observation: str) -> str:
        valid_targets = self._get_valid_targets(observation)
        if not valid_targets:
            excluded = {self.player_id} | set(self.teammates)
            valid_targets = [p for p in self.alive_players if p not in excluded]
        if not valid_targets:
            return "[0]"
        
        if random.random() < self.exploration_rate:
            target = random.choice(valid_targets)
        else:
            target = self._expectimax_select(valid_targets, is_night=True)
        
        return f"[{target}]"

    def _voting_action(self, observation: str) -> str:
        valid_targets = self._get_valid_targets(observation)
        if not valid_targets:
            valid_targets = [p for p in self.alive_players if p != self.player_id]
            if self.role == "Mafia":
                valid_targets = [p for p in valid_targets if p not in self.teammates]
        if not valid_targets:
            return "[0]"
        
        if self.role != "Mafia":
            confirmed_alive = [p for p in self.confirmed_mafia if p in valid_targets]
            if confirmed_alive:
                target = confirmed_alive[0]
                self._log(f"Voting confirmed Mafia: {target}")
                return f"[{target}]"
            
            for accuser, accused in self.detective_accusations.items():
                if accused in valid_targets and accuser in self.alive_players:
                    self._log(f"Voting detective-accused: {accused}")
                    return f"[{accused}]"
        
        target = self._expectimax_select(valid_targets, is_night=False)
        return f"[{target}]"

    def _discussion_action(self, observation: str) -> str:
        self.statements_this_day += 1
        
        # Mafia uses hardcoded strategic templates (more effective for deception)
        if self.role == "Mafia":
            return self._mafia_discussion()
        
        # Other roles use LLM for more natural/adaptive responses
        return self._generate_discussion_llm()

    def _mafia_discussion(self) -> str:
        """Hardcoded Mafia discussion - strategic deception templates."""
        # If someone is confirmed Mafia and it's not us/teammate, pretend to agree
        other_mafia = [p for p in self.confirmed_mafia if p in self.alive_players 
                       and p != self.player_id and p not in self.teammates]
        if other_mafia:
            return f"I agree, Player {other_mafia[0]} does seem suspicious."
        
        # Deflect to a village player who seems suspicious
        village_suspects = [p for p in self._get_top_suspects(exclude_confirmed=True) 
                          if p not in self.teammates]
        if village_suspects:
            target = village_suspects[0]
            phrases = [
                f"I've been watching Player {target} and they seem nervous.",
                f"Player {target} hasn't really helped the village. Suspicious.",
                f"Something about Player {target} doesn't feel right to me.",
                f"Has anyone else noticed Player {target} being evasive?",
                f"I'm getting bad vibes from Player {target}.",
                f"Player {target} is being too quiet. What are they hiding?",
            ]
            return random.choice(phrases)
        
        # Generic deflection
        deflections = [
            "Let's not rush to judgment. We need solid evidence before voting.",
            "I think we should hear from everyone before deciding.",
            "We need to be careful not to vote out an innocent.",
            "Does anyone have concrete evidence?",
        ]
        return random.choice(deflections)

    def _generate_discussion_llm(self) -> str:
        """Use LLM to generate contextually appropriate discussion message."""
        alive_list = sorted(self.alive_players)
        
        knowledge = []
        if self.confirmed_mafia:
            alive_mafia = [p for p in self.confirmed_mafia if p in self.alive_players]
            if alive_mafia:
                knowledge.append(f"Player(s) {alive_mafia} confirmed Mafia")
        if self.confirmed_village:
            alive_village = [p for p in self.confirmed_village if p in self.alive_players]
            if alive_village:
                knowledge.append(f"Player(s) {alive_village} confirmed innocent")
        
        suspects = self._get_top_suspects(exclude_confirmed=True, n=2)
        if suspects:
            suspect_info = [(p, round(self.suspicion_scores.get(p, 0.5), 2)) for p in suspects]
            knowledge.append(f"Most suspicious: {suspect_info}")
        
        for accuser, accused in self.detective_accusations.items():
            if accuser != self.player_id and accuser in self.alive_players and accused in self.alive_players:
                knowledge.append(f"Player {accuser} accused Player {accused} of being Mafia")
        
        knowledge_str = "; ".join(knowledge) if knowledge else "No confirmed information yet"
        
        # Role-specific strategic prompts
        if self.role == "Detective":
            alive_mafia = [p for p in self.confirmed_mafia if p in self.alive_players]
            if alive_mafia:
                # Critical: Be VERY direct about confirmed Mafia
                role_context = f"""You are the Detective and you KNOW Player {alive_mafia[0]} is Mafia.
Your goal: Convince everyone to vote out Player {alive_mafia[0]} NOW.
Be assertive and urgent. Say you investigated them. Rally the village."""
            else:
                role_context = """You are the Detective but haven't found Mafia yet.
Share your suspicions but don't reveal your role yet - you might get killed."""
                
        elif self.role == "Doctor":
            role_context = f"""You are the Doctor (keep this secret - Mafia will kill you).
Your goal: Help village find Mafia. Support any Detective claims you see.
Act like a helpful villager. Agree with credible accusations."""
        else:  # Villager
            # Check if there's a detective accusation to support
            active_accusations = [(acc, tgt) for acc, tgt in self.detective_accusations.items() 
                                 if acc in self.alive_players and tgt in self.alive_players]
            if active_accusations:
                accuser, target = active_accusations[0]
                role_context = f"""You are a Villager. Player {accuser} claims Player {target} is Mafia.
Your goal: Support this accusation strongly. Rally others to vote [{target}].
Be a team player - trust the Detective claim."""
            else:
                role_context = """You are a Villager with no special information.
Your goal: Share observations, ask questions, help find Mafia through logic."""
        
        recent_statements = []
        for pid in alive_list:
            if pid != self.player_id and self.player_statements[pid]:
                last_stmt = self.player_statements[pid][-1][:100]
                recent_statements.append(f"Player {pid}: {last_stmt}")
        recent_context = "\n".join(recent_statements[-3:]) if recent_statements else "No recent statements"
        
        prompt = f"""You are Player {self.player_id} in a Mafia game.

{role_context}

Alive players: {alive_list}
Your knowledge: {knowledge_str}

Recent discussion:
{recent_context}

Write ONE short sentence (10-20 words) to say in the discussion. Be strategic and natural.
Do not use brackets or player tags. Just output your message directly."""

        response = self._generate_llm_response(prompt, temperature=0.7)
        
        response = response.split('\n')[0].strip()
        response = re.sub(r'^\[.*?\]:\s*', '', response)
        response = re.sub(r'^(I would say:|My response:|Message:)\s*', '', response, flags=re.IGNORECASE)
        
        if not response or len(response) < 5:
            return self._fallback_discussion()
        
        return response[:200]

    def _fallback_discussion(self) -> str:
        """Fallback discussion in case LLM fails."""
        suspects = self._get_top_suspects(exclude_confirmed=True)
        
        # Detective with confirmed Mafia - be VERY direct
        if self.role == "Detective" and self.confirmed_mafia:
            alive_mafia = [p for p in self.confirmed_mafia if p in self.alive_players]
            if alive_mafia:
                target = alive_mafia[0]
                phrases = [
                    f"I am the Detective. I investigated Player {target} and they ARE Mafia. Vote them NOW.",
                    f"Listen everyone - I'm Detective and Player {target} is confirmed Mafia. We must eliminate them.",
                    f"Player {target} is Mafia - I investigated them. Trust me, vote them out.",
                ]
                return random.choice(phrases)
        
        # Villager/Doctor - support detective accusations
        if self.role in ["Villager", "Doctor"]:
            for accuser, accused in self.detective_accusations.items():
                if accused in self.alive_players and accuser in self.alive_players:
                    return f"I believe Player {accuser}. We should vote Player {accused}."
        
        if suspects:
            return f"I find Player {suspects[0]} suspicious based on their behavior."
        
        return "We need to share information and work together to find the Mafia."

    def _get_top_suspects(self, exclude_confirmed: bool = True, n: int = 2) -> List[int]:
        candidates = []
        for pid in self.alive_players:
            if pid == self.player_id or pid in self.teammates:
                continue
            if exclude_confirmed and pid in self.confirmed_village:
                continue
            candidates.append((pid, self.suspicion_scores.get(pid, 0.5)))
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [pid for pid, _ in candidates[:n]]

    def get_belief_summary(self) -> str:
        lines = ["=== Belief State ==="]
        for pid in sorted(self.alive_players):
            suspicion = self.suspicion_scores.get(pid, 0.5)
            status = "MAFIA" if pid in self.confirmed_mafia else "VILLAGE" if pid in self.confirmed_village else ""
            lines.append(f"P{pid}: susp={suspicion:.2f} {status}")
        return "\n".join(lines)


def test_agent():
    """Test the ExpectimaxMafiaAgent."""
    print("Loading ExpectimaxMafiaAgent with LLM...")
    print("(First run may take a moment to download the model)\n")
    
    agent = ExpectimaxMafiaAgent(
        model_name="microsoft/Phi-3-mini-4k-instruct",
        verbose=True,
        quantize=False
    )
    
    print("=" * 60)
    print("TEST 1: First observation (Welcome + Night phase as Detective)")
    print("=" * 60)
    
    first_obs = """Welcome to Secret Mafia! You are Player 2.
Your role: Detective
Team: Village

Players: Player 0, Player 1, Player 2, Player 3, Player 4, Player 5

Night phase - choose one player to investigate: [0], [1], [3], [4], [5]"""
    
    response = agent(first_obs)
    print(f"Response: {response}")
    print(agent.get_belief_summary())
    
    assert re.search(r"\[\d+\]", response), "Night action should be in [X] format"
    print("✓ Night action format correct")
    
    print("\n" + "=" * 60)
    print("TEST 2: Investigation result - found Mafia")
    print("=" * 60)
    
    result_obs = """Player 3 IS a Mafia member.

Day breaks. Discuss for 3 rounds, then a vote will follow."""
    
    response = agent(result_obs)
    print(f"Response: {response}")
    print(agent.get_belief_summary())
    
    assert 3 in agent.confirmed_mafia, "Player 3 should be confirmed Mafia"
    assert 3 in agent.investigated_players, "Player 3 should be marked as investigated"
    print("✓ Mafia confirmation tracked correctly")
    print("✓ Investigated players tracked correctly")
    
    print("\n" + "=" * 60)
    print("TEST 3: Day Discussion - LLM generates strategic response")
    print("=" * 60)
    
    day_obs = """[Player 0]: I think we should be careful today.
[Player 1]: Does anyone have any leads?"""
    
    response = agent(day_obs)
    print(f"LLM Response: {response}")
    
    assert len(response) > 5, "Discussion should have content"
    assert len(response) < 250, "Discussion should be concise"
    print("✓ LLM generated valid discussion response")
    
    print("\n" + "=" * 60)
    print("TEST 4: Voting Phase - should vote for confirmed Mafia [3]")
    print("=" * 60)
    
    vote_obs = """Voting phase - submit one vote in format [X]. Valid: [0], [1], [3], [4], [5]"""
    
    response = agent(vote_obs)
    print(f"Response: {response}")
    
    vote_match = re.search(r"\[(\d+)\]", response)
    assert vote_match, "Vote should be in [X] format"
    if vote_match.group(1) == '3':
        print("✓ Correctly voted for confirmed Mafia")
    else:
        print(f"✗ Voted for {vote_match.group(1)} instead of confirmed Mafia 3")
    
    print("\n" + "=" * 60)
    print("TEST 5: New game as Mafia")
    print("=" * 60)
    
    agent.reset()
    
    mafia_obs = """Welcome to Secret Mafia! You are Player 1.
Your role: Mafia
Team: Mafia

Players: Player 0, Player 1, Player 2, Player 3, Player 4, Player 5

Your teammates are: Player 1, Player 4.

Night has fallen. Mafia, agree on a victim.
Valid targets: [0], [2], [3], [5]"""
    
    response = agent(mafia_obs)
    print(f"Response: {response}")
    
    vote_match = re.search(r"\[\d+\]", response)
    assert vote_match, "Night action should be in [X] format"
    target = int(vote_match.group(0).strip("[]"))
    assert target not in [1, 4], "Mafia should not target teammates"
    print(f"✓ Mafia targeted Player {target} (not a teammate)")
    
    print("\n" + "=" * 60)
    print("TEST 6: Doctor protection logic")
    print("=" * 60)
    
    agent.reset()
    
    doctor_obs = """Welcome to Secret Mafia! You are Player 0.
Your role: Doctor
Team: Village

Players: Player 0, Player 1, Player 2, Player 3, Player 4, Player 5

Night phase - choose one player to protect: [1], [2], [3], [4], [5]"""
    
    response = agent(doctor_obs)
    print(f"Response: {response}")
    
    vote_match = re.search(r"\[\d+\]", response)
    assert vote_match, "Protection should be in [X] format"
    print(f"✓ Doctor chose to protect Player {vote_match.group(0)}")
    
    agent.claimed_detective = 2
    agent.belief_state[2]["Detective"] = 0.8
    
    doctor_night2 = """Night phase - choose one player to protect: [1], [2], [3], [4], [5]"""
    response = agent(doctor_night2)
    print(f"After Detective claim, protection: {response}")
    
    vote_match = re.search(r"\[\d+\]", response)
    if vote_match and vote_match.group(0) == '[2]':
        print("✓ Doctor correctly prioritized protecting claimed Detective")
    else:
        print(f"Note: Doctor protected {vote_match.group(0)} (may vary due to exploration)")
    
    print("\n" + "=" * 60)
    print("TEST 7: LLM statement analysis")
    print("=" * 60)
    
    agent.reset()
    agent._initialize_from_observation("""Welcome to Secret Mafia! You are Player 0.
Your role: Villager
Team: Village
Players: Player 0, Player 1, Player 2, Player 3, Player 4, Player 5""")
    
    test_statement = "I investigated Player 3 last night and they are definitely Mafia!"
    print(f"Analyzing statement: '{test_statement}'")
    agent._analyze_statement_llm(2, test_statement)
    
    print(f"Claimed detective: {agent.claimed_detective}")
    print(f"Detective accusations: {dict(agent.detective_accusations)}")
    print(f"Suspicion scores: {dict(agent.suspicion_scores)}")
    
    if agent.claimed_detective == 2:
        print("✓ LLM correctly identified Detective claim")
    if 3 in agent.suspicion_scores and agent.suspicion_scores[3] > 0.5:
        print("✓ LLM correctly increased suspicion on accused player")
    
    print("\n" + "=" * 60)
    print("TEST 8: Discussion generation for different roles")
    print("=" * 60)
    
    # Test Mafia discussion (hardcoded templates)
    agent.reset()
    agent._initialize_from_observation("""Welcome to Secret Mafia! You are Player 1.
Your role: Mafia
Team: Mafia
Players: Player 0, Player 1, Player 2, Player 3, Player 4, Player 5
Your teammates are: Player 1, Player 4.""")
    
    response = agent._mafia_discussion()
    print(f"Mafia discussion (template): {response}")
    assert "i am mafia" not in response.lower(), "Mafia shouldn't reveal role"
    assert len(response) > 10, "Should have meaningful content"
    print("✓ Mafia generated appropriate cover discussion")
    
    # Test Villager discussion (LLM)
    agent.reset()
    agent._initialize_from_observation("""Welcome to Secret Mafia! You are Player 0.
Your role: Villager
Team: Village
Players: Player 0, Player 1, Player 2, Player 3, Player 4, Player 5""")
    
    response = agent._generate_discussion_llm()
    print(f"Villager discussion (LLM): {response}")
    assert len(response) > 5, "Should have content"
    print("✓ Villager generated LLM discussion")
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_agent()