# ExpectimaxMafiaAgent

Mafia agent combining LLM reasoning with expectimax decision-making for the Social Deduction Track.

## Files Added

| File | Description |
|------|-------------|
| `src/myagent.py` | ExpectimaxMafiaAgent implementation |
| `src/offline_play.py` | Test script for local games |
| `requirements.txt` | install this using pip |

## Features

- **Expectimax action selection** - Computes expected utility for targeting decisions
- **Belief state tracking** - Maintains role probabilities for each player
- **Suspicion scoring** - Detects deception via behavioral heuristics (contradictions, over-accusation, vote-switching)
- **LLM integration** - Uses Phi-3-mini for discussion generation
- **Role-specific strategies** - Different logic for Mafia, Detective, Doctor, Villager

## Testing

```bash
# Run unit tests
python src/myagent.py

# Run full game against random agents
python src/offline_play.py
```

## Usage

```python
from myagent import ExpectimaxMafiaAgent

agent = ExpectimaxMafiaAgent(
    model_name="microsoft/Phi-3-mini-4k-instruct",
    verbose=True
)
```