# ExpectimaxMafiaAgent - Testing Instructions

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set HuggingFace token

Get a token at: https://huggingface.co/settings/tokens

### 3. Set OpenRouter token
Create a `.env` file in the `src/` folder:
```
OPENROUTER_API_KEY=your_key_here
```
Get a key at: https://openrouter.ai/keys

## Running Tests

### Unit tests
```bash
python myagent.py
```

### Evaluation against LLM opponents
Edit `NUM_EPISODES` in `offline_ta_eval.py` to set number of games, then:
```bash
python offline_ta_eval.py
python offline_evaluation.py
```
offline_ta_eval.py is default setup - myagent vs 5 standard agents (OpenRouter agent = google/gemini-2.0-flash-lite-001)
offline_evaluation.py is default setup - myagent vs 5 simple random agents

ask mac for help if needed