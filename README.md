# RP Pipeline

A restructured, modular roleplay dataset generation pipeline for creating high-quality RP scenes from character cards.

## Structure

```
restructured/pipeline/
├── config/
│   ├── prompts.yaml      # All system prompts for generation, analysis, judging
│   └── settings.yaml     # Default configuration (models, paths, limits, etc.)
├── scripts/
│   ├── generate.py       # Generate scenes from character cards
│   ├── analyze.py        # Analyze scenes for tics and quality issues
│   ├── cleanup.py        # Clean scenes by removing tics
│   ├── rewrite.py        # Rewrite scenes using LLM for quality improvement
│   ├── judge.py          # Judge scenes using LLM-based evaluation
│   └── full_pipeline.py  # End-to-end pipeline orchestrator
├── src/rp_pipeline/
│   ├── config/
│   │   └── settings.py   # Settings management with YAML + env var support
│   ├── core/
│   │   ├── generation.py # Scene generation logic
│   │   ├── analysis.py   # Tic detection and quality analysis
│   │   └── cleanup.py    # Scene cleanup and rewriting
│   ├── data/
│   │   ├── schemas.py    # Pydantic data models (CharacterCard, Scene, Turn, etc.)
│   │   └── cards.py      # Card database and formatting
│   ├── models/
│   │   ├── base.py       # Abstract model provider interface
│   │   ├── openrouter.py  # OpenRouter API implementation
│   │   ├── featherless.py # Featherless API implementation
│   │   └── nvidia.py      # NVIDIA API implementation
│   └── utils/
│       ├── logging.py    # Structured logging (JSON/text, file/console)
│       └── caching.py     # Disk/memory caching and checkpointing
└── requirements.txt
```

---

## Quick Start

### 1. Install Dependencies

```bash
cd restructured/pipeline
pip install -r requirements.txt
```

### 2. Configure API Keys

Set your API keys as environment variables:

```bash
export OPENROUTER_API_KEY="your-openrouter-key"
export FEATHERLESS_API_KEY="your-featherless-key"
export NVIDIA_API_KEY="your-nvidia-key"
```

Or update `config/settings.yaml` with your preferred provider/model combinations.

### 3. Prepare Character Cards

Character cards should be in JSONL format with these fields:

```json
{
  "assistant_name": "Alice",
  "user_name": "Bob",
  "assistant_character": "Alice is a quick-witted detective...",
  "user_character": "Bob is a retired professor...",
  "scenario": "Alice and Bob meet in a dimly lit library...",
  "genre": "mystery",
  "tone": "suspenseful",
  "assistant_appearance": "Tall, with dark hair and sharp eyes...",
  "user_appearance": "Short, with a salt-and-pepper beard..."
}
```

Save cards in `data/input/cards/` (one per line in `.jsonl` files) or specify path via `--cards`.

---

## Configuration

### YAML Config (`config/settings.yaml`)

```yaml
# Model providers
models:
  providers:
    arli_ai:
      base_url: https://api.arli_ai.com/v1
      api_key_env: ARLI_API_KEY
    openrouter:
      base_url: https://openrouter.ai/api/v1
      api_key_env: OPENROUTER_API_KEY
    featherless:
      base_url: https://api.featherless.ai/v1
      api_key_env: FEATHERLESS_API_KEY

# Default models for each role
defaults:
  generation:
    provider: arli_ai
    model: DeepSeek-V4-Flash-0731
    max_tokens: 4096
    temperature: 0.85
  judging:
    provider: arli_ai
    model: DeepSeek-V4-Flash-0731
    max_tokens: 1200
    temperature: 0.3
  rewriting:
    provider: arli_ai
    model: DeepSeek-V4-Flash-0731
    max_tokens: 5000
    temperature: 0.7

# Generation parameters
generation:
  cards_per_batch: 5
  variants_per_card: 3
  target_turns: 8
  turn_length: long  # short, medium, or long
  either_opener: true

# Paths
paths:
  input:
    cards_dir: data/input/cards
    default_cards: card-compare/kimi-batch/cards.jsonl
  output:
    base: data/output
    raw: data/output/raw
    analyzed: data/output/analyzed
    cleaned: data/output/cleaned
    final: data/output/final
  cache: data/cache
  logs: data/logs

# Limits
limits:
  max_concurrent: 4
  max_retries: 3
  rate_limit_delay: 60
  timeout: 300

# Quality thresholds
quality:
  min_token_count: 50
  max_token_count: 6144
  min_turn_count: 4
  max_turn_count: 12

# Logging
logging:
  level: INFO
  format: json  # or text
  file: data/logs/pipeline.log

# Caching
cache:
  enabled: true
  dir: data/cache
  ttl: 86400  # 24 hours
```

### CLI Arguments Override Config

All CLI scripts accept arguments that override the YAML configuration. For example:

```bash
# Override model and provider
python scripts/generate.py --model deepseek-ai/DeepSeek-V4-Pro --provider featherless

# Override output directory
python scripts/generate.py --output my_output/scenes

# Override generation parameters
python scripts/generate.py --target-turns 10 --turn-length medium --batch-size 10
```

---

## CLI Scripts

### generate.py - Generate Scenes

```bash
python scripts/generate.py \
  --cards data/input/cards/cards.jsonl \
  --output data/output/raw \
  --model xiaomi/mimo-v2.5-pro \
  --provider openrouter \
  --batch-size 5 \
  --variants 3 \
  --target-turns 8 \
  --turn-length long \
  --either-opener \
  --max-tokens 4096 \
  --temperature 0.85 \
  --resume \
  --no-checkpoint
```

**Output**: JSONL files in OpenAI messages format:
```json
{"id": "scene_0001", "card_id": "Alice|Bob", "messages": [{"role": "user", "content": "..."}, ...], "metadata": {...}}
```

### analyze.py - Analyze Scenes for Tics

```bash
python scripts/analyze.py \
  --input data/output/raw \
  --output data/output/analyzed \
  --tic-rate-threshold 5.0 \
  --resume
```

Detects:
- Emotion telling (felt, knew, realized, etc.)
- Hedging words (somehow, somewhat, etc.)
- Lazy comparisons ("like a...", "the way a...")
- Narrator intrusion
- Physical clichés
- Dialogue tags
- Pronoun repetition (3+ consecutive same openers)

**Output**: Analysis results with tic counts, emotion tells, and recommendations.

### cleanup.py - Clean Scenes

```bash
python scripts/cleanup.py \
  --input data/output/raw \
  --analysis data/output/analyzed \
  --output data/output/cleaned \
  --use-rewrite \
  --no-rewrite \
  --rewrite-model xiaomi/mimo-v2.5-pro \
  --rewrite-provider openrouter
```

Removes tics via:
1. Pattern matching (direct text replacement)
2. LLM rewriting (optional, for complex issues)

**Output**: Cleaned scenes in JSONL format.

### rewrite.py - Rewrite Scenes for Quality

```bash
python scripts/rewrite.py \
  --input data/output/cleaned \
  --output data/output/final \
  --model xiaomi/mimo-v2.5-pro \
  --provider openrouter \
  --style-rewrite \
  --max-tokens 5000 \
  --temperature 0.7
```

Uses LLM to substantially improve scenes:
- Better initiative from ASSISTANT character
- Each turn changes something (no spinning in place)
- Emotions shift in response to events
- Actions/dialogue carry meaning without narrator explanation
- Varied turn lengths matching the beat

### judge.py - Judge Scene Quality

```bash
python scripts/judge.py \
  --input data/output/final \
  --output data/output/judged \
  --model deepseek-ai/DeepSeek-V4-Pro \
  --provider featherless \
  --judge-type both  # behavioral, style, or both
```

Evaluates on:
- **Behavioral**: Initiative, responsiveness, emotional range, structural variety, subtext trust, scene advancement
- **Style**: Concrete vs abstract, functional vs decorative, word precision, sentence rhythm, dialogue craft, show vs tell, freshness vs cliché

### full_pipeline.py - End-to-End Pipeline

```bash
python scripts/full_pipeline.py \
  --cards data/input/cards/cards.jsonl \
  --output data/output \
  --model xiaomi/mimo-v2.5-pro \
  --provider openrouter \
  --batch-size 5 \
  --variants 3 \
  --skip-generate \
  --skip-analyze \
  --skip-cleanup \
  --skip-rewrite \
  --no-rewrite \
  --resume
```

Runs all stages in order. Use `--skip-*` to skip specific stages.

---

## Output Format

All scripts output JSONL (JSON Lines) files in **OpenAI messages format**:

```json
{
  "id": "scene_0001",
  "card_id": "Alice|Bob",
  "messages": [
    {"role": "user", "content": "Hello, Alice."},
    {"role": "assistant", "content": "Hi Bob! How are you?"}
  ],
  "metadata": {
    "genre": "mystery",
    "tone": "suspenseful",
    "turn_count": 8,
    "total_word_count": 500,
    "total_token_count": 650
  }
}
```

Analysis and judgment scripts add their results to the metadata or as separate fields.

---

## Python API Usage

### Generate Scenes

```python
from rp_pipeline.core.generation import SceneGenerator
from rp_pipeline.data.cards import CardDatabase

# Load cards
card_db = CardDatabase("data/input/cards/cards.jsonl")

# Generate scenes
generator = SceneGenerator()
for card in card_db:
    scene, response = generator.generate_scene(card, target_turns=8)
    print(f"Generated scene with {scene.turn_count} turns")
```

### Analyze Scenes

```python
from rp_pipeline.core.analysis import SceneAnalyzer

analyzer = SceneAnalyzer()
tic_result, quality_result = analyzer.analyze(scene)

print(f"Tics found: {tic_result.total_tic_count}")
print(f"Tic rate: {tic_result.tic_rate} per 1000 words")
print(f"Needs cleanup: {tic_result.needs_cleanup}")
```

### Clean Scenes

```python
from rp_pipeline.core.cleanup import SceneCleaner

cleaner = SceneCleaner()
cleanup_result = cleaner.clean_scene(
    scene,
    tic_result=tic_result,
    quality_issues=quality_result,
    use_rewrite=True
)

print(f"Changes made: {cleanup_result.changes_made}")
print(f"Tics removed: {cleanup_result.tics_removed}")
```

### Logging

```python
from rp_pipeline.utils.logging import StructuredLogger, get_logger

# Get the global logger
logger = StructuredLogger()
logger.info("Starting pipeline", stage="generate", batch_size=5)

# Or get a named logger
scene_logger = get_logger("generation")
scene_logger.info("Generated scene", scene_id="0001")
```

### Caching

```python
from rp_pipeline.utils.caching import DiskCache, get_checkpoint

# Disk cache for intermediate results
cache = DiskCache()
cache.set("my_key", my_value, ttl=3600)  # 1 hour TTL
value = cache.get("my_key")

# Checkpointing for resumable pipelines
checkpoint = get_checkpoint("generate")
checkpoint.start_stage("generate")
checkpoint.update("card_001", success=True)
checkpoint.mark_complete()
```

---

## Environment Variables

| Variable | Purpose | Provider |
|----------|---------|----------|
| `ARLI_API_KEY` | Arli AI API key | Arli AI |
| `OPENROUTER_API_KEY` | OpenRouter API key | OpenRouter |
| `FEATHERLESS_API_KEY` | Featherless API key | Featherless |
| `NVIDIA_API_KEY` | NVIDIA API key | NVIDIA |
| `GEN_MODEL` | Override default generation model | - |
| `JUDGE_MODEL` | Override default judging model | - |
| `MAX_TOKENS` | Override default max tokens | - |
| `TEMPERATURE` | Override default temperature | - |
| `OUTPUT_DIR` | Override default output directory | - |
| `CARDS_FILE` | Override default cards file | - |

---

## File Structure

```
restructured/pipeline/
├── config/
│   ├── prompts.yaml      # System prompts for all stages
│   └── settings.yaml     # Default configuration
├── data/
│   ├── input/
│   │   └── cards/        # Character cards (JSONL)
│   ├── output/
│   │   ├── raw/          # Generated scenes
│   │   ├── analyzed/     # Analysis results
│   │   ├── cleaned/      # Cleaned scenes
│   │   ├── final/        # Rewritten scenes
│   │   └── judged/       # Judgment results
│   ├── cache/            # Checkpoint and cache files
│   └── logs/             # Log files
├── scripts/              # CLI scripts
└── src/rp_pipeline/      # Python package
```

---

## Prompts

All system prompts are defined in `config/prompts.yaml`. You can customize:

- Card generation prompts
- Scene generation prompts (with turn length variants)
- Judge prompts (behavioral and style)
- Rewrite prompts
- Direction generation prompts

---

## Quality Thresholds

Configure in `config/settings.yaml` under `quality:`

```yaml
quality:
  min_token_count: 50
  max_token_count: 6144
  min_turn_count: 4
  max_turn_count: 12
  similarity_threshold: 0.85
  tic_rate_threshold: 5.0  # Tics per 1000 words to flag for cleanup
```

---

## Troubleshooting

### "No module named 'rp_pipeline'"

Make sure you're running from the `restructured/pipeline/` directory or have `src/` in your PYTHONPATH:

```bash
cd restructured/pipeline
PYTHONPATH=src:$PYTHONPATH python scripts/generate.py
```

### API Key Not Found

Set the environment variable for your provider:

```bash
export OPENROUTER_API_KEY="your-key-here"
```

### File Not Found Errors

Check that your input paths exist and are accessible. Use absolute paths if needed.

---

## License

This is part of the glitter-project repository.
