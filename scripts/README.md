# CLI Scripts

This directory contains command-line scripts for running the RP Pipeline.

## Scripts Overview

| Script | Purpose | Typical Input | Typical Output |
|--------|---------|---------------|----------------|
| `generate.py` | Generate RP scenes from character cards | Character cards (JSONL) | Raw scenes (JSONL) |
| `analyze.py` | Analyze scenes for tics and quality issues | Raw scenes | Analysis results (JSONL) |
| `cleanup.py` | Clean scenes by removing tics | Raw scenes + analysis | Cleaned scenes (JSONL) |
| `rewrite.py` | Rewrite scenes for quality improvement | Cleaned scenes | Final scenes (JSONL) |
| `judge.py` | Judge scene quality using LLM | Final scenes | Judgment results (JSONL) |
| `full_pipeline.py` | Run all stages end-to-end | Character cards | All intermediate + final outputs |

---

## Common Options

All scripts support these common options:

```
--config PATH         Path to settings YAML file (default: config/settings.yaml)
--log-level LEVEL     Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL (default: INFO)
--log-format FORMAT   Log format: json or text (default: from config)
--resume             Resume from last checkpoint
--no-checkpoint      Disable checkpointing
```

---

## generate.py

Generate RP scenes from character cards.

### Usage

```bash
python generate.py [OPTIONS]
```

### Options

```
# Input/Output
--cards, -c PATH        Path to cards JSONL file or directory
--output, -o PATH       Output directory for generated scenes

# Model Configuration
--model, -m NAME       Model to use for generation
--provider, -p NAME    Provider: openrouter, featherless, nvidia

# Generation Parameters
--batch-size, -b N      Number of cards per batch (default: from config)
--variants, -v N        Variants to generate per card (default: from config)
--target-turns, -t N    Target number of turns per scene (default: from config)
--turn-length, -l TYPE  Turn length: short, medium, long (default: from config)
--either-opener         Allow either USER or ASSISTANT to open (default: True)
--no-either-opener     Force USER to open

# Model Parameters
--max-tokens N         Maximum tokens for generation
--temperature F        Temperature for generation
```

### Examples

```bash
# Generate with default config
python generate.py --cards data/input/cards/cards.jsonl

# Generate with specific model
python generate.py \
  --cards data/input/cards/cards.jsonl \
  --model xiaomi/mimo-v2.5-pro \
  --provider openrouter

# Generate with custom parameters
python generate.py \
  --cards data/input/cards/cards.jsonl \
  --batch-size 10 \
  --variants 5 \
  --target-turns 10 \
  --turn-length medium \
  --output my_output/scenes

# Resume interrupted generation
python generate.py \
  --cards data/input/cards/cards.jsonl \
  --resume
```

### Output Format

Each scene is saved as a separate JSONL file in OpenAI messages format:

```json
{
  "id": "Alice_Bob_v0",
  "card_id": "Alice|Bob",
  "messages": [
    {"role": "user", "content": "[USER - Turn 1] Hello, Alice."},
    {"role": "assistant", "content": "[ASSISTANT - Turn 2] Hi Bob! How are you?"}
  ],
  "metadata": {
    "assistant_name": "Alice",
    "user_name": "Bob",
    "genre": "slice-of-life",
    "tone": "friendly",
    "turn_count": 8,
    "total_word_count": 450,
    "total_token_count": 585
  }
}
```

---

## analyze.py

Analyze scenes for tics (emotion-telling, clichés, etc.) and quality issues.

### Usage

```bash
python analyze.py [OPTIONS]
```

### Options

```
# Input/Output
--input, -i PATH       Input directory or file with scenes (JSONL)
--output, -o PATH      Output directory for analysis results

# Analysis Parameters
--tic-rate-threshold F  Tic rate threshold for flagging (default: 5.0)
```

### Examples

```bash
# Analyze all scenes in output directory
python analyze.py --input data/output/raw --output data/output/analyzed

# Analyze with custom threshold
python analyze.py \
  --input data/output/raw \
  --output data/output/analyzed \
  --tic-rate-threshold 3.0

# Analyze single file
python analyze.py --input data/output/raw/scene_001.jsonl --output data/output/analyzed
```

### Output Format

```json
{
  "scene_id": "Alice_Bob_v0",
  "card_id": "Alice|Bob",
  "metadata": {...},
  "scene_stats": {
    "turn_count": 8,
    "total_word_count": 450,
    "total_token_count": 585
  },
  "tic_analysis": {
    "tics": {
      "emotion_telling": 3,
      "hedging": 1,
      "lazy_comparisons": 2
    },
    "emotion_tells": ["felt happy", "knew she was"],
    "total_tic_count": 6,
    "tic_rate": 13.33,
    "needs_cleanup": true
  },
  "quality_analysis": {
    "turn_variety": {"passes": true, "issues": []},
    "scene_advancement": {"passes": true, "issues": []},
    "responsiveness": {"passes": true, "issues": []},
    "overall_pass": true
  }
}
```

---

## cleanup.py

Clean scenes by removing detected tics.

### Usage

```bash
python cleanup.py [OPTIONS]
```

### Options

```
# Input/Output
--input, -i PATH       Input directory or file with scenes (JSONL)
--analysis, -a PATH    Directory with analysis results (to reuse existing analysis)
--output, -o PATH      Output directory for cleaned scenes

# Cleanup Options
--use-rewrite          Use LLM rewriting for complex issues (default: True)
--no-rewrite           Use pattern matching only (no LLM calls)
--rewrite-model NAME   Model for rewriting (default: from config)
--rewrite-provider NAME Provider for rewriting: openrouter, featherless, nvidia
```

### Examples

```bash
# Clean with LLM rewrite
python cleanup.py \
  --input data/output/raw \
  --output data/output/cleaned

# Clean with pattern matching only (faster, no API calls)
python cleanup.py \
  --input data/output/raw \
  --output data/output/cleaned \
  --no-rewrite

# Clean with specific rewrite model
python cleanup.py \
  --input data/output/raw \
  --output data/output/cleaned \
  --rewrite-model deepseek-ai/DeepSeek-V4-Pro \
  --rewrite-provider featherless

# Reuse existing analysis
python cleanup.py \
  --input data/output/raw \
  --analysis data/output/analyzed \
  --output data/output/cleaned
```

### Output Format

Same as `generate.py` but with cleanup metadata:

```json
{
  "scene_id": "Alice_Bob_v0",
  "card_id": "Alice|Bob",
  "messages": [...],
  "metadata": {...},
  "cleanup_info": {
    "changes_made": ["Turn 2: removed 2 tics", "Turn 5: fixed pronoun repetition"],
    "tics_removed": {"emotion_telling": 2, "hedging": 1},
    "validation_passed": true
  }
}
```

---

## rewrite.py

Rewrite scenes using LLM for substantial quality improvement.

### Usage

```bash
python rewrite.py [OPTIONS]
```

### Options

```
# Input/Output
--input, -i PATH       Input directory or file with scenes (JSONL)
--output, -o PATH      Output directory for rewritten scenes

# Rewrite Options
--model, -m NAME       Model for rewriting (default: from config)
--provider, -p NAME    Provider: openrouter, featherless, nvidia
--style-rewrite        Use style-focused rewrite (more aggressive prose improvement)

# Model Parameters
--max-tokens N         Maximum tokens for rewriting
--temperature F        Temperature for rewriting
```

### Examples

```bash
# Rewrite with default config
python rewrite.py --input data/output/cleaned --output data/output/final

# Rewrite with specific model
python rewrite.py \
  --input data/output/cleaned \
  --output data/output/final \
  --model xiaomi/mimo-v2.5-pro \
  --provider openrouter

# Style-focused rewrite (more aggressive)
python rewrite.py \
  --input data/output/cleaned \
  --output data/output/final \
  --style-rewrite
```

### Output Format

Same as `generate.py` - rewritten scenes in OpenAI messages format.

---

## judge.py

Judge scene quality using LLM-based evaluation.

### Usage

```bash
python judge.py [OPTIONS]
```

### Options

```
# Input/Output
--input, -i PATH       Input directory or file with scenes (JSONL)
--output, -o PATH      Output directory for judgments

# Judge Options
--model, -m NAME       Model for judging (default: from config)
--provider, -p NAME    Provider: openrouter, featherless, nvidia
--judge-type, -j TYPE  Judgment type: behavioral, style, both (default: behavioral)

# Model Parameters
--max-tokens N         Maximum tokens for judgment
--temperature F        Temperature for judgment
```

### Examples

```bash
# Judge with behavioral analysis
python judge.py --input data/output/final --output data/output/judged

# Judge with both behavioral and style analysis
python judge.py \
  --input data/output/final \
  --output data/output/judged \
  --judge-type both

# Judge with specific model
python judge.py \
  --input data/output/final \
  --output data/output/judged \
  --model deepseek-ai/DeepSeek-V4-Pro \
  --provider featherless
```

### Output Format

```json
{
  "scene_id": "Alice_Bob_v0",
  "card_id": "Alice|Bob",
  "metadata": {...},
  "scene_stats": {...},
  "judgments": {
    "behavioral": "The ASSISTANT character shows good initiative...",
    "style": "Prose is generally concrete and specific..."
  }
}
```

---

## full_pipeline.py

Run the complete pipeline end-to-end.

### Usage

```bash
python full_pipeline.py [OPTIONS]
```

### Options

```
# Input
--cards, -c PATH       Path to cards JSONL file or directory
--output-dir, -o PATH  Base output directory (default: data/output)

# Stage Toggles
--skip-generate        Skip generation stage
--skip-analyze         Skip analysis stage
--skip-cleanup         Skip cleanup stage
--skip-rewrite         Skip rewrite stage

# Generation Options (passed to generate.py)
--model, -m NAME       Model for generation
--provider, -p NAME    Provider for generation
--batch-size, -b N      Batch size
--variants, -v N        Variants per card
--target-turns, -t N    Target turns
--turn-length, -l TYPE  Turn length

# Cleanup Options (passed to cleanup.py)
--no-rewrite           Disable LLM rewriting in cleanup
--rewrite-model NAME   Model for rewriting

# Checkpointing
--resume             Resume from checkpoints
--no-checkpoint      Disable all checkpointing

# Logging
--log-level LEVEL     Logging level
--log-format FORMAT   Log format
```

### Examples

```bash
# Run complete pipeline
python full_pipeline.py \
  --cards data/input/cards/cards.jsonl \
  --output data/output

# Run with custom model
python full_pipeline.py \
  --cards data/input/cards/cards.jsonl \
  --model xiaomi/mimo-v2.5-pro \
  --provider openrouter

# Skip analysis and cleanup, just generate and rewrite
python full_pipeline.py \
  --cards data/input/cards/cards.jsonl \
  --skip-analyze \
  --skip-cleanup

# Resume interrupted pipeline
python full_pipeline.py \
  --cards data/input/cards/cards.jsonl \
  --resume
```

### Output Structure

When run without `--output-dir`, creates:
```
data/output/
├── raw/          # From generate.py
├── analyzed/     # From analyze.py
├── cleaned/      # From cleanup.py
└── final/        # From rewrite.py
```

---

## Configuration Precedence

Settings are applied in this order (later overrides earlier):

1. Default values in `config/settings.yaml`
2. Environment variables (e.g., `OPENROUTER_API_KEY`)
3. CLI arguments (e.g., `--model`, `--provider`)

---

## Checkpointing

All scripts support checkpointing for resumable execution:

- Use `--resume` to continue from where you left off
- Use `--no-checkpoint` to disable checkpointing
- Checkpoints are stored in `data/cache/<stage>_checkpoint.json`
- Each script tracks: items processed, successful, failed, last item ID

Example:
```bash
# Start a long generation
python generate.py --cards data/cards.jsonl --batch-size 100

# If interrupted, resume later
python generate.py --cards data/cards.jsonl --resume
```

---

## Parallel Execution

For better throughput, you can run scripts in parallel across different card batches or model providers:

```bash
# Run two generation processes with different models
python generate.py --cards batch1.jsonl --model model-a --output output-a &
python generate.py --cards batch2.jsonl --model model-b --output output-b &

# Process different stages in parallel (if you have enough GPU/quota)
python generate.py --cards cards.jsonl --output output/raw &
python analyze.py --input output/raw --output output/analyzed &
```

---

## Error Handling

- Failed items are logged and checkpointed
- Use `--log-level DEBUG` for more detailed error information
- Check `data/logs/pipeline.log` for structured logs

---

## Performance Tips

1. **Batch size**: Larger batches = fewer API calls but more memory usage
2. **Caching**: Enable caching to avoid re-processing the same inputs
3. **Checkpointing**: Always use `--resume` for long-running jobs
4. **Model selection**: Faster models for generation, higher-quality for judging
5. **Parallelism**: Run multiple instances with different `--output` directories
