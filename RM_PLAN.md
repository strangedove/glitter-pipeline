# RP Reward Model — Data & Training Plan (deferred)

Status: **plan only, not implemented** (2026-08-31). Revisit when a wider
model pool is available. Owner: toasty. Written by master session.

## Goal

Train a reward model over **full multi-turn RP scenes** (not single turns)
to score quality for judging and RL-style training. Turn-level rewards can
be distilled later from the same RM by masking; full-scene granularity is
the correct first target.

## Confirmed pair structure

The right unit is a **preference pair**: two full scenes with the
- same characters, scenario, and general flow,
- same turn count and speaker order,
- same core trajectory,

where one is **notably better** than the other on craft axes. This strips
content confounds so the RM's signal isolates quality. The PrefRewriter
stage (post 4d6a24b) produces exactly this shape, with structural
verification. Current generator/rewriter pool (DeepSeek V4 Flash drafts,
Gemma-4-31B-Agares edits) is adequate for **pipeline bring-up only** —
both models will likely be replaced.

## Why the current two-model pool is insufficient

1. **Provenance confound**: if every pair is (DeepSeek draft, Gemma edit),
   the RM learns a style discriminator, not quality. It will downscore
   good prose that resembles either source.
2. **Length bias**: current rewrites compress ~20-30%. An RM trained on
   these learns "shorter = better"; RL against it collapses length.
3. **Margin noise**: some pairs differ in content as well as quality
   (similarity 0.02-0.08 in the first 9-pair batch). These teach content
   preferences, not craft.
4. **Scale**: preference RMs need hundreds of pairs minimum; the first
   batch was 9.

## Data plan

### 1. Source variety (primary fix)
- Multiple **generator** models for rejected/draft sides (any RP-capable
  model; Arli catalog is the easy pool).
- Multiple **rewriter** models for chosen sides (different families, not
  just Gemma).
- Crossover matrix: drafts from A graded against edits by B, drafts from B
  edited by A, etc. The "better half" must not be predictable from
  authorship style.
- Quality filter on sources: drop generators whose drafts fail the
  structural gate (turn-marker format, non-empty) at high rates — the
  current rescue net is a crutch, not an RM signal source.

### 2. Pair gates (ship only pairs with correct signal)
- **Structural verifier** (already in rp-rewrite): exact turn count +
  role sequence match.
- **Similarity floor** (~0.3 sequence ratio): chosen must be recognizably
  the same scene as rejected. Tunable; measure the distribution before
  fixing the threshold.
- **Judge margin gate**: rubric-score both sides on the target axes
  (agency/initiative, forward momentum, emotional shifts, show-don't-tell,
  turn-length variation, tic density); keep pairs only with a decisive
  margin. Store the axis scores in the pair manifest — they double as
  axis labels for later analysis.
- **Length-bias audit**: correlate chosen-margin with word-count delta on
  a held-out slice; if positive, the pair set (or the RM) is length-biased.
  Mix in pairs where the better version is longer.

### 3. Synthetic hard negatives (cheap, controlled, bidirectional)
Deterministic degrader applied to *good* scenes — no LLM needed:
- Re-inject the exact tics the TIC detector finds (lazy similes,
  emotion-telling, pronoun-opener streaks, "said, voice X" tags) at the
  detected injection sites.
- Flatten assistant agency (demote assistant initiative in alternating
  turns).
- Normalize turn lengths toward the mean (kills the "VARY TURN LENGTH"
  signal; also produces longer-chosen pairs, countering length bias).
- Optionally shuffle two adjacent turns (breaks causality/momentum).
Each degradation isolates one axis → pairs where the ONLY difference is
the trained signal, in both directions. Build on `TicRemover`/
`TicDetector` in `core/analysis.py` (injection sites = existing matches).

### 4. Scale target
Start: 300-500 gated pairs. Monitor: RM accuracy on a held-out pair set
and on crossover combinations never seen in training.

## Training notes
- Bradley-Terry pairwise loss on full-scene representations; base model
  should be an RP-strong instruct model (decide pool at revisit).
- Eval protocol: (a) held-out pair accuracy split by source combination —
  if accuracy is high on seen sources and low on unseen, the RM learned
  provenance, not quality; (b) score correlation with judge rubric scores
  on the targeted axes; (c) length-debias check above.
- Keep the pair manifest (source models, axis scores, similarity, margins)
  — it is the debug surface when the RM misbehaves.

## Pipeline work items when revisited
1. Pair-builder stage: raw drafts + rewrites → gated pairs + manifest
   (most gates already exist in rp-rewrite; add sim floor + judge margin).
2. Hard-negative degrader module (deterministic, test-driven).
3. Crossover runner: N generators × M rewriters × k cards, checkpointed.
4. RM training harness + the eval protocol above.

## Reference results from the 9-pair bring-up batch (2026-08-31)
- Tic rate 3.29 → 1.57/1k words (6/9 improved), structure 9/9 exact.
- Turn-length variation collapsed (cv ~0.4 → ~0.2) — the one targeted
  axis the rewriter regresses on; watch for it in future rewriters.
- Similarity spread 0.02-0.49; only sim ≥ ~0.3 pairs look like true
  edit-in-place preference pairs.

## TODO: behavioral rewriter edit-in-place constraints (added 2026-09-02)

Finding: Kimi (K2.6/K3/K2-Instruct-0905) and MiMo behavioral rewrites regenerate
the scene (median prose similarity 0.03-0.06, word counts -47% to -60%) instead
of editing the original draft. Structure survives (verifier) but the
preference-pair premise — "improved version of THE scene" — is weakened: the
chosen side is effectively a fresh draft.

Fix to implement + test on next data-gen pass:
1. Prompt-level: per-turn edit contract in `pref_rewrite_system` — "rewrite the
   scene turn by turn; turn N of your output must correspond to turn N of the
   input; keep at least half of each turn's concrete details; length within
   +/-20% of the original turn" (replace the current whole-scene instructions).
2. Gate-level: add a similarity floor to the rewrite stage (reject rewrites
   below ~0.3 scene similarity alongside the structural verifier).
3. Optional: two-stage rewrite — first pass edits in place; only scenes the
   judge still fails get the free-form regeneration treatment.
4. Test: re-run one Kimi + one Gemma rewrite pass on the same corpus; measure
   similarity distribution, word delta, and judge flag-rate delta before/after.
Note: Gemma's style pass already edits surgically (sim 0.98 on a 3-tic scene);
the behavioral seat is the one that regenerates.
