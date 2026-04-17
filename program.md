# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch-materials/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch-materials/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
   - `findings.md` — accumulated knowledge from prior sessions (if it exists). Read this first.
   - `guidance.md` — human steering notes (if it exists). Follow any directions here.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.
- Update `findings.md` — you must keep this file current (see Knowledge Management below).

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, and training constants (time budget, max epochs, etc).
- Overfit models such as Neural Networks by setting the number of epochs to high values. Observe loss and use regularization for such approaches and reduce number of epochs accordingly to avoid overfitting.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_model` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val_mae.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_mae gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_mae improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_mae improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_mae:          47.597133
val_r2:           -29.542863
val_rmse:         48.254829
training_seconds: 1.8
total_seconds:    2.6
peak_vram_mb:     0.0
num_steps:        1300
num_params:       102,401
num_epochs:       100
```

Note that the script is configured to always stop after 5 minutes, so depending on the computing platform of this computer the numbers might look different. You can extract the key metric from the log file:

```
grep "^val_mae:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 7 columns:

```
commit	val_mae  val_r2   val_rmse	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_mae achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_mae	memory_gb	status	description
a1b2c3d	0.997900	44.0	keep	baseline
b2c3d4e	0.993200	44.2	keep	increase LR to 0.04
c3d4e5f	1.005000	44.0	discard	switch to GeLU activation
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

## Knowledge management

You maintain two files to preserve knowledge across experiments and sessions:

### `findings.md` — what you know (required)

After every 5 experiments (or after any significant discovery), update `findings.md`. This is the compressed knowledge that future sessions will read instead of parsing hundreds of results.tsv entries. Keep it under 200 lines. Structure:

```markdown
## Current best
val_mae: X.XXXXXX (iter N, commit abc1234)
Key config: depth=N, dim=N, batch=N, LR=X, ...

## What works
- [finding]: [evidence from which iter(s)]

## What doesn't work
- [thing tried]: [why it failed, iter(s)]

## Structural findings
- [architectural insight]: [evidence]

## Unexplored directions
- [idea]: [why it might work]
```

Commit `findings.md` to the branch alongside successful experiments. This file is your institutional memory — without it, the next session will waste GPU time re-discovering what you already know.

### `guidance.md` — human steering (optional, read-only for you)

If this file exists, read it before planning each experiment. It contains directions from the human operator:
- Research goals or priorities
- Constraints or preferences
- Papers, ideas, or reference implementations to consider
- Explicit instructions to change strategy

You must follow guidance.md directions. You must NOT modify this file — it is the human's communication channel to you.

## Stagnation detection

Track your improvement trajectory. You are in a **stagnation plateau** when:
- The last 10+ kept experiments each improved val_mae by less than 0.1% relative
- OR the last 10+ experiments were all discarded/crashed

When you detect stagnation, **switch modes**:

1. **Stop parameter tuning.** Small LR/WD/batch adjustments will not break through a structural ceiling.
2. **Write a stagnation note** in findings.md: document the current ceiling, what parameter space you've exhausted, and why you believe the architecture is the bottleneck.
3. **Try structural changes**: fundamentally different architectures, not incremental tweaks. Examples:
   - Different attention mechanism (linear attention, local+global hybrid)
   - Different nonlinearity (GELU, SwiGLU, etc.)
   - Different normalization (LayerNorm, DeepNorm)
   - Different residual topology (DenseNet-style, U-Net skip connections)
   - Significantly different model shape (much deeper+narrower, or shallower+wider)
   - Different optimizer (Lion, Sophia, schedule-free Adam)
4. **Rewind if needed.** If structural exploration leads to 3+ consecutive crashes or regressions, rewind to the last known good commit and try a *different* structural direction. Do not keep hammering the same failing approach.

This prevents the depth-first search trap: you explore one direction deeply, but when it plateaus, you step back and try a fundamentally different path with the knowledge you've accumulated.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

LOOP FOREVER:

1. **Plan** (guidance step):
   - Read `guidance.md` if it exists — follow any human directions.
   - Read `findings.md` if it exists — review what's known.
   - Check for stagnation (see above).
   - Decide: parameter tuning or structural exploration? Write a one-line rationale.
2. Tune `train.py` with the chosen experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_mae:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If val_mae improved (lower), you "advance" the branch, keeping the git commit.
9. If val_mae is equal or worse, you git reset back to where you started.
10. Every 5 experiments: update `findings.md` and commit it.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate.

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
