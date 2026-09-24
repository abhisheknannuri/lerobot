# Chunk-anchor-relative actions for ACT (`use_relative_actions`)

Status as of 2026-08-31: **implemented and unit-verified, NOT yet adopted** -
the currently-deployed checkpoint (`policy_bc_PickAndInsertCubeStation1_merged_deltaJoint`)
still uses the OLD static-per-frame-delta dataset convention
(`convert_to_delta_joint_dataset.py`, in the `residual-offpolicy-rl` repo) and
is unaffected by anything in this doc. This is ready for whenever you decide
to retrain with the new scheme.

## Why this exists

Symptom: with `--n-action-steps 12` on the deltaJoint checkpoint, the gripper
sometimes doesn't close when it should (works better with `n_action_steps=1`,
worse with higher chunk execution - backwards from what you'd expect from
ACT's chunking design). Root cause, verified against the actual code (not
assumed from any comment):

Your current dataset (`convert_to_delta_joint_dataset.py`) computes
**per-frame** deltas: `delta[t] = action[t] - state[t]`, a DIFFERENT
reference state for every target, computed once, statically, at dataset
conversion time. For chunk position `k>0` during inference, the model must
implicitly predict its own future state just to make that frame's delta
target meaningful - an unnecessarily hard, leaky regression target. Worse,
`policy_client.py::reconstruct_absolute_action()` (in `residual-offpolicy-rl`)
reconstructs `absolute[t+k] = predicted[k] + state_measured_right_now`, which
is only correct for `k=0` - for `k>0` it uses the WRONG reference state
(this tick's, not the tick the chunk was inferred at), and the error
compounds across the chunk.

OpenPI's approach (`openpi/src/openpi/transforms.py::DeltaActions`/
`AbsoluteActions`, `openpi/src/openpi/training/data_loader.py`,
`openpi/scripts/compute_norm_stats.py`) fixes this: define the WHOLE chunk's
delta against **one single anchor state** (the observation-time state),
`delta[t+k] = action[t+k] - state[t]` for every `k`. Reconstruction then adds
back that SAME anchor to every step - mathematically well-defined regardless
of chunk position, and normalization stats are computed on this exact
post-transform distribution (`compute_norm_stats.py` applies the same
transform before running `RunningStats`).

This exact mechanism already exists, generically, in THIS lerobot fork -
`src/lerobot/processor/relative_action_processor.py`
(`RelativeActionsProcessorStep`/`AbsoluteActionsProcessorStep`, explicitly
labeled "Mirrors OpenPI's DeltaActions") - wired for pi0/pi0.5/pi0_fast
already, just not for ACT.

## What changed (this repo only - `residual-offpolicy-rl` is untouched)

Backups taken before editing: `*.20260831_<timestamp>.bak` next to each file.

1. **`src/lerobot/policies/act/configuration_act.py`** - added 3 fields,
   copied verbatim from `configuration_pi0.py`:
   ```python
   use_relative_actions: bool = False
   relative_exclude_joints: list[str] = field(default_factory=lambda: ["gripper"])
   action_feature_names: list[str] | None = None
   ```
2. **`src/lerobot/policies/act/processor_act.py`** - `make_act_pre_post_processors()`
   now builds a `RelativeActionsProcessorStep` (inserted after `DeviceProcessorStep`,
   before `NormalizerProcessorStep` in the preprocessor) and a paired
   `AbsoluteActionsProcessorStep` (inserted after `UnnormalizerProcessorStep`,
   before the final `DeviceProcessorStep(cpu)` in the postprocessor) - identical
   ordering to `processor_pi0.py`. Both steps are always constructed but are
   `enabled=config.use_relative_actions` - **default `False` means this is a
   complete no-op for every existing/older checkpoint**, verified:
   ```
   default pre steps: [Rename, AddBatchDim, Device, RelativeActionsProcessorStep(disabled), Normalizer]
   ```
3. **`custom_scripts/policy_server.py`** - fixes a SECOND bug that porting the
   processor alone does not: `RelativeActionsProcessorStep._last_state` gets
   overwritten on every `/predict` call regardless of whether ACT's internal
   `select_action()` queue (`policy._action_queue`, a deque, only exists when
   `temporal_ensemble_coeff is None`) is about to refill or just popping an
   already-planned step. Added: check `len(policy._action_queue)` BEFORE
   calling `select_action()` - if empty (genuine refill), cache this tick's
   state as the new anchor; if not, force `relative_step._last_state` back to
   the cached anchor before the postprocessor runs, so `AbsoluteActionsProcessorStep`
   reconstructs every chunk position against the state that was actually live
   when THAT chunk was inferred. `/reset` also clears the cached anchor
   (belt-and-suspenders; the natural empty-queue-on-next-tick logic already
   handles this, but explicit is safer). Inert no-op when
   `state.relative_step is None or not .enabled` (every existing checkpoint) -
   `_load_policy()` logs a warning if `n_action_steps>1` is requested on such
   a checkpoint, same as before.

## Verification performed (no GPU/real checkpoint needed, no hardware touched)

1. `py_compile` on all 3 files.
2. Numeric round-trip: built a synthetic `(B=2, chunk=5, action_dim=7)` batch,
   ran `RelativeActionsProcessorStep` then `AbsoluteActionsProcessorStep`,
   confirmed exact recovery of the original absolute action at every chunk
   position, confirmed the gripper dim (excluded) is untouched.
2. Confirmed `make_act_pre_post_processors(ACTConfig())` (all defaults) is a
   behavior-preserving no-op vs. before this change.
3. Simulated 7 ticks / 3 chunk refills (`n_action_steps=3`) with a DRIFTING
   fake robot state and the real `RelativeActionsProcessorStep`/
   `AbsoluteActionsProcessorStep` classes wired through a fake queue-based
   policy (mirroring `ACTPolicy.select_action()`'s real deque logic).
   Confirmed every reconstructed action matches its OWN chunk's anchor-state
   formula, and explicitly confirmed it does NOT match what the old
   per-tick-state (buggy) formula would have produced whenever chunk
   position `k>0`.

**Not verified**: an actual training run, an actual real-hardware rollout
with a `use_relative_actions=True` checkpoint. No GPU/robot access from this
session - do a short real test before trusting this for real data collection.

## How to actually use this, when you're ready

### 1. Recompute normalization stats (REQUIRED - stats are NOT interchangeable
between the raw-absolute and relative-chunk distributions)

Run on your ORIGINAL ABSOLUTE-action dataset (NOT the `_deltaJoint` one -
that already has static per-frame deltas baked in; combining that with this
transform would double-delta and be wrong):

```bash
.venv/bin/python -m lerobot.scripts.lerobot_edit_dataset \
    --repo_id <your_absolute_dataset_repo_id_or_local_root> \
    --new_repo_id <name>_relative_stats \
    --operation.type recompute_stats \
    --operation.relative_action true \
    --operation.chunk_size <ACT's chunk_size, e.g. 25 - MUST match what you'll train with> \
    --operation.relative_exclude_joints "['gripper']" \
    --operation.num_workers 4
```
Writes to a NEW repo/root by default (non-destructive). Only add
`--operation.overwrite true` (with `--new_repo_id` == `--repo_id`) if you
deliberately want to modify the original dataset's `stats.json` in place.

### 2. Train ACT with the new flag

Add to your training config/CLI overrides:
```
--policy.use_relative_actions=true
--policy.relative_exclude_joints="['gripper']"
--policy.action_feature_names="[<your 7 action dim names, in order>]"
```
Point `--dataset.repo_id`/`--dataset.root` at the dataset from step 1 (the
one with the relative-action stats, still storing ABSOLUTE actions on disk -
the transform happens at batch-construction time, not on disk).

### 3. Deploy

```bash
.venv/bin/python custom_scripts/policy_server.py \
    --checkpoint <path>/pretrained_model/ --port 5090 --n-action-steps <N>
```
`N > 1` is now safe for a `use_relative_actions=True` checkpoint (the fix in
`policy_server.py` above handles it). Startup logs will confirm:
```
use_relative_actions=True (chunk-anchor-relative reconstruction active=True)
```

### 4. What needs to change on the `residual-offpolicy-rl` side when you adopt this

**Nothing changes today because you're not using this yet.** For the record,
when you do:

- `policy_client.py::PolicyClient.predict_full()`'s returned `"action"` field
  will now already be a fully-reconstructed ABSOLUTE action (server-side,
  correct across all chunk positions) for such a checkpoint - exactly like it
  already is for an `action_space="absolute"` checkpoint today.
- `reconstruct_absolute_action(action_space="delta_joint", ...)` must NOT be
  called for such a checkpoint - calling it would double-apply the state
  addition. The caller needs a way to know which case it's in. `/health`
  currently exposes `likely_action_space` (a heuristic from the dataset
  repo_id's name) but nothing about `use_relative_actions` - the clean fix is
  a new `/health` field, e.g. `"server_reconstructs_absolute": true`, so
  `infer_app`/`live_infer_deploy.py`/`real_residual_env.py`/
  `train_residual_td3.py` can branch on that instead of guessing from the
  checkpoint name. Not added yet (deliberately - avoid touching this repo for
  a feature you said you won't use yet).
- Separately (unrelated to this feature, found while investigating it, see
  below): `train_residual_td3.py`'s offline `base_action` relabeling loop
  never calls `/reset` between dataset frames - with `n_action_steps>1` at
  the server, most relabeling queries silently pop a stale queued action
  from an unrelated frame. This affects the OLD scheme just as much as this
  one and needs its own decision - see
  `residual-offpolicy-rl/trossen_real/ACT_RELATIVE_ACTIONS_AND_OFFLINE_RELABELING.md`.

## Open items / not done

- Bandwidth optimization you raised (fetch a whole chunk once via
  `predict_action_chunk()` directly, execute it client-side across
  `n_action_steps` ticks without re-POSTing images every tick) - good idea,
  separate feature, not built.
- The `/health` field above, for automatic client-side detection.
- Any actual training run / real-hardware validation.
