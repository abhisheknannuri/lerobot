"""Standalone ACT policy inference server - HTTP, stdlib only (no Flask/FastAPI
dependency needed in this venv).

WHY A SEPARATE SERVER: this checkpoint (`ACTPolicy` + its exact processor/
config classes) can only be loaded correctly from THIS repo's `lerobot`
package/venv - but live camera capture (`pyrealsense2`) is NOT installed
here, and the robot arm's own HTTP client
(`trossen_real.teleop.follower_client.FollowerClient`) lives in the OTHER
(`residual-offpolicy-rl`) workspace's package. Rather than trying to force
both venvs' dependencies into one process, this mirrors the SAME "one
server, adapt the client" pattern already used for
`follower_single_server.py`: this process owns ONLY the policy (loads once,
stays warm), and exposes a tiny HTTP API. The actual live control loop
(`trossen_real/scripts/live_infer_deploy.py`, in the OTHER workspace) owns
the arm connection + cameras and just POSTs observations here, getting back
a ready-to-send real-unit action - no cross-venv imports anywhere.

Endpoints:
    GET  /health   -> {"loaded": true, "checkpoint": ..., "chunk_size":25, "n_action_steps":20, ...}
    POST /reset    -> clears the policy's internal action-chunk queue (call
                       once at the start of a new rollout/episode - otherwise
                       a stale chunk from a previous rollout could still be
                       queued up and get executed first).
    POST /predict  -> body: {
                         "observation.state": [7 floats], ... (one key per
                         non-image entry in the checkpoint's input_features),
                         "observation.images.<cam_name>": {
                             "shape": [H, W, 3], "dtype": "uint8",
                             "data_b64": "<base64 of the raw HWC uint8 bytes>"
                         }, ... (one key per image entry in input_features)
                       }
                       response: {"action": [7 floats]}  - REAL units
                       (radians/meters, matching command_space: joint's
                       native representation) - already unnormalized, ready
                       to send directly to `follower_single_server.py`'s
                       `/move_to_joint_positions`.

`/predict` internally calls `policy.select_action()` (NOT
`predict_action_chunk()`) - this is the standard ACT receding-horizon
deployment API: it manages the chunk-of-`chunk_size`/execute-
`n_action_steps` bookkeeping internally (a deque), only actually running a
new forward pass (and only looking at the CURRENT observation you just
POSTed) once every `n_action_steps` calls - the other calls just pop the
next already-planned action off the queue, ignoring the observation you
sent that tick (matches the paper's/this repo's own intended usage - see
`ACTPolicy.select_action()`'s docstring/implementation). So: call
`/predict` EVERY control tick regardless (cheap when queued), not just
every `n_action_steps` ticks - the server handles when to actually re-plan.

CHUNK-ANCHOR-RELATIVE ACTIONS (`ACTConfig.use_relative_actions=True`, see
configuration_act.py) + `n_action_steps > 1`: correctly supported, and the
resulting `"action"` is ALREADY fully reconstructed to absolute units by
this server (via the postprocessor's `AbsoluteActionsProcessorStep`) -
callers should treat it exactly like an absolute-action checkpoint, no
client-side math needed. Every action in a chunk is reconstructed against
the ONE state that was live when that chunk was actually inferred (a
genuine `select_action()` refill tick, i.e. `policy._action_queue` was
empty), not whatever the current tick's state happens to be - see the
`_PolicyState.relative_step`/`cached_anchor_state` bookkeeping in
`_predict()`. Verified empirically 2026-09-02 against a real trained
checkpoint: fed a realistic joint state, the returned `"action"` closely
tracked that same state's magnitude (not a near-zero delta), both on the
genuine refill tick and a subsequent cached tick using the same anchor.

An OLDER checkpoint trained WITHOUT `use_relative_actions` (static
per-frame delta baked into the DATASET itself, e.g. via
`convert_to_delta_joint_dataset.py`) is a DIFFERENT case: this server does
NOT reconstruct anything for it (`state.relative_step` is None, no
postprocessor step ever adds a state back) - it returns the raw predicted
delta as-is, and the CLIENT is responsible for reconstruction (see
`policy_client.py::reconstruct_absolute_action()`'s `"delta_joint"` branch).
`n_action_steps > 1` is ALSO fine for this case (CORRECTED 2026-09-02 - a
previous version of this comment wrongly claimed it required 1): the
dataset's per-frame target was `raw_action[i] - state[i]` for every row
`i` independently (see that script), so a chunk position `k` popped `k`
ticks after inference needs `predicted[k] + real_state_at_this_tick` - as
long as the CLIENT uses the FRESHLY MEASURED state at the tick it actually
executes that action (not the stale state from when the chunk was
inferred), the reconstruction is correct at every chunk position, not just
0. `trossen_real`'s `infer_loop.py` already does this (fetches
`follower_state` fresh every tick, before reconstruction) - confirmed by
reading that code, not assumed. `n_action_steps` is therefore a pure
compute/reactivity tradeoff for EVERY checkpoint type here (fewer forward
passes vs. less responsive to real-time disturbances), not a correctness
constraint.

Run (from this repo's root, inside its own venv):

    uv run python custom_scripts/policy_server.py \\
        --checkpoint outputs/train/policy_bc_PickAndInsertCube/checkpoints/050000/pretrained_model \\
        --port 5070
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


class _PolicyState:
    def __init__(self) -> None:
        self.policy = None
        self.preprocessor = None
        self.postprocessor = None
        self.checkpoint_path: str | None = None
        self.non_image_keys: list[str] = []
        self.image_keys: list[str] = []
        # Heuristic guess only (see _detect_likely_action_space()) - the model
        # itself has NO field recording whether it was trained on absolute or
        # delta_joint actions (that's purely a property of the dataset it saw,
        # invisible to ACTConfig) - this exists so callers can cross-check
        # their own --action-space choice against it and get a loud warning on
        # mismatch, instead of silently sending delta output through as if it
        # were absolute (or vice versa) - both are dangerous on real hardware.
        self.likely_action_space: str | None = None
        self.action_space_source: str | None = None
        # `dataset_to_policy_features()` classifies EVERY `observation.*`
        # column in the training dataset's schema as a STATE-typed
        # `input_features` entry, regardless of whether the loaded POLICY
        # actually reads it. Confirmed by reading `configs/policies.py`'s
        # `robot_state_feature`/`image_features` properties AND both
        # `modeling_act.py`'s and `modeling_diffusion.py`'s forward/
        # select_action code directly (not assumed, checked for BOTH policy
        # types this server can load - see `_load_policy_class()`): the ONLY
        # keys either one ever actually reads from the batch are the literal
        # `observation.state` key and the VISUAL-typed image keys - every
        # other `input_features` entry (e.g. `observation.joint_pos_raw`,
        # `observation.intervened`, `observation.policy_action`, ...) is
        # normalized by the preprocessor but then NEVER READ by the model
        # itself - provably dead weight, not a real dependency. This set is
        # computed once at load time so `_predict()` can zero-fill any of
        # THOSE keys a client didn't send, while still hard-erroring if the
        # one real key (`observation.state`) or an image key is missing.
        # (If this server is ever extended to load a policy type that reads
        # MORE of input_features than this - e.g. `env_state_feature` - this
        # assumption must be re-checked for that policy specifically; it is
        # NOT a general lerobot property.)
        self.dead_weight_keys: set[str] = set()
        self._warned_dead_weight_keys: set[str] = set()
        # --- Chunk-anchor-relative action bookkeeping (see _predict()) -----
        # Reference to the preprocessor's RelativeActionsProcessorStep, if the
        # loaded checkpoint's ACTConfig.use_relative_actions=True (None for
        # any older/absolute/static-delta checkpoint - the fix below is then
        # a complete no-op, preserving existing behavior exactly).
        self.relative_step = None
        # The state (as RelativeActionsProcessorStep caches it internally) as
        # of the tick that most recently triggered a genuine chunk refill -
        # NOT necessarily this tick's state. See _predict().
        self.cached_anchor_state = None


state = _PolicyState()


def _detect_action_space_from_stats(preprocessor) -> str | None:
    """PRIMARY, checkpoint-self-contained heuristic: reads the checkpoint's
    OWN saved normalizer stats for the `action` feature (baked into the
    checkpoint itself - no dependency on any external dataset path being
    reachable from wherever the server happens to run, unlike the repo_id
    string match below). Real numbers confirmed 2026-09-02 on this project's
    checkpoints: a delta_joint action's per-dim mean/std cluster tightly
    near 0 (e.g. max|mean|~0.02, max std~0.04 on the ForIter1_v3_deltaJoint
    checkpoint) because `action[i] = raw_action[i] - state[i]`; an absolute
    joint action's mean reflects real joint angles and is NOT centered near
    zero (e.g. max|mean|~1.8 rad on the non-delta merged checkpoint - a
    shoulder/elbow joint sitting away from 0 rad most of the time). A
    threshold well inside that ~90x gap is robust. Returns None if the
    `action` key isn't in the loaded stats at all (shouldn't normally
    happen, but don't crash startup over this diagnostic feature)."""
    from lerobot.processor import NormalizerProcessorStep

    norm_step = next((s for s in preprocessor.steps if isinstance(s, NormalizerProcessorStep)), None)
    if norm_step is None or "action" not in getattr(norm_step, "stats", {}):
        return None
    mean = np.asarray(norm_step.stats["action"]["mean"], dtype=np.float64)
    max_abs_mean = float(np.max(np.abs(mean)))
    return "delta_joint" if max_abs_mean < 0.5 else "absolute"


def _detect_action_space_from_repo_id(checkpoint: str) -> str | None:
    """FALLBACK heuristic only (used when the stats-based one above can't
    run): reads `train_config.json` (saved alongside every checkpoint) and
    checks whether its `dataset.repo_id` contains the literal substring
    "delta" (case-insensitive) - matches this project's OLDER naming
    convention (`..._deltajoint`/`..._deltaJoint`), but is fragile: a
    dataset like `PickAndInsertCube_TS1_ForIter1_v3` IS delta_joint action
    space but has no "delta" in its name at all - confirmed to
    misclassify as "absolute" on exactly that checkpoint, 2026-09-02.
    Returns None if `train_config.json` is missing/unreadable or has no
    repo_id - callers should treat `None` as "unknown", not "absolute".
    """
    train_config_path = Path(checkpoint) / "train_config.json"
    try:
        with open(train_config_path) as f:
            train_config = json.load(f)
        repo_id = str(train_config.get("dataset", {}).get("repo_id", ""))
    except Exception:
        return None
    if not repo_id:
        return None
    return "delta_joint" if "delta" in repo_id.lower() else "absolute"


def _detect_likely_action_space(checkpoint: str, preprocessor, action_space_override: str | None) -> tuple[str | None, str]:
    """Returns (likely_action_space, source) - source is one of
    "explicit_cli_override", "checkpoint_action_stats", "dataset_repo_id_string"
    "unknown" - always log `source` alongside the value so it's clear how
    much to trust it. An explicit `--action-space` always wins outright."""
    if action_space_override is not None:
        return action_space_override, "explicit_cli_override"
    from_stats = _detect_action_space_from_stats(preprocessor)
    if from_stats is not None:
        return from_stats, "checkpoint_action_stats"
    from_repo_id = _detect_action_space_from_repo_id(checkpoint)
    if from_repo_id is not None:
        logger.warning(
            "Falling back to the fragile dataset.repo_id string-match heuristic for action_space (could not "
            "read action stats from the checkpoint itself) - verify this is right, or pass --action-space explicitly."
        )
        return from_repo_id, "dataset_repo_id_string"
    return None, "unknown"


def _load_policy_class(checkpoint: str):
    """Reads the checkpoint's OWN saved `config.json` `"type"` field (e.g.
    "act", "diffusion") and returns the matching policy class via
    `lerobot.policies.factory.get_policy_class()` - the same registry
    `make_policy()` itself uses. Confirmed 2026-09-03: this server used to
    hardcode `ACTPolicy` unconditionally, which would silently misload (or
    outright fail to load) a Diffusion checkpoint's real architecture."""
    from lerobot.policies.factory import get_policy_class

    config_path = Path(checkpoint) / "config.json"
    with open(config_path) as f:
        policy_type = json.load(f)["type"]
    logger.info("Checkpoint's own config.json says policy type=%r - loading that class.", policy_type)
    return get_policy_class(policy_type)


def _load_policy(checkpoint: str, device_override: str | None, n_action_steps_override: int | None,
                  action_space_override: str | None = None) -> None:
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.processor import RelativeActionsProcessorStep

    policy_class = _load_policy_class(checkpoint)
    logger.info("Loading policy from checkpoint: %s", checkpoint)
    policy = policy_class.from_pretrained(checkpoint)
    if device_override:
        policy.config.device = device_override
    if n_action_steps_override is not None:
        # n_action_steps is a PURE inference-time parameter (controls how many
        # steps of each predict_action_chunk() call select_action()'s internal
        # deque serves before re-running the model - see ACTPolicy.reset()/
        # select_action()) - it does NOT affect the saved weights or how the
        # model was trained, so there's no need to retrain to change it.
        # Safe to DECREASE post-hoc (confirmed against modeling_act.py): the
        # queue's maxlen is cached from config.n_action_steps at __init__/
        # reset() time, but the actual number of actions pushed per refill is
        # `predict_action_chunk(...)[:, :self.config.n_action_steps]`, read
        # FRESH every call - so mutating this attribute BEFORE the first
        # .reset()/select_action() call works correctly. Only INCREASING
        # n_action_steps above what a previous .reset() already cached as
        # maxlen would silently truncate - not a concern here since this
        # override happens immediately after from_pretrained(), before reset.
        if n_action_steps_override > policy.config.n_action_steps:
            logger.warning(
                "n_action_steps_override=%d is LARGER than the checkpoint's trained value (%d) - this is fine "
                "right now (override happens before the first reset()/select_action() call), but do not try to "
                "increase it again later via /reset alone.",
                n_action_steps_override, policy.config.n_action_steps,
            )
        policy.config.n_action_steps = n_action_steps_override
    # DiffusionConfig has a STRICTER constraint than ACT's plain
    # `n_action_steps <= chunk_size`: `n_action_steps <= horizon - n_obs_steps + 1`
    # (see DiffusionPolicy.select_action()'s docstring) - `horizon`/`n_obs_steps`
    # only exist on Diffusion-family configs, so this check is a no-op for ACT.
    max_n_action_steps = getattr(policy.config, "horizon", None)
    n_obs_steps = getattr(policy.config, "n_obs_steps", None)
    if max_n_action_steps is not None and n_obs_steps is not None:
        limit = max_n_action_steps - n_obs_steps + 1
        if policy.config.n_action_steps > limit:
            logger.warning(
                "n_action_steps=%d exceeds this Diffusion checkpoint's valid range (horizon=%d - "
                "n_obs_steps=%d + 1 = %d) - select_action() may behave incorrectly. Pass a smaller "
                "--n-action-steps.", policy.config.n_action_steps, max_n_action_steps, n_obs_steps, limit,
            )
    policy.to(policy.config.device)
    policy.eval()
    policy.reset()

    # NOTE: the preprocessor/postprocessor pipelines loaded via `pretrained_path`
    # bake in whatever device was SAVED at train time (e.g. "cuda") in their own
    # JSON, independent of `policy.config.device` above - overriding just the
    # policy's device is NOT enough on a machine where that saved device isn't
    # what you want (e.g. forcing --device cpu here, or running on a machine
    # with a different GPU setup than the one used for training): the
    # preprocessor would still move inputs to the SAVED device, causing a
    # "tensors on different devices" error the moment the model (now on a
    # DIFFERENT device) tries to use them. Explicitly override the device
    # processor step in BOTH pipelines to match wherever the policy actually
    # ended up living (matches the exact override pattern `lerobot_train.py`
    # itself uses when resuming training from a checkpoint on a new device).
    logger.info("Loading preprocessor/postprocessor FROM THE CHECKPOINT ITSELF (saved normalizer/unnormalizer stats).")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": policy.config.device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )

    from lerobot.utils.constants import OBS_STATE

    input_keys = list(policy.config.input_features.keys())
    image_keys = [k for k in input_keys if k.startswith("observation.images.")]
    non_image_keys = [k for k in input_keys if k not in image_keys]
    # See _PolicyState.dead_weight_keys's comment - everything except the
    # literal OBS_STATE key is never read by ACTPolicy's forward pass.
    dead_weight_keys = {k for k in non_image_keys if k != OBS_STATE}

    state.policy = policy
    state.preprocessor = preprocessor
    state.postprocessor = postprocessor
    state.checkpoint_path = checkpoint
    state.non_image_keys = non_image_keys
    state.image_keys = image_keys
    state.dead_weight_keys = dead_weight_keys
    state.likely_action_space, state.action_space_source = _detect_likely_action_space(
        checkpoint, preprocessor, action_space_override
    )
    state.relative_step = next(
        (s for s in preprocessor.steps if isinstance(s, RelativeActionsProcessorStep)), None
    )
    # Override with the AUTHORITATIVE signal, unconditionally, after the
    # heuristic above: a use_relative_actions=True checkpoint's "action" is
    # ALREADY reconstructed to real absolute units by this server (see the
    # module docstring's CHUNK-ANCHOR-RELATIVE ACTIONS section) - so
    # "absolute" isn't just a workaround label here, it's what a caller
    # should genuinely treat it as. Without this, both the stats-based and
    # repo_id-string heuristics can mislabel it "delta_joint" (a relative
    # checkpoint's saved action stats are ALSO small/near-zero, same as the
    # OLD static per-frame delta convention, since the dataset's stats.json
    # was recomputed to reflect relative deltas before training - confirmed
    # 2026-09-02) - a client trusting that label would wrongly re-add the
    # state on top of an already-absolute value, roughly doubling the real
    # target. This override doesn't change any WEBUI/client-side handling:
    # the client already treats "absolute" as pass-through, which is
    # exactly correct for this case too.
    #
    # Applied UNCONDITIONALLY, even over an explicit --action-space
    # override: unlike the other two heuristics (which are genuine guesses
    # that explicit user knowledge should be allowed to override),
    # relative_step.enabled reflects the ACTUAL loaded model config, not a
    # guess - an explicit `--action-space delta_joint` against a checkpoint
    # that's really relative-reconstructed would be a real user mistake
    # (the exact double-application bug this exists to prevent), so it's
    # not honored here.
    if state.relative_step is not None and state.relative_step.enabled:
        if action_space_override is not None and action_space_override != "absolute":
            logger.warning(
                "--action-space=%s was given, but this checkpoint has use_relative_actions=True and its "
                "'action' is ALREADY reconstructed to absolute units by this server - overriding to 'absolute' "
                "regardless, to avoid a client double-applying the state offset on top of an already-absolute "
                "value.", action_space_override,
            )
        state.likely_action_space = "absolute"
        state.action_space_source = "relative_action_reconstructed_to_absolute"
    state.cached_anchor_state = None
    state._warned_dead_weight_keys = set()

    logger.info("Loaded. non-image state keys: %s", non_image_keys)
    if dead_weight_keys:
        logger.info(
            "Of those, these are DEAD WEIGHT for this policy type (declared in input_features but never "
            "read by its forward pass - see _PolicyState.dead_weight_keys) - a client omitting them will "
            "be auto-zero-filled, with no effect on the model's output: %s", sorted(dead_weight_keys),
        )
    logger.info("Loaded. image keys: %s", image_keys)
    logger.info("chunk_size=%d, n_action_steps=%d, device=%s", policy.config.chunk_size, policy.config.n_action_steps, policy.config.device)
    logger.info("likely_action_space=%s (source=%s)", state.likely_action_space, state.action_space_source)
    _relative_active = state.relative_step is not None and state.relative_step.enabled
    logger.info("use_relative_actions=%s (chunk-anchor-relative reconstruction active=%s)", _relative_active, _relative_active)
    # NOTE (corrected 2026-09-02): n_action_steps > 1 does NOT require
    # use_relative_actions=True for correctness - see the module docstring's
    # CHUNK-ANCHOR-RELATIVE ACTIONS section for the full derivation. A
    # previous version of this function warned otherwise; that warning was
    # wrong and has been removed.


def _decode_image(payload: dict) -> torch.Tensor:
    """{"shape":[H,W,3], "dtype":"uint8", "data_b64": ...} -> (C,H,W) float32 tensor in [0,1] -
    exactly matching `video_utils.py`'s own `permute(2,0,1)` + `/255` convention, so live camera
    frames are formatted IDENTICALLY to how LeRobotDataset itself would have loaded them."""
    raw = base64.b64decode(payload["data_b64"])
    arr = np.frombuffer(raw, dtype=np.dtype(payload["dtype"])).reshape(payload["shape"])
    return torch.from_numpy(arr.copy()).permute(2, 0, 1).contiguous().float() / 255.0


def _action_queue_len(policy) -> int | None:
    """Length of `policy`'s internal action-chunk queue, checked BEFORE this
    tick's `select_action()` call - 0 means this call will be a genuine
    refill (fresh model inference), >0 means it'll just pop an
    already-planned step. Returns None if the policy has neither queue shape
    known here (e.g. a temporal-ensembling policy with no queue concept at
    all) - callers should treat that as "not applicable", not "refill".

    Policy-type-aware: ACTPolicy keeps a top-level `_action_queue` deque
    (see modeling_act.py). DiffusionPolicy keeps `_queues`, a dict of deques
    keyed by feature name, with the action queue at `_queues["action"]` (see
    modeling_diffusion.py's `reset()`/`select_action()`) - confirmed by
    reading that code directly, not assumed."""
    action_queue = getattr(policy, "_action_queue", None)
    if action_queue is not None:
        return len(action_queue)
    queues = getattr(policy, "_queues", None)
    if isinstance(queues, dict) and "action" in queues:
        return len(queues["action"])
    return None


def _dead_weight_placeholder(key: str) -> list[float]:
    """Zero-filled placeholder for a key in `state.dead_weight_keys` (see
    that field's comment - provably never read by ACTPolicy's forward pass,
    so the exact value here has NO effect on the model's output). Shape
    comes from the checkpoint's own declared `input_features[key].shape`,
    not a hardcoded guess. Logs a ONE-TIME warning per key so a missing
    key is never silently invisible."""
    if key not in state._warned_dead_weight_keys:
        logger.warning(
            "Client did not send '%s' - it's dead weight for this policy type (see "
            "_PolicyState.dead_weight_keys), zero-filling with no effect on the output.", key,
        )
        state._warned_dead_weight_keys.add(key)
    shape = state.policy.config.input_features[key].shape
    return [0.0] * shape[0]


def _predict(obs: dict) -> dict:
    model_input = {}
    for key in state.non_image_keys:
        if key not in obs:
            if key not in state.dead_weight_keys:
                raise KeyError(key)  # a REAL input (observation.state) is missing - fail loudly, as before
            model_input[key] = torch.tensor(_dead_weight_placeholder(key), dtype=torch.float32).unsqueeze(0)
            continue
        model_input[key] = torch.tensor(obs[key], dtype=torch.float32).unsqueeze(0)  # (1, 7)
    for key in state.image_keys:
        model_input[key] = _decode_image(obs[key]).unsqueeze(0)  # (1, 3, H, W)

    with torch.no_grad():
        model_input = state.preprocessor(model_input)

        # --- Chunk-anchor-relative anchor-state fix (only active when the
        # loaded checkpoint has use_relative_actions=True) ------------------
        # The preprocessor call above just made RelativeActionsProcessorStep
        # cache THIS tick's state into `_last_state` unconditionally - it has
        # no notion of select_action()'s internal n_action_steps queue. That's
        # only correct on a tick that's actually about to trigger a fresh
        # chunk inference (queue empty); on a tick that's just popping an
        # already-planned step from a PREVIOUS chunk, the postprocessor's
        # AbsoluteActionsProcessorStep must keep reconstructing against the
        # state that was live when THAT chunk was inferred, not this tick's.
        # temporal-ensembling policies (config.temporal_ensemble_coeff is not
        # None) have no action queue at all - every tick is a genuine fresh
        # inference there already, so this block is skipped entirely for them.
        # POLICY-TYPE-AWARE (added 2026-09-03 for Diffusion support): ACT keeps
        # its queue at the top-level `_action_queue` attribute; DiffusionPolicy
        # keeps it at `_queues["action"]` (a dict of deques) instead - checked
        # directly against modeling_diffusion.py's `reset()`/`select_action()`,
        # not assumed. `_action_queue_len()` below returns None (skip this
        # block) if neither shape is found.
        action_queue_len = _action_queue_len(state.policy)
        if state.relative_step is not None and state.relative_step.enabled and action_queue_len is not None:
            is_refill_tick = action_queue_len == 0
            if is_refill_tick:
                state.cached_anchor_state = state.relative_step.get_cached_state()
            elif state.cached_anchor_state is not None:
                state.relative_step._last_state = state.cached_anchor_state

        action_normalized = state.policy.select_action(model_input)  # (1, action_dim), STILL normalized
        action_real = state.postprocessor(action_normalized)  # NOW real units

    return {
        # Real units (radians/meters) - already unnormalized, ready to send directly (or reconstruct
        # a delta against, for delta_joint checkpoints) - this is the field every existing caller uses.
        "action": action_real.squeeze(0).cpu().tolist(),
        # DIAGNOSTIC ONLY - the model's raw output BEFORE the postprocessor's unnormalize step (still
        # z-scored). Not meant to be used for control - purely so a caller can log/inspect what the
        # network itself actually predicted, independent of the unnormalization math, e.g. to check
        # whether a "gripper not closing" symptom traces back to the model's own raw prediction (never
        # confidently predicting "closed") vs. something going wrong in unnormalization/reconstruction
        # downstream.
        "action_normalized": action_normalized.squeeze(0).cpu().tolist(),
    }


class _Handler(BaseHTTPRequestHandler):
    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 (stdlib method name)
        if self.path == "/health":
            self._send_json({
                "loaded": state.policy is not None,
                "checkpoint": state.checkpoint_path,
                "non_image_keys": state.non_image_keys,
                "dead_weight_keys": sorted(state.dead_weight_keys),
                "image_keys": state.image_keys,
                "chunk_size": state.policy.config.chunk_size if state.policy else None,
                "n_action_steps": state.policy.config.n_action_steps if state.policy else None,
                "device": str(state.policy.config.device) if state.policy else None,
                "likely_action_space": state.likely_action_space,
                "action_space_source": state.action_space_source,
            })
        else:
            self._send_json({"error": f"unknown GET path {self.path}"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/reset":
            if state.policy is not None:
                state.policy.reset()
            # Not strictly required for correctness (the queue is now empty,
            # so the very next /predict tick is naturally a refill tick and
            # will re-cache a fresh anchor - see _predict()) but explicit is
            # safer than relying on that invariant silently holding.
            state.cached_anchor_state = None
            self._send_json({"status": "ok"})
            return
        if self.path == "/predict":
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                result = _predict(body)
                self._send_json(result)
            except Exception as exc:
                logger.exception("predict failed")
                self._send_json({"error": str(exc)}, status=500)
            return
        self._send_json({"error": f"unknown POST path {self.path}"}, status=404)

    def log_message(self, fmt, *args) -> None:  # quieter default stdlib access log
        logger.info("%s - %s", self.address_string(), fmt % args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Path to a checkpoint's pretrained_model/ dir")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5070)
    parser.add_argument("--device", default=None, help="Override device (default: whatever the checkpoint's config says)")
    parser.add_argument(
        "--action-space", choices=["absolute", "delta_joint"], default=None,
        help="Explicitly tell the server which action space this checkpoint was trained on, instead of "
             "relying on auto-detection (checked against the checkpoint's own saved action stats first, "
             "falling back to a fragile dataset.repo_id string match - see _detect_likely_action_space()). "
             "Use this whenever you already know, or when the checkpoint's dataset name doesn't follow the "
             "'...delta...' naming convention (confirmed to misdetect on PickAndInsertCube_TS1_ForIter1_v3, "
             "which IS delta_joint despite having no 'delta' in its name).",
    )
    parser.add_argument(
        "--n-action-steps", type=int, default=None,
        help="Override n_action_steps (default: whatever the checkpoint was trained with). Pure inference-time "
             "parameter, safe to decrease OR increase (up to chunk_size) post-hoc without retraining - see "
             "_load_policy(). Values up to chunk_size are correctly handled for ANY checkpoint type (relative, "
             "static-per-frame-delta, or absolute) as long as the CLIENT reconstructing the final action uses "
             "the FRESHLY MEASURED state at the tick it actually executes each chunk position, not a stale "
             "chunk-inference-time state - see the module docstring's CHUNK-ANCHOR-RELATIVE ACTIONS section for "
             "the derivation (corrected 2026-09-02 - an earlier version of this help text wrongly claimed 1 was "
             "required for non-relative checkpoints). This is a compute/reactivity tradeoff, not a correctness one.",
    )
    args = parser.parse_args()

    _load_policy(args.checkpoint, args.device, args.n_action_steps, args.action_space)

    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    logger.info("Policy server listening on %s:%d", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
