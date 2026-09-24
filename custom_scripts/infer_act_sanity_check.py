"""Sanity-check a trained ACT checkpoint against real ground-truth data.

Loads a checkpoint (policy weights + its OWN saved normalizer/unnormalizer
stats - see `pretrained_model/policy_preprocessor_step_3_normalizer_processor
.safetensors` / `policy_postprocessor_step_0_unnormalizer_processor.safetensors`,
NOT the dataset's meta/stats.json - the checkpoint is self-contained), then
either:

  - (default) picks ONE random window from a random episode, runs one
    forward pass to get a predicted action CHUNK, and compares it directly
    against the actual recorded ground-truth actions for that same window.
  - (`--full-episode`) treats EVERY frame in the chosen episode as an
    "anchor", exactly like training's own data loading does (each dataset
    index is a valid anchor whose target is the next `chunk_size` actions,
    modulo padding at episode boundaries) - runs a forward pass FOR EVERY
    anchor (batched, `--batch-size` at a time) and aggregates the error
    over the whole episode: mean absolute error per prediction-horizon step
    (t=0..chunk_size-1, averaged over ALL anchors) and per action dimension
    (averaged over ALL anchors AND all horizon steps) - a much more robust
    picture than a single random sample, at the cost of `ep_len - chunk_size
    + 1` forward passes (batched, so still fast).

IMPORTANT: `policy.predict_action_chunk()` returns the model's raw
NORMALIZED output - it does NOT unnormalize. This script explicitly applies
the saved `postprocessor` before comparing against the dataset's raw
(real-unit) actions - skipping this step is the #1 way this kind of check
goes wrong (comparing normalized model output against raw meters/radians
looks like total garbage even for a correctly-trained model).

DELTA-ACTION CHECKPOINTS (`--action-space delta_joint`, e.g. trained on
`trossen_real/scripts/convert_to_delta_joint_dataset.py` output - see that
repo's `PickAndInsertCube.md`): the MAE comparison logic below needs NO
changes either way - `gt_actions` is read directly from `episode_dataset`,
which for a delta-trained run already contains delta-valued actions on
disk (the conversion happened BEFORE this dataset was ever written), so
comparing predicted vs. GT stays an apples-to-apples comparison in
whichever space the dataset is actually in. `--action-space` here is purely
informational: it prints the GT action's mean/std for the sampled window
so you can visually confirm you're looking at the space you expect (delta
joints should be near-zero-mean and small-std; gripper stays
absolute/large either way).

Run (from the repo root, inside this repo's own venv - NOT the
residual-offpolicy-rl workspace's deps/lerobot, which is v2.1-only and
can't read this v3-format dataset):

    # single random anchor (original behavior)
    uv run python custom_scripts/infer_act_sanity_check.py \\
        --checkpoint outputs/train/policy_bc_PickAndInsertCube/checkpoints/050000/pretrained_model \\
        --dataset-root /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/trossen_real/datasets/20260805_150824_and_153134_153727_PickAndInsertCube \\
        --dataset-repo-id 20260805_150824_and_153134_153727_PickAndInsertCube

    # every frame in one episode as an anchor, aggregated
    uv run python custom_scripts/infer_act_sanity_check.py \\
        --checkpoint outputs/train/policy_bc_PickAndInsertCube/checkpoints/050000/pretrained_model \\
        --dataset-root /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/trossen_real/datasets/20260805_150824_and_153134_153727_PickAndInsertCube \\
        --dataset-repo-id 20260805_150824_and_153134_153727_PickAndInsertCube \\
        --full-episode --episode 49
"""

import argparse
import random

import huggingface_hub
import torch

# Bulletproof offline patch (same as custom_scripts/v3_dataset_inspection.py) -
# without this, LeRobotDataset can try to hit the HF Hub for metadata even
# when both repo_id and root point at a fully-local dataset.
_LOCAL_DATASET_ROOT = None


def _patch_offline(dataset_root: str) -> None:
    global _LOCAL_DATASET_ROOT
    _LOCAL_DATASET_ROOT = dataset_root
    huggingface_hub.snapshot_download = lambda *a, **kw: kw.get("local_dir", _LOCAL_DATASET_ROOT) or _LOCAL_DATASET_ROOT


def _dim_names(action_dim: int, action_space: str = "absolute") -> list[str]:
    if action_dim == 7:
        if action_space == "delta_joint":
            return [f"d_joint_{i}" for i in range(6)] + ["gripper"]
        return [f"joint_{i}" for i in range(6)] + ["gripper"]
    return [f"dim_{i}" for i in range(action_dim)]


def _print_gt_sanity_note(gt_actions: torch.Tensor, dim_names: list[str], action_space: str) -> None:
    """Print the sampled window's GT action mean/std per dim - a quick visual
    check that you're actually looking at the space you think you are (see
    module docstring's DELTA-ACTION CHECKPOINTS section)."""
    flat = gt_actions.reshape(-1, gt_actions.shape[-1])
    mean = flat.mean(dim=0).tolist()
    std = flat.std(dim=0).tolist()
    print(f"\nGT action stats over this sampled window (action_space={action_space}):")
    for name, m, s in zip(dim_names, mean, std):
        print(f"  {name:>10}: mean={m:+.4f}  std={s:.4f}")
    if action_space == "delta_joint":
        joint_means = [abs(m) for m in mean[:6]]
        if max(joint_means) > 0.2:
            print("  WARNING: expected near-zero-mean joint dims for a delta_joint checkpoint, but the largest "
                  "|mean| above is > 0.2 - double check --action-space / that this dataset was actually delta-converted.")


def _predict_chunk_batch(
    policy, preprocessor, postprocessor, episode_dataset, anchors: list[int],
    non_image_state_keys: list[str], image_keys: list[str], chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one batched forward pass for a list of anchor frame indices.

    Returns (predicted_real, gt_actions), both shape (len(anchors), chunk_size, action_dim).
    """
    model_input = {}
    for key in non_image_state_keys + image_keys:
        model_input[key] = torch.stack([episode_dataset[a][key] for a in anchors])  # (B, ...)

    gt_actions = torch.stack(
        [torch.stack([episode_dataset[a + t]["action"] for t in range(chunk_size)]) for a in anchors]
    )  # (B, chunk_size, action_dim)

    with torch.no_grad():
        model_input = preprocessor(model_input)
        predicted_normalized = policy.predict_action_chunk(model_input)  # (B, chunk_size, action_dim), STILL normalized
        predicted_real = postprocessor(predicted_normalized).cpu()  # NOW in real units

    assert predicted_real.shape == gt_actions.shape, f"{predicted_real.shape} vs {gt_actions.shape}"
    return predicted_real, gt_actions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Path to a checkpoint's pretrained_model/ dir")
    parser.add_argument("--dataset-root", required=True, help="Local path to the (v3) dataset used for training")
    parser.add_argument("--dataset-repo-id", required=True, help="repo_id to pass to LeRobotDataset (must match info.json)")
    parser.add_argument("--episode", type=int, default=None, help="Episode index to sample from (default: random)")
    parser.add_argument("--start-frame", type=int, default=None, help="(single-anchor mode only) start frame within the episode (default: random, chunk_size-safe)")
    parser.add_argument("--full-episode", action="store_true", help="Treat every valid frame in the episode as an anchor and aggregate error, instead of a single random window")
    parser.add_argument("--batch-size", type=int, default=32, help="(--full-episode only) anchors per forward pass batch (default 32)")
    parser.add_argument("--verbose", action="store_true", help="(--full-episode only) also print each anchor's own overall MAE")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for episode/frame selection (default: unset)")
    parser.add_argument("--device", default=None, help="Override device (default: whatever the checkpoint's config says)")
    parser.add_argument(
        "--action-space", choices=["absolute", "delta_joint"], default="absolute",
        help="Purely informational (see module docstring) - just changes dimension labels and the printed "
             "GT-stats sanity note. 'delta_joint' expects joint dims near-zero-mean/small-std, gripper unaffected.",
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    _patch_offline(args.dataset_root)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    print(f"Loading policy from checkpoint: {args.checkpoint}")
    policy = ACTPolicy.from_pretrained(args.checkpoint)
    if args.device:
        policy.config.device = args.device
    policy.to(policy.config.device)
    policy.eval()

    print("Loading preprocessor/postprocessor FROM THE CHECKPOINT ITSELF "
          "(saved normalizer/unnormalizer stats - NOT the dataset's meta/stats.json).")
    preprocessor, postprocessor = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=args.checkpoint)

    input_keys = list(policy.config.input_features.keys())
    image_keys = [k for k in input_keys if k.startswith("observation.images.")]
    non_image_state_keys = [k for k in input_keys if k not in image_keys]
    print(f"Model input_features ({len(input_keys)} total): {input_keys}")
    print(f"  -> non-image state keys used as input: {non_image_state_keys}")
    print(f"  -> image keys used as input: {image_keys}")

    chunk_size = policy.config.chunk_size

    # First load the full dataset (metadata only cost) just to know episode count/lengths.
    full_dataset = LeRobotDataset(repo_id=args.dataset_repo_id, root=args.dataset_root)
    n_episodes = full_dataset.num_episodes
    episode_idx = args.episode if args.episode is not None else random.randint(0, n_episodes - 1)
    print(f"Dataset: {n_episodes} episodes, {full_dataset.num_frames} frames total, fps={full_dataset.fps}")
    print(f"Sampling from episode {episode_idx}")

    episode_dataset = LeRobotDataset(repo_id=args.dataset_repo_id, root=args.dataset_root, episodes=[episode_idx])
    ep_len = len(episode_dataset)
    if ep_len <= chunk_size:
        raise ValueError(f"Episode {episode_idx} only has {ep_len} frames, need > chunk_size={chunk_size}")

    max_start = ep_len - chunk_size  # last anchor whose full chunk_size-step GT chunk stays inside the episode
    dim_names = None

    if not args.full_episode:
        # --- Original single-anchor behavior ---
        start_frame = args.start_frame if args.start_frame is not None else random.randint(0, max_start - 1)
        if not (0 <= start_frame <= max_start):
            raise ValueError(f"--start-frame {start_frame} out of range [0, {max_start}] for episode {episode_idx} ({ep_len} frames)")
        print(f"Start frame: {start_frame} (episode has {ep_len} frames, chunk_size={chunk_size})")

        predicted_real, gt_actions = _predict_chunk_batch(
            policy, preprocessor, postprocessor, episode_dataset, [start_frame],
            non_image_state_keys, image_keys, chunk_size,
        )
        err = (predicted_real[0] - gt_actions[0]).abs()  # (chunk_size, action_dim)
        action_dim = err.shape[1]
        dim_names = _dim_names(action_dim, args.action_space)
        _print_gt_sanity_note(gt_actions[0], dim_names, args.action_space)

        print(f"\n{'step':>4} | " + " | ".join(f"{n:>10}" for n in dim_names))
        for t in range(chunk_size):
            print(f"{t:>4} | " + " | ".join(f"{err[t, d].item():>10.4f}" for d in range(action_dim)))

        mae_per_dim = err.mean(dim=0)
        print(f"\nMean absolute error per dimension over the {chunk_size}-step chunk:")
        for name, val in zip(dim_names, mae_per_dim.tolist()):
            print(f"  {name:>10}: {val:.4f}")
        print(f"\nOverall mean absolute error: {err.mean().item():.4f}")
        print(f"First-step MAE (should be smallest - least extrapolation): {err[0].mean().item():.4f}")
        print(f"Last-step MAE (expect this to be largest - most extrapolation, no intermediate real obs seen): {err[-1].mean().item():.4f}")
        return

    # --- Full-episode mode: every valid frame index is an anchor, exactly like training's own data loading ---
    anchors = list(range(0, max_start + 1))
    print(f"\n--full-episode: {len(anchors)} anchors (frame 0..{max_start}) in episode {episode_idx}, "
          f"batch_size={args.batch_size}, chunk_size={chunk_size} -> {len(anchors)} forward passes total, batched.")

    err_sum_per_step_dim = None  # (chunk_size, action_dim), accumulated sum over all anchors
    per_anchor_mae: list[float] = []

    for batch_start in range(0, len(anchors), args.batch_size):
        batch_anchors = anchors[batch_start : batch_start + args.batch_size]
        predicted_real, gt_actions = _predict_chunk_batch(
            policy, preprocessor, postprocessor, episode_dataset, batch_anchors,
            non_image_state_keys, image_keys, chunk_size,
        )
        err = (predicted_real - gt_actions).abs()  # (B, chunk_size, action_dim)

        if err_sum_per_step_dim is None:
            action_dim = err.shape[-1]
            dim_names = _dim_names(action_dim, args.action_space)
            err_sum_per_step_dim = torch.zeros(chunk_size, action_dim)
            _print_gt_sanity_note(gt_actions, dim_names, args.action_space)

        err_sum_per_step_dim += err.sum(dim=0)
        per_anchor_mae.extend(err.mean(dim=(1, 2)).tolist())

        if args.verbose:
            for anchor, mae in zip(batch_anchors, err.mean(dim=(1, 2)).tolist()):
                print(f"  anchor {anchor:>4}: overall MAE = {mae:.4f}")

        done = batch_start + len(batch_anchors)
        print(f"  processed {done}/{len(anchors)} anchors...", end="\r")

    print()  # newline after the \r progress line

    mae_per_step_dim = err_sum_per_step_dim / len(anchors)  # (chunk_size, action_dim)
    mae_per_step = mae_per_step_dim.mean(dim=1)  # (chunk_size,), averaged over dims - shows error-vs-horizon-depth
    mae_per_dim = mae_per_step_dim.mean(dim=0)  # (action_dim,), averaged over horizon steps

    print(f"\nMean absolute error per prediction-horizon step, averaged over ALL {len(anchors)} anchors "
          f"(and over all {len(dim_names)} action dims):")
    for t, val in enumerate(mae_per_step.tolist()):
        print(f"  step {t:>3}: {val:.4f}")

    print(f"\nMean absolute error per dimension, averaged over ALL {len(anchors)} anchors and all {chunk_size} horizon steps:")
    for name, val in zip(dim_names, mae_per_dim.tolist()):
        print(f"  {name:>10}: {val:.4f}")

    print(f"\nOverall mean absolute error (all anchors, all horizon steps, all dims): {mae_per_step_dim.mean().item():.4f}")
    print(f"First-step MAE (t=0, least extrapolation): {mae_per_step[0].item():.4f}")
    print(f"Last-step MAE (t={chunk_size - 1}, most extrapolation): {mae_per_step[-1].item():.4f}")
    per_anchor_mae_t = torch.tensor(per_anchor_mae)
    print(f"\nPer-anchor overall MAE across the episode: mean={per_anchor_mae_t.mean().item():.4f}, "
          f"std={per_anchor_mae_t.std().item():.4f}, max={per_anchor_mae_t.max().item():.4f} "
          f"(at anchor {anchors[per_anchor_mae_t.argmax().item()]}), min={per_anchor_mae_t.min().item():.4f}")


if __name__ == "__main__":
    main()

