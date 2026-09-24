import torch
from lerobot.configs import PreTrainedConfig
from lerobot.policies.factory import get_policy_class

# 1. Point to the 'pretrained_model' folder inside your checkpoint
checkpoint_path = "/home/qte9489/personal_abhi/Thesis-Docs/Reward_Func/reward_func_ws/RLRewardResearchWS/GeneralistRewardModels/lerobot/outputs/train/policy_bc_PickAndInsertCubeStation1_merged_absolute/checkpoints/020000/pretrained_model"

print("Loading policy...")
# 2. Load the policy class and weights from the checkpoint
policy_cfg = PreTrainedConfig.from_pretrained(checkpoint_path)
policy_cls = get_policy_class(policy_cfg.type)
policy = policy_cls.from_pretrained(checkpoint_path, config=policy_cfg)
policy.eval()

# 3. Dynamically create dummy inputs based on the policy's configuration
# LeRobot stores expected input feature shapes in config.input_features
dummy_input = {}
for key, feature in policy.config.input_features.items():
    # Add a batch dimension of 1 (e.g., [1, 3, 224, 224] for images)
    dummy_input[key] = torch.randn(1, *feature.shape, device=policy.config.device)

print(f"Constructed dummy inputs for: {list(dummy_input.keys())}")
print("Exporting to ONNX (this might take a minute for large models)...")

# 4. Export to ONNX
# Use a wrapper so ONNX sees named tensor inputs instead of a Python dict.
class PolicyForOnnx(torch.nn.Module):
    def __init__(self, wrapped_policy: torch.nn.Module, input_keys: list[str]):
        super().__init__()
        self.wrapped_policy = wrapped_policy
        self.input_keys = input_keys

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        batch = {k: v for k, v in zip(self.input_keys, inputs, strict=True)}
        return self.wrapped_policy.predict_action_chunk(batch)


input_keys = list(dummy_input.keys())
onnx_module = PolicyForOnnx(policy, input_keys)

torch.onnx.export(
    onnx_module,
    tuple(dummy_input[k] for k in input_keys),
    "lerobot_bc_policy.onnx",
    opset_version=17,  # Use a modern opset to handle complex transformer/CNN layers
    input_names=input_keys,
    output_names=["action_chunk"],
)

print("Success! You can now upload 'lerobot_bc_policy.onnx' to Netron.")