import os
import huggingface_hub

# 1. Define your exact paths and names
LOCAL_DATASET_PATH = "/home/qte9489/personal_abhi/Thesis-Docs/Reward_Func/reward_func_ws/RLRewardResearchWS/DatasetUtil/merged_7stage"
ACTUAL_REPO_ID = "merged_7stage" 

# 2. The Bulletproof Offline Patch
# We intercept the underlying Hugging Face downloader. If LeRobot tries to trigger a 
# network request for any missing file, this instantly feeds it your local directory instead.
huggingface_hub.snapshot_download = lambda *args, **kwargs: kwargs.get("local_dir", LOCAL_DATASET_PATH) or LOCAL_DATASET_PATH

from lerobot.datasets.lerobot_dataset import LeRobotDataset

def main():
    print(f"Attempting to load strictly local v3 dataset from: {LOCAL_DATASET_PATH}...")

    try:
        # Both the repo_id and the root must be exactly correct so it trusts the local info.json
        dataset = LeRobotDataset(repo_id=ACTUAL_REPO_ID, root=LOCAL_DATASET_PATH)
        
        print("\n✅ Dataset loaded successfully!")
        
        print(f"\n📊 --- Dataset Statistics ---")
        print(f"Total Episodes : {dataset.num_episodes}")
        print(f"Total Frames   : {dataset.num_frames}")
        print(f"FPS            : {dataset.fps}")

        print(f"\n🔑 --- Data Features & Columns ---")
        print("Schema Features:")
        for key, feature in dataset.features.items():
            print(f"  - {key}: {feature}")

    except Exception as e:
        print(f"\n❌ Error loading dataset: {e}")

if __name__ == "__main__":
    main()