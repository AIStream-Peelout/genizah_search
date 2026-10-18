"""Upload the image fine-tune package (dataset + exported v22b vision tower) to private HF repos and tag the dataset.

Same authentication as ``push_dataset.py``: ``HF1_TOKEN`` from the sibling repo's ``.env``, passed to the client and
never printed. The dataset commit is tagged (``v1`` …) because the image notebook pins ``DATA_REVISION``.
"""

import argparse
from pathlib import Path

from dotenv import dotenv_values
from huggingface_hub import HfApi

from embed_utils import AUDIT_ROOT


def main() -> None:
    """Create the repos if needed, upload both folders, tag the dataset."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", default="v1", help="dataset tag the notebook's DATA_REVISION pins")
    parser.add_argument("--dataset-dir", default=str(AUDIT_ROOT / "hf_image" / "dataset_v1"))
    parser.add_argument("--dataset-repo", default="isaacmg/genizah-image-train")
    parser.add_argument("--tower-dir", default=str(AUDIT_ROOT / "hf_image" / "tower_heb-v22b-step1200"))
    parser.add_argument("--tower-repo", default="isaacmg/qwen3-vl-8b-heb-v22b-vision-tower")
    parser.add_argument("--skip-tower", action="store_true")
    args = parser.parse_args()
    token = dotenv_values(Path.home() / "Documents/GitHub/historical-document-analysis/.env").get("HF1_TOKEN")
    api = HfApi(token=token)
    api.create_repo(args.dataset_repo, repo_type="dataset", private=True, exist_ok=True)
    info = api.upload_folder(folder_path=args.dataset_dir, repo_id=args.dataset_repo, repo_type="dataset",
                             ignore_patterns=["*.sizes.json"],
                             commit_message=f"genizah image embedder training data {args.tag}")
    api.create_tag(args.dataset_repo, tag=args.tag, repo_type="dataset", revision=info.oid, exist_ok=True)
    print(f"uploaded {args.dataset_dir} -> {args.dataset_repo}@{info.oid} (tag {args.tag})", flush=True)
    if not args.skip_tower:
        api.create_repo(args.tower_repo, repo_type="model", private=True, exist_ok=True)
        info = api.upload_folder(folder_path=args.tower_dir, repo_id=args.tower_repo, repo_type="model",
                                 commit_message="Qwen3-VL-8B vision tower from the heb-v22b-step1200 checkpoint")
        print(f"uploaded {args.tower_dir} -> {args.tower_repo}@{info.oid}", flush=True)


if __name__ == "__main__":
    main()
