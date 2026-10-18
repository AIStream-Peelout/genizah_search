"""Upload a packaged dataset folder to the private HF dataset repo and tag the commit.

Authenticates with ``HF1_TOKEN`` from the sibling repo's ``.env`` (the token its eval chain already uses, per
Isaac 2026-10-04); the value is passed to the client and never printed.
The tag (``v1``, ``v2`` …) is what the Colab notebook's ``DATA_REVISION`` pins, so every trained model can be
traced to the exact data it saw.
"""

import argparse
from pathlib import Path

from dotenv import dotenv_values
from huggingface_hub import HfApi

from embed_utils import AUDIT_ROOT


def main() -> None:
    """Create the repo if needed, upload, tag."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True, help="packaged folder under hf_dataset/ and the tag to create")
    parser.add_argument("--repo", default="isaacmg/genizah-embed-train")
    parser.add_argument("--keep-stale", action="store_true",
                        help="keep remote files that are not in this package (default: the snapshot mirrors the package)")
    args = parser.parse_args()
    folder = AUDIT_ROOT / "hf_dataset" / args.name
    token = dotenv_values(Path.home() / "Documents/GitHub/historical-document-analysis/.env").get("HF1_TOKEN")
    api = HfApi(token=token)
    api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
    # delete_patterns="*": files of an older package that this one doesn't contain are removed in the same commit,
    # so the tagged snapshot holds exactly this package (older tags still point at their own snapshots)
    info = api.upload_folder(folder_path=str(folder), repo_id=args.repo, repo_type="dataset",
                             delete_patterns=None if args.keep_stale else "*",
                             commit_message=f"genizah embedder training data {args.name}")
    # never silently keep an existing tag on an older commit: a re-push of a tag name must be explicit
    existing = {t.name for t in api.list_repo_refs(args.repo, repo_type="dataset").tags}
    if args.name in existing:
        raise SystemExit(f"tag {args.name} already exists on {args.repo}; delete it first or use a new name")
    api.create_tag(args.repo, tag=args.name, repo_type="dataset", revision=info.oid)
    print(f"uploaded {folder} -> {args.repo}@{info.oid} (tag {args.name})")


if __name__ == "__main__":
    main()
