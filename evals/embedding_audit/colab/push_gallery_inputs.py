"""Upload ``<AUDIT_ROOT>/hf_gallery`` to the private dataset repo and tag it (the notebook pins ``inputs-v1``).

Same authentication as ``push_image_dataset.py``: ``HF1_TOKEN`` from the sibling repo's ``.env``, passed to the client
and never printed. Every Hugging Face cache is pointed at the NAS first (the Studio's internal disk is nearly full).
Not run by the packaging step; run it by hand when ready (see ``hf_gallery_UPLOAD.txt``).
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from embed_utils import AUDIT_ROOT  # noqa: E402


def main() -> None:
    """Create the repo if needed, upload the package, tag the commit."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", default=str(AUDIT_ROOT / "hf_gallery"))
    parser.add_argument("--repo", default="isaacmg/genizah-image-gallery")
    parser.add_argument("--tag", default="inputs-v1")
    args = parser.parse_args()
    os.environ["HF_HOME"] = str(AUDIT_ROOT / "hf")
    os.environ["HF_XET_CACHE"] = str(AUDIT_ROOT / "hf" / "xet")
    from dotenv import dotenv_values
    from huggingface_hub import HfApi

    token = dotenv_values(Path.home() / "Documents/GitHub/historical-document-analysis/.env").get("HF1_TOKEN")
    api = HfApi(token=token)
    api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
    info = api.upload_folder(folder_path=args.dir, repo_id=args.repo, repo_type="dataset",
                             commit_message=f"gallery embedding inputs {args.tag}")
    api.create_tag(args.repo, tag=args.tag, repo_type="dataset", revision=info.oid, exist_ok=True)
    print(f"uploaded {args.dir} -> {args.repo}@{info.oid} (tag {args.tag})", flush=True)


if __name__ == "__main__":
    main()
