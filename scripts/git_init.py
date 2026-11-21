"""
Initializes and commits files to a local git repository using pure Python (dulwich).
No git.exe installation required.
"""
import os
import sys
from pathlib import Path
from dulwich.repo import Repo
from dulwich import porcelain

repo_path = Path(__file__).resolve().parent.parent

print(f"Target repository: {repo_path}")

# Initialize repo
try:
    repo = Repo(str(repo_path))
    print("[OK] Existing Git repository opened.")
except Exception:
    repo = Repo.init(str(repo_path))
    print("[OK] Initialized new Git repository with Dulwich.")

# Read gitignore patterns
ignored_patterns = [
    "__pycache__", ".pth", ".whl", ".tar.gz", ".zip", "dataset_raw", ".git", ".log"
]

def should_ignore(p: Path):
    s = str(p)
    return any(ign in s for ign in ignored_patterns)

# Add files
files_to_add = []
for root, dirs, files in os.walk(repo_path):
    if ".git" in dirs:
        dirs.remove(".git")
    for file in files:
        full_p = Path(root) / file
        rel_p = full_p.relative_to(repo_path).as_posix()
        if not should_ignore(full_p):
            files_to_add.append(rel_p)

print(f"Staging {len(files_to_add)} files...")
porcelain.add(str(repo_path), paths=files_to_add)

# Commit
try:
    commit_id = porcelain.commit(
        str(repo_path),
        message=b"Initial commit: DentalVision AI Streamlit & FastAPI application",
        author=b"DentalVision AI <dental@vision.ai>",
        committer=b"DentalVision AI <dental@vision.ai>"
    )
    print(f"[OK] Committed successfully: {commit_id.decode()[:8]}")
except Exception as e:
    print(f"Commit note: {e}")
