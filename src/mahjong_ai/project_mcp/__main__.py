"""Run the repository reader over stdio; the tunnel owns this child process."""
import argparse
from pathlib import Path

from .server import create_server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--audit-path", type=Path)
    args = parser.parse_args()
    root = args.workspace_root.expanduser().resolve(strict=True)
    if not (root / ".git").exists():
        parser.error("--workspace-root must be a Git working tree")
    audit = args.audit_path or root / ".local/project_mcp/audit.jsonl"
    create_server(root, audit_path=audit).run(transport="stdio")


if __name__ == "__main__":
    main()
