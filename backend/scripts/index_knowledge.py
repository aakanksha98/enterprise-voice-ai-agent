import argparse
from pathlib import Path

from backend.app.config import load_local_env
from backend.app.rag.ingestion import index_knowledge_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index semantic business knowledge records into Neon pgvector.",
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to a JSON knowledge file or a directory of JSON files",
    )
    arguments = parser.parse_args()

    load_local_env()
    paths = (
        sorted(arguments.path.glob("*.json"))
        if arguments.path.is_dir()
        else [arguments.path]
    )
    indexed_count = sum(index_knowledge_file(path) for path in paths)
    print(f"Indexed {indexed_count} knowledge records from {len(paths)} file(s).")


if __name__ == "__main__":
    main()
