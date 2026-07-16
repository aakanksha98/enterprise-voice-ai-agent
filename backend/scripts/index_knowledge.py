import argparse
from pathlib import Path

from backend.app.rag.ingestion import index_knowledge_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index semantic business knowledge records into Neon pgvector.",
    )
    parser.add_argument("path", type=Path, help="Path to a JSON knowledge file")
    arguments = parser.parse_args()

    indexed_count = index_knowledge_file(arguments.path)
    print(f"Indexed {indexed_count} knowledge records.")


if __name__ == "__main__":
    main()
