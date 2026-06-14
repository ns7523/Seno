from pathlib import Path


SENSITIVE_PATTERNS = (
    ".env",
    "*.db",
    "*.sqlite",
    "*.sqlite3",
    "*.zip",
    "token.json",
    "client_secret*.json",
    "*secret*.json",
)
EXCLUDED_DIRS = {".git", "venv", ".venv", "__pycache__", ".pytest_cache"}


def main() -> None:
    matches: list[Path] = []
    for pattern in SENSITIVE_PATTERNS:
        matches.extend(Path(".").glob(pattern))
        matches.extend(Path(".").glob(f"**/{pattern}"))
    unique = sorted({path for path in matches if path.is_file() and not (EXCLUDED_DIRS & set(path.parts))})
    if not unique:
        print("No potentially sensitive files found.")
        return
    print("Potentially sensitive files found. Review before committing or uploading:")
    for path in unique:
        print(f"- {path}")
    print("No files were deleted.")


if __name__ == "__main__":
    main()
