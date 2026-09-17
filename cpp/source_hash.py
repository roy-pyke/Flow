"""Hash the actual source/build recipe, in deterministic repository order."""
from hashlib import sha256
from pathlib import Path


def source_hash(root: Path | None = None) -> str:
    root = Path(__file__).resolve().parent if root is None else root
    paths = sorted([root / "CMakeLists.txt", root / "pyproject.toml", root / "source_hash.py", *root.glob("src/*")])
    payload = "".join(f"{path.relative_to(root).as_posix()}:{sha256(path.read_bytes()).hexdigest()}\n" for path in paths if path.is_file())
    return sha256(payload.encode()).hexdigest()


if __name__ == "__main__":
    print(source_hash())
