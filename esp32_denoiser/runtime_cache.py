"""Content fingerprints for process-local compiled C runtime caches."""
from hashlib import sha256
from pathlib import Path


def runtime_fingerprint(source: Path, names: tuple[str, ...], wrapper: str = "") -> str:
    """Include every compiled source/header and any generated wrapper text.

    Re-read content before each cache lookup so editing a shared header cannot
    silently retain an older library in a long-running evaluation process.
    """
    digest = sha256(wrapper.encode("utf-8"))
    for name in sorted(names):
        content = (source / name).read_bytes()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(content)
    return digest.hexdigest()
