from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def read_fasta(path):
    header, body = None, []
    with Path(path).open() as handle:
        for line in handle:
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(body)
                header, body = line[1:].rstrip("\r\n"), []
            elif line.strip():
                body.append(line.strip())
        if header is not None:
            yield header, "".join(body)


def header_fields(header):
    return dict(part.split("=", 1) for part in header.split()[1:] if "=" in part)


def write_record(handle, sequence_id, sequence, metadata=None):
    description = " ".join(f"{key}={value}" for key, value in (metadata or {}).items())
    handle.write(f">{sequence_id}" + (f" {description}" if description else "") + "\n")
    for start in range(0, len(sequence), 80):
        handle.write(sequence[start : start + 80] + "\n")


def protect(paths, force=False):
    existing = [str(path) for path in paths if Path(path).exists()]
    if existing and not force:
        raise FileExistsError(
            "Output exists; choose another output or use --force: "
            + ", ".join(existing)
        )


def seed_torch(torch, seed):
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
