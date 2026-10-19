import argparse
import json
from pathlib import Path

from fasta_io import ROOT, header_fields, protect, read_fasta, write_record


ACGT = set("ACGT")
EVO1_PROMPT = "|r__Duplodnaviria;k__"


def phagen_spans(token_ids, bos, eos, base_by_id, pad):
    start = None
    index = 0
    for position, token in enumerate(token_ids):
        if start is None:
            if token == bos:
                start = position
            continue
        if token == bos:
            start = position
        elif token == pad:
            start = None
        elif token == eos:
            content = token_ids[start + 1 : position]
            if content and not (set(content) - base_by_id.keys()):
                yield (
                    index,
                    start,
                    position,
                    "".join(base_by_id[token] for token in content),
                )
                index += 1
            start = None


def extract_phagen(path, tokenizer, minimum):
    base_by_id = {tokenizer.convert_tokens_to_ids(base): base for base in "ACGT"}
    for header, text in read_fasta(path):
        fields = header_fields(header)
        sequence_index = int(fields["sequence_index"])
        token_ids = tokenizer(text, add_special_tokens=False).input_ids
        for index, start, end, sequence in phagen_spans(
            token_ids,
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
            base_by_id,
            tokenizer.pad_token_id,
        ):
            if len(sequence) >= minimum:
                yield (
                    f"seq{sequence_index}_span{index}",
                    sequence,
                    {
                        "sequence_index": sequence_index,
                        "prompt_index": int(fields.get("prompt_index", 0)),
                        "return_index": int(fields.get("return_index", 0)),
                        "span_index": index,
                        "prompt": fields.get("prompt", ""),
                        "prompt_with_bos": fields.get("prompt_with_bos", ""),
                        "start": start,
                        "end": end,
                        "length": end - start + 1,
                        "content_length": len(sequence),
                        "num_pad_inside": 0,
                    },
                )


def extract_megadna(path, minimum, mode="closed"):
    for raw_index, (header, text) in enumerate(read_fasta(path)):
        sequence_index = header_fields(header).get("sequence_index", raw_index)
        index = 0
        candidates = (
            text.split("#")[:-1]
            if mode in {"closed", "closed_clean"}
            else text.split("#")
        )
        for candidate in candidates:
            sequence = candidate.upper()
            if mode in {"all_clean", "closed_clean"}:
                sequence = "".join(base for base in sequence if base in ACGT)
            if not sequence or set(sequence) - ACGT:
                continue
            if len(sequence) >= minimum:
                yield (
                    f"run{sequence_index}_ext{index}",
                    sequence,
                    {
                        "sequence_index": sequence_index,
                        "span_index": index,
                        "content_length": len(sequence),
                    },
                )
            index += 1


def split_evo1(raw, prompt=EVO1_PROMPT):
    for index, document in enumerate((prompt + raw).split(" |")):
        document = document[1:] if document.startswith("|") else document
        taxonomy, sequence = (
            document.split("|", 1) if "|" in document else ("", document)
        )
        if (
            sequence
            and not (set(sequence) - ACGT)
            and taxonomy.split(";")[0] == "r__Duplodnaviria"
        ):
            yield index, taxonomy, sequence


def extract_evo(path, model, minimum, prompt=EVO1_PROMPT):
    with Path(path).open() as handle:
        for line in handle:
            record = json.loads(line)
            sequence_index, seed = record["sequence_index"], record.get("seed", "NA")
            if model == "evo1":
                for index, taxonomy, sequence in split_evo1(record["raw"], prompt):
                    if len(sequence) >= minimum:
                        taxonomy = (
                            ("|" + taxonomy + "|")
                            .replace(" ", "")
                            .replace("\t", "")
                            .replace("\n", "")
                        )
                        yield (
                            f"evo1_{sequence_index}_{index}",
                            sequence,
                            {
                                "sequence_index": sequence_index,
                                "doc_index": index,
                                "seed": seed,
                                "tax": taxonomy,
                                "length": len(sequence),
                            },
                        )
            else:
                sequence = record["raw"]
                if len(sequence) >= minimum and not (set(sequence) - ACGT):
                    yield (
                        f"evo2_{sequence_index}",
                        sequence,
                        {
                            "sequence_index": sequence_index,
                            "seed": seed,
                            "chunk": record.get("chunk", "NA"),
                            "length": len(sequence),
                        },
                    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=["phagen", "megadna", "evo1", "evo2"], required=True
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        help="Local PhaGen tokenizer directory; unused for other models.",
    )
    parser.add_argument("--min-length", type=int, default=2000)
    parser.add_argument(
        "--megadna-mode",
        choices=["closed", "closed_clean", "all_clean"],
        default="closed",
        help="Production: closed. S4 configurations specify their recorded extraction mode.",
    )
    parser.add_argument("--prompt", default=EVO1_PROMPT, help="Evo1 taxonomy prefix.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.min_length < 1:
        parser.error("--min-length must be positive.")
    output = args.output or ROOT / "data/cohorts" / f"{args.model}.fasta"
    if args.input.resolve() == output.resolve():
        parser.error("Input and output must differ.")
    protect([output], args.force)
    if args.model == "phagen":
        if not args.checkpoint:
            parser.error("--checkpoint is required for PhaGen tokenization.")
        from phagen import load_tokenizer

        records = extract_phagen(
            args.input,
            load_tokenizer(args.checkpoint, local_files_only=True),
            args.min_length,
        )
    elif args.model == "megadna":
        records = extract_megadna(args.input, args.min_length, args.megadna_mode)
    else:
        records = extract_evo(args.input, args.model, args.min_length, args.prompt)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w") as handle:
        for sequence_id, sequence, fields in records:
            write_record(handle, sequence_id, sequence, fields)
            count += 1
    print(f"Wrote {count} contigs to {output}")


if __name__ == "__main__":
    main()
