

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


OUTFMT = "6 qseqid sseqid pident length evalue bitscore qstart qend sstart send qlen slen"


def fasta_records(path):
    
    identifier, parts = None, []
    with Path(path).open() as handle:
        for line in handle:
            if line.startswith(">"):
                if identifier is not None:
                    yield identifier, "".join(parts)
                header = line[1:].split()
                if not header:
                    raise ValueError(f"Empty FASTA header: {path}")
                identifier, parts = header[0], []
            elif line.strip():
                if identifier is None:
                    raise ValueError(f"Sequence before FASTA header: {path}")
                parts.append(line.strip())
    if identifier is not None:
        yield identifier, "".join(parts)


def summarize_alignments(path, expected_queries, subject_lengths):
    
    pairs = defaultdict(lambda: [0, 0])
    with Path(path).open() as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            fields = line.rstrip().split("\t")
            if len(fields) != 12:
                raise ValueError(f"{path}:{number}: expected 12 BLAST fields")
            query, subject = fields[:2]
            if query not in expected_queries or subject not in subject_lengths:
                raise ValueError(f"Unknown query/subject at {path}:{number}")
            aligned, query_length, subject_length = map(int, (fields[3], fields[10], fields[11]))
            if aligned <= 0 or query_length != int(expected_queries[query]["length_bp"]):
                raise ValueError(f"Invalid query/HSP length at {path}:{number}")
            if subject_length != subject_lengths[subject]:
                raise ValueError(f"Invalid subject length at {path}:{number}")
            pairs[(query, subject)][0] += aligned
            pairs[(query, subject)][1] += 1
    details = defaultdict(list)
    for (query, subject), (aligned, hsps) in sorted(pairs.items()):
        details[query].append({
            "sseqid": subject,
            "subject_length": subject_lengths[subject],
            "summed_hsp_alignment_length": aligned,
            "hsp_count": hsps,
            "identity": aligned / subject_lengths[subject],
        })
    summaries = []
    for query, metadata in expected_queries.items():
        hits = details[query]
        values = [hit["identity"] for hit in hits]
        summaries.append({
            **metadata,
            "query_id": query,
            "matched_subjects": len(hits),
            "similarity_score": sum(values),
            "subjects_identity_gt_0_4": sum(value > 0.4 for value in values),
            "subjects_identity_gt_0_9": sum(value > 0.9 for value in values),
        })
    return summaries, details


def write_table(path, rows, delimiter="\t"):
    rows = list(rows)
    if not rows:
        raise ValueError(f"Cannot write a headerless empty table: {path}")
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)
