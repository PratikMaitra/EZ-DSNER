"""
dsner_data.py — Flexible format converter for DS-NER methods.

Convert between ANY pair of the data formats used by the 8 DS-NER methods.

=== Supported Formats ===

  conll_bio     : TOKEN BIO_TAG              (2-col CoNLL, used by test/dev)
  conll_dist    : TOKEN O DIST_LABEL         (3-col distant-labeled, CuPuL train)
  conll_bio_dist: TOKEN BIO_TAG DIST_LABEL   (3-col with both, test_distant_labeling)
  bond_json     : [{"str_words":[], "tags":[]}]  (BOND / ATSEN / SCDL / DeSERT)
  roster        : train_text.txt + train_label_dist.txt + types.txt
  autoner_raw   : one token per line (no labels)
  autoner_ck    : TOKEN TIE_BREAK TYPE  (with <s>/<eof> boundaries)
  mproto_jsonl  : {"tokens":[], "entities":[{start,end,type}]}  per line

=== Usage ===

  from dsner_data import FormatConverter

  fc = FormatConverter(types=["Trait", "Gene"])   # or auto-detect

  # Read from any format
  sents = fc.read("./data/qtl/train_ALL.txt", fmt="conll_dist")
  sents = fc.read("./data/qtl/test.txt", fmt="conll_bio")
  sents = fc.read("./bond_data/train.json", fmt="bond_json")

  # Auto-detect format
  sents = fc.read("./data/qtl/train_ALL.txt")

  # Write to any format
  fc.write(sents, "./output/train.json", fmt="bond_json")
  fc.write(sents, "./output/roster_dir/", fmt="roster", split="train")
  fc.write(sents, "./output/train.jsonl", fmt="mproto_jsonl")
  fc.write(sents, "./output/test.txt", fmt="conll_bio")
  fc.write(sents, "./output/train_ALL.txt", fmt="conll_dist")

  # Convert directly between formats
  fc.convert("./input.txt", "./output.json", src_fmt="conll_dist", dst_fmt="bond_json")

  # Prepare one dataset for all methods at once
  from dsner_data import DataPreparer
  prep = DataPreparer("./data/qtl", "./original_dsner", "qtl")
  prep.prepare_all()

=== CLI ===

  # Format-to-format conversion
  python dsner_data.py convert --input train_ALL.txt --output train.json \\
      --src_fmt conll_dist --dst_fmt bond_json --types Trait Gene

  # Prepare dataset for all methods
  python dsner_data.py prepare --data_dir ./data/qtl --output_root ./original_dsner \\
      --dataset qtl

  # List supported formats
  python dsner_data.py formats
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("dsner_data")


# ═══════════════════════════════════════════════════════════════════════════
# Core data structure
# ═══════════════════════════════════════════════════════════════════════════

class Sentence:
    """Universal sentence representation that all formats convert to/from.

    Attributes
    ----------
    tokens : list[str]
        The words.
    bio_tags : list[str] or None
        BIO tags like ["B-Trait", "I-Trait", "O"].
    dist_labels : list[int] or None
        Distant integer labels like [1, 1, 0].  0=O, 1+=entity type index.
    """
    __slots__ = ("tokens", "bio_tags", "dist_labels")

    def __init__(
        self,
        tokens: List[str],
        bio_tags: Optional[List[str]] = None,
        dist_labels: Optional[List[int]] = None,
    ):
        self.tokens = tokens
        self.bio_tags = bio_tags
        self.dist_labels = dist_labels

    def __len__(self):
        return len(self.tokens)

    def __repr__(self):
        has_bio = "bio" if self.bio_tags else ""
        has_dist = "dist" if self.dist_labels else ""
        labels = "+".join(filter(None, [has_bio, has_dist])) or "unlabeled"
        return f"Sentence({len(self.tokens)} tokens, {labels})"


# ═══════════════════════════════════════════════════════════════════════════
# Label conversion helpers (on Sentence objects)
# ═══════════════════════════════════════════════════════════════════════════

def ensure_bio(sentences: List[Sentence], types: List[str]) -> List[Sentence]:
    """Ensure every sentence has bio_tags. Convert from dist_labels if missing."""
    result = []
    for s in sentences:
        if s.bio_tags is not None:
            result.append(s)
            continue
        if s.dist_labels is None:
            result.append(Sentence(s.tokens, ["O"] * len(s), None))
            continue
        bio = []
        prev: Optional[str] = None
        for dl in s.dist_labels:
            if dl == 0:
                bio.append("O")
                prev = None
            else:
                et = types[dl - 1] if dl - 1 < len(types) else f"TYPE{dl}"
                bio.append(f"I-{et}" if prev == et else f"B-{et}")
                prev = et
        result.append(Sentence(s.tokens, bio, s.dist_labels))
    return result


def ensure_dist(sentences: List[Sentence], types: List[str]) -> List[Sentence]:
    """Ensure every sentence has dist_labels. Convert from bio_tags if missing."""
    t2i = {t: i + 1 for i, t in enumerate(types)}
    result = []
    for s in sentences:
        if s.dist_labels is not None:
            result.append(s)
            continue
        if s.bio_tags is None:
            result.append(Sentence(s.tokens, None, [0] * len(s)))
            continue
        dist = [t2i.get(re.sub(r"^[BIES]-", "", t), 0) if t != "O" else 0 for t in s.bio_tags]
        result.append(Sentence(s.tokens, s.bio_tags, dist))
    return result


def extract_types_from_sentences(sentences: List[Sentence]) -> List[str]:
    """Extract entity type names from sentences (from bio_tags or dist_labels)."""
    types = set()
    max_dl = 0
    for s in sentences:
        if s.bio_tags:
            for t in s.bio_tags:
                if t != "O":
                    types.add(re.sub(r"^[BIES]-", "", t))
        if s.dist_labels:
            m = max(s.dist_labels)
            if m > max_dl:
                max_dl = m
    if types:
        return sorted(types)
    return [f"TYPE{i}" for i in range(1, max_dl + 1)]


# ═══════════════════════════════════════════════════════════════════════════
# Format registry
# ═══════════════════════════════════════════════════════════════════════════

SUPPORTED_FORMATS = {
    "conll_bio":      "TOKEN BIO_TAG                  (2-col, standard CoNLL)",
    "conll_dist":     "TOKEN O DIST_LABEL             (3-col, distant-labeled train)",
    "conll_bio_dist": "TOKEN BIO_TAG DIST_LABEL       (3-col, bio + distant)",
    "bond_json":      '[{"str_words":[], "tags":[]}]  (BOND/ATSEN/SCDL/DeSERT)',
    "roster":         "train_text.txt + label file     (RoSTER directory)",
    "autoner_raw":    "one token per line, no labels   (AutoNER raw_text.txt)",
    "autoner_ck":     "TOKEN TIE_BREAK TYPE            (AutoNER .ck truth file)",
    "mproto_jsonl":   '{"tokens":[], "entities":[]}    (MProto, one JSON/line)',
}

# Which methods use which format for train vs test
METHOD_FORMATS = {
    "BOND":    {"train": "bond_json",  "test": "bond_json"},
    "ATSEN":   {"train": "bond_json",  "test": "bond_json"},
    "SCDL":    {"train": "bond_json",  "test": "bond_json"},
    "DeSERT":  {"train": "bond_json",  "test": "bond_json"},
    "RoSTER":  {"train": "roster",     "test": "roster"},
    "CuPuL":   {"train": "conll_dist", "test": "conll_bio"},
    "AutoNER": {"train": "autoner_raw","test": "autoner_ck"},
    "MProto":  {"train": "mproto_jsonl","test": "mproto_jsonl"},
}


# ═══════════════════════════════════════════════════════════════════════════
# Readers  (format → List[Sentence])
# ═══════════════════════════════════════════════════════════════════════════

def _mkdirs(path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def read_types(path: str) -> List[str]:
    """Read types.txt — one type name per line."""
    types = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t:
                types.append(t)
    return types


def read_conll_bio(path: str) -> List[Sentence]:
    """Read 2-column CoNLL BIO: TOKEN TAG."""
    sents, toks, tags = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("-DOCSTART"):
                if toks:
                    sents.append(Sentence(toks, tags))
                    toks, tags = [], []
                continue
            p = line.split()
            toks.append(p[0])
            tags.append(p[1] if len(p) >= 2 else "O")
    if toks:
        sents.append(Sentence(toks, tags))
    logger.info("read_conll_bio: %d sentences from %s", len(sents), path)
    return sents


def read_conll_dist(path: str) -> List[Sentence]:
    """Read 3-column distant-labeled: TOKEN O DIST_LABEL."""
    sents, toks, dists = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("-DOCSTART"):
                if toks:
                    sents.append(Sentence(toks, dist_labels=dists))
                    toks, dists = [], []
                continue
            p = line.split()
            toks.append(p[0])
            dists.append(int(p[2]) if len(p) >= 3 else 0)
    if toks:
        sents.append(Sentence(toks, dist_labels=dists))
    logger.info("read_conll_dist: %d sentences from %s", len(sents), path)
    return sents


def read_conll_bio_dist(path: str) -> List[Sentence]:
    """Read 3-column with both tags: TOKEN BIO_TAG DIST_LABEL."""
    sents, toks, tags, dists = [], [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("-DOCSTART"):
                if toks:
                    sents.append(Sentence(toks, tags, dists))
                    toks, tags, dists = [], [], []
                continue
            p = line.split()
            toks.append(p[0])
            tags.append(p[1] if len(p) >= 2 else "O")
            dists.append(int(p[2]) if len(p) >= 3 else 0)
    if toks:
        sents.append(Sentence(toks, tags, dists))
    logger.info("read_conll_bio_dist: %d sentences from %s", len(sents), path)
    return sents


def read_bond_json(path: str, types: Optional[List[str]] = None) -> List[Sentence]:
    """Read BOND JSON: [{"str_words": [...], "tags": [int, ...]}].

    Needs types list or labels.json sidecar to decode tag integers back to BIO.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Try loading tag list from sidecar
    tag_list = None
    labels_path = os.path.join(os.path.dirname(path), "labels.json")
    if os.path.exists(labels_path):
        with open(labels_path) as f:
            tag_list = json.load(f)
    elif types:
        tag_list = ["O"]
        for t in types:
            tag_list.extend([f"B-{t}", f"I-{t}"])

    sents = []
    for item in data:
        tokens = item["str_words"]
        tag_ids = item["tags"]
        if tag_list:
            bio = [tag_list[i] if i < len(tag_list) else "O" for i in tag_ids]
        else:
            bio = ["O" if i == 0 else f"TAG{i}" for i in tag_ids]
        sents.append(Sentence(tokens, bio))
    logger.info("read_bond_json: %d sentences from %s", len(sents), path)
    return sents


def read_roster(directory: str, split: str = "train") -> List[Sentence]:
    """Read RoSTER format from a directory.

    Reads: {split}_text.txt + {split}_label_{dist|true}.txt.
    """
    text_path = os.path.join(directory, f"{split}_text.txt")
    # Try both label file variants
    label_path = None
    for suffix in ["dist", "true"]:
        lp = os.path.join(directory, f"{split}_label_{suffix}.txt")
        if os.path.exists(lp):
            label_path = lp
            break
    if not label_path:
        # Fallback: just the text
        label_path = None

    sents = []
    with open(text_path, "r", encoding="utf-8") as ft:
        lines_text = ft.readlines()
    lines_label = []
    if label_path:
        with open(label_path, "r", encoding="utf-8") as fl:
            lines_label = fl.readlines()

    for i, text_line in enumerate(lines_text):
        tokens = text_line.strip().split()
        if not tokens:
            continue
        if i < len(lines_label):
            tags = lines_label[i].strip().split()
            sents.append(Sentence(tokens, tags))
        else:
            sents.append(Sentence(tokens))
    logger.info("read_roster: %d sentences from %s (%s)", len(sents), directory, split)
    return sents


def read_autoner_raw(path: str) -> List[Sentence]:
    """Read AutoNER raw_text.txt — tokens only, no labels."""
    sents, toks = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if toks:
                    sents.append(Sentence(toks))
                    toks = []
                continue
            toks.append(line)
    if toks:
        sents.append(Sentence(toks))
    logger.info("read_autoner_raw: %d sentences from %s", len(sents), path)
    return sents


def read_autoner_ck(path: str) -> List[Sentence]:
    """Read AutoNER .ck truth file: TOKEN TIE_BREAK TYPE.

    <s> and <eof> are sentence boundary markers.
    TIE_BREAK: I = Break (entity start), O = Tie (continuation or outside).
    """
    sents, toks, tags = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if toks:
                    sents.append(Sentence(toks, tags))
                    toks, tags = [], []
                continue
            p = line.split()
            if p[0] in ("<s>", "<eof>"):
                continue
            token = p[0]
            tie_break = p[1] if len(p) >= 2 else "O"
            etype = p[2] if len(p) >= 3 else "O"
            toks.append(token)
            if etype == "O":
                tags.append("O")
            elif tie_break == "I":  # I = Break = entity start
                tags.append(f"B-{etype}")
            else:
                tags.append(f"I-{etype}")
    if toks:
        sents.append(Sentence(toks, tags))
    logger.info("read_autoner_ck: %d sentences from %s", len(sents), path)
    return sents


def read_mproto_jsonl(path: str) -> List[Sentence]:
    """Read MProto JSONL: one JSON per line with tokens + span entities."""
    sents = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            tokens = obj["tokens"]
            bio = ["O"] * len(tokens)
            for ent in obj.get("entities", []):
                start, end, etype = ent["start"], ent["end"], ent["type"]
                if start < len(bio):
                    bio[start] = f"B-{etype}"
                for j in range(start + 1, min(end, len(bio))):
                    bio[j] = f"I-{etype}"
            sents.append(Sentence(tokens, bio))
    logger.info("read_mproto_jsonl: %d sentences from %s", len(sents), path)
    return sents


# Reader dispatch
_READERS = {
    "conll_bio": read_conll_bio,
    "conll_dist": read_conll_dist,
    "conll_bio_dist": read_conll_bio_dist,
    "bond_json": read_bond_json,
    "roster": read_roster,
    "autoner_raw": read_autoner_raw,
    "autoner_ck": read_autoner_ck,
    "mproto_jsonl": read_mproto_jsonl,
}


def auto_detect_format(path: str) -> str:
    """Guess the format of a file or directory."""
    p = Path(path)

    # Directory → roster
    if p.is_dir():
        if any((p / f).exists() for f in ["train_text.txt", "test_text.txt"]):
            return "roster"
        raise ValueError(f"Cannot detect format for directory: {path}")

    suffix = p.suffix.lower()
    name = p.name.lower()

    # By extension
    if suffix == ".json":
        return "bond_json"
    if suffix == ".jsonl":
        return "mproto_jsonl"
    if suffix == ".ck":
        return "autoner_ck"

    # By content inspection
    if suffix in (".txt", ".bio", ".conll", ""):
        with open(path, "r", encoding="utf-8") as f:
            bio_tags_seen = set()
            col_counts = set()
            for i, line in enumerate(f):
                if i > 200:
                    break
                line = line.rstrip("\n")
                if not line or line.startswith("-DOCSTART") or line.startswith("<s>") or line.startswith("<eof>"):
                    if line.startswith("<s>") or line.startswith("<eof>"):
                        return "autoner_ck"
                    continue
                parts = line.split()
                col_counts.add(len(parts))
                if len(parts) == 1:
                    return "autoner_raw"
                if len(parts) >= 2:
                    bio_tags_seen.add(parts[1])

            has_bio = any(t.startswith(("B-", "I-", "S-", "E-")) for t in bio_tags_seen)
            if 3 in col_counts:
                if has_bio:
                    return "conll_bio_dist"
                else:
                    return "conll_dist"
            if 2 in col_counts:
                return "conll_bio"

    return "conll_bio"  # fallback


# ═══════════════════════════════════════════════════════════════════════════
# Writers  (List[Sentence] → file)
# ═══════════════════════════════════════════════════════════════════════════

def write_conll_bio(sentences: List[Sentence], path: str, types: List[str]):
    """Write 2-column CoNLL BIO: TOKEN TAG."""
    sents = ensure_bio(sentences, types)
    _mkdirs(path)
    with open(path, "w", encoding="utf-8") as f:
        for s in sents:
            for tok, tag in zip(s.tokens, s.bio_tags):
                f.write(f"{tok} {tag}\n")
            f.write("\n")
    logger.info("write_conll_bio: %d sents → %s", len(sents), path)


def write_conll_dist(sentences: List[Sentence], path: str, types: List[str]):
    """Write 3-column distant: TOKEN O DIST_LABEL."""
    sents = ensure_dist(sentences, types)
    _mkdirs(path)
    with open(path, "w", encoding="utf-8") as f:
        for s in sents:
            for tok, dl in zip(s.tokens, s.dist_labels):
                f.write(f"{tok} O {dl}\n")
            f.write("\n")
    logger.info("write_conll_dist: %d sents → %s", len(sents), path)


def write_conll_bio_dist(sentences: List[Sentence], path: str, types: List[str]):
    """Write 3-column: TOKEN BIO_TAG DIST_LABEL."""
    sents = ensure_bio(sentences, types)
    sents = ensure_dist(sents, types)
    _mkdirs(path)
    with open(path, "w", encoding="utf-8") as f:
        for s in sents:
            for tok, tag, dl in zip(s.tokens, s.bio_tags, s.dist_labels):
                f.write(f"{tok} {tag} {dl}\n")
            f.write("\n")
    logger.info("write_conll_bio_dist: %d sents → %s", len(sents), path)


def write_bond_json(sentences: List[Sentence], path: str, types: List[str]):
    """Write BOND JSON format."""
    sents = ensure_bio(sentences, types)
    tag_list = ["O"]
    for t in types:
        tag_list.extend([f"B-{t}", f"I-{t}"])
    tag2id = {t: i for i, t in enumerate(tag_list)}

    data = []
    for s in sents:
        tags_int = [tag2id.get(t, 0) for t in s.bio_tags]
        data.append({"str_words": s.tokens, "tags": tags_int})

    _mkdirs(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    d = os.path.dirname(path)
    with open(os.path.join(d, "tag_to_id.json"), "w") as f:
        json.dump(tag2id, f)
    with open(os.path.join(d, "labels.json"), "w") as f:
        json.dump(tag_list, f)
    logger.info("write_bond_json: %d sents → %s", len(data), path)


def write_roster(
    sentences: List[Sentence], directory: str, types: List[str], split: str = "train",
):
    """Write RoSTER format directory."""
    sents = ensure_bio(sentences, types)
    os.makedirs(directory, exist_ok=True)
    lbl = "dist" if split == "train" else "true"
    text_path = os.path.join(directory, f"{split}_text.txt")
    label_path = os.path.join(directory, f"{split}_label_{lbl}.txt")

    with open(text_path, "w", encoding="utf-8") as ft, \
         open(label_path, "w", encoding="utf-8") as fl:
        for s in sents:
            ft.write(" ".join(s.tokens) + "\n")
            fl.write(" ".join(s.bio_tags) + "\n")

    with open(os.path.join(directory, "types.txt"), "w") as f:
        for t in types:
            f.write(t + "\n")
    logger.info("write_roster: %d sents → %s (%s)", len(sents), directory, split)


def write_autoner_raw(sentences: List[Sentence], path: str, **_):
    """Write AutoNER raw_text.txt — tokens only."""
    _mkdirs(path)
    with open(path, "w", encoding="utf-8") as f:
        for s in sentences:
            for tok in s.tokens:
                f.write(tok + "\n")
            f.write("\n")
    logger.info("write_autoner_raw: %d sents → %s", len(sentences), path)


def write_autoner_ck(sentences: List[Sentence], path: str, types: List[str]):
    """Write AutoNER .ck truth file."""
    sents = ensure_bio(sentences, types)
    _mkdirs(path)
    with open(path, "w", encoding="utf-8") as f:
        for s in sents:
            f.write("<s> O O\n")
            prev = "O"
            for tok, tag in zip(s.tokens, s.bio_tags):
                if tag == "O":
                    f.write(f"{tok} O O\n")
                    prev = "O"
                else:
                    et = re.sub(r"^[BIES]-", "", tag)
                    is_begin = tag.startswith(("B-", "S-")) or et != prev
                    f.write(f"{tok} {'I' if is_begin else 'O'} {et}\n")
                    prev = et
            f.write("<eof> O O\n\n")
    logger.info("write_autoner_ck: %d sents → %s", len(sents), path)


def write_mproto_jsonl(
    sentences: List[Sentence], path: str, types: List[str], source: str = "ds",
):
    """Write MProto JSONL format."""
    sents = ensure_bio(sentences, types)
    _mkdirs(path)
    with open(path, "w", encoding="utf-8") as f:
        for idx, s in enumerate(sents):
            entities = []
            i = 0
            while i < len(s.bio_tags):
                tag = s.bio_tags[i]
                if tag != "O":
                    et = re.sub(r"^[BIES]-", "", tag)
                    start = i
                    i += 1
                    while i < len(s.bio_tags) and s.bio_tags[i].startswith("I-"):
                        i += 1
                    entities.append({"start": start, "end": i, "type": et})
                else:
                    i += 1
            f.write(json.dumps({
                "id": f"{source}_{idx}",
                "tokens": s.tokens,
                "entities": entities,
                "human_entities": [],
            }, ensure_ascii=False) + "\n")
    logger.info("write_mproto_jsonl: %d sents → %s", len(sents), path)


# Writer dispatch
_WRITERS = {
    "conll_bio": write_conll_bio,
    "conll_dist": write_conll_dist,
    "conll_bio_dist": write_conll_bio_dist,
    "bond_json": write_bond_json,
    "roster": write_roster,
    "autoner_raw": write_autoner_raw,
    "autoner_ck": write_autoner_ck,
    "mproto_jsonl": write_mproto_jsonl,
}


# ═══════════════════════════════════════════════════════════════════════════
# FormatConverter — the main conversion API
# ═══════════════════════════════════════════════════════════════════════════

class FormatConverter:
    """Convert DS-NER data between any pair of supported formats.

    Parameters
    ----------
    types : list[str], optional
        Entity type names (e.g. ["Trait", "Gene"]).
        Auto-detected from data if not provided.
    types_file : str, optional
        Path to types.txt to read types from.
    """

    def __init__(
        self,
        types: Optional[List[str]] = None,
        types_file: Optional[str] = None,
    ):
        if types:
            self.types = types
        elif types_file and os.path.exists(types_file):
            self.types = read_types(types_file)
        else:
            self.types = None  # auto-detect on first read

    def read(
        self,
        path: str,
        fmt: Optional[str] = None,
        split: str = "train",
    ) -> List[Sentence]:
        """Read data from any supported format.

        Parameters
        ----------
        path : str
            File or directory path.
        fmt : str, optional
            Format name. Auto-detected if omitted.
        split : str
            For roster format, which split to read (train/test).
        """
        if fmt is None:
            fmt = auto_detect_format(path)
            logger.info("Auto-detected format: %s", fmt)

        if fmt not in _READERS:
            raise ValueError(f"Unknown format '{fmt}'. Supported: {list(_READERS.keys())}")

        if fmt == "roster":
            sents = read_roster(path, split=split)
        elif fmt == "bond_json":
            sents = read_bond_json(path, types=self.types)
        else:
            sents = _READERS[fmt](path)

        # Auto-detect types if not set
        if self.types is None:
            self.types = extract_types_from_sentences(sents)
            logger.info("Auto-detected types: %s", self.types)

        return sents

    def write(
        self,
        sentences: List[Sentence],
        path: str,
        fmt: str,
        split: str = "train",
        source: str = "ds",
    ):
        """Write data to any supported format.

        Parameters
        ----------
        sentences : list[Sentence]
            The data.
        path : str
            Output file or directory.
        fmt : str
            Target format name.
        split : str
            For roster format (train/test).
        source : str
            For mproto_jsonl, prefix for sample IDs.
        """
        if self.types is None:
            self.types = extract_types_from_sentences(sentences)

        if fmt == "roster":
            write_roster(sentences, path, self.types, split=split)
        elif fmt == "mproto_jsonl":
            write_mproto_jsonl(sentences, path, self.types, source=source)
        elif fmt == "autoner_raw":
            write_autoner_raw(sentences, path)
        elif fmt in _WRITERS:
            _WRITERS[fmt](sentences, path, self.types)
        else:
            raise ValueError(f"Unknown format '{fmt}'. Supported: {list(_WRITERS.keys())}")

    def convert(
        self,
        input_path: str,
        output_path: str,
        src_fmt: Optional[str] = None,
        dst_fmt: str = "conll_bio",
        split: str = "train",
    ):
        """Convert between any two formats.

        Parameters
        ----------
        input_path : str
            Source file or directory.
        output_path : str
            Destination file or directory.
        src_fmt : str, optional
            Source format (auto-detected if None).
        dst_fmt : str
            Target format.
        split : str
            For roster format.
        """
        sents = self.read(input_path, fmt=src_fmt, split=split)
        self.write(sents, output_path, fmt=dst_fmt, split=split)
        logger.info("Converted %s → %s (%d sentences)", src_fmt or "auto", dst_fmt, len(sents))


# ═══════════════════════════════════════════════════════════════════════════
# DataPreparer — batch preparation for all 8 methods
# ═══════════════════════════════════════════════════════════════════════════

class DataPreparer:
    """Prepare one dataset for all 8 DS-NER methods using FormatConverter.

    Parameters
    ----------
    data_dir : str
        Directory with train_ALL.txt, test.txt, types.txt.
    output_root : str
        Root of the original_dsner project.
    dataset_name : str
        Dataset name (e.g., "qtl").
    types : list[str], optional
        Entity types. Auto-read from types.txt if omitted.
    """

    _DIR_NAMES = {
        "ATSEN": "ATSEN", "AutoNER": "AutoNER", "BOND": "BOND",
        "CuPuL": "CuPuL", "DeSERT": "DeSERT", "MProto": "mproto",
        "RoSTER": "RoSTER", "SCDL": "SCDL",
    }

    _TRAIN_FILES = ["train_ALL.txt", "train.txt", "train.bio"]
    _TEST_FILES  = ["test.txt", "test.bio"]
    _DEV_FILES   = ["dev.txt", "dev.bio", "valid.txt"]

    def __init__(
        self,
        data_dir: str,
        output_root: str,
        dataset_name: str = "qtl",
        types: Optional[List[str]] = None,
    ):
        self.data_dir = Path(data_dir)
        self.output_root = Path(output_root)
        self.ds = dataset_name

        # Init converter with types
        types_file = self.data_dir / "types.txt"
        self.fc = FormatConverter(
            types=types,
            types_file=str(types_file) if types_file.exists() else None,
        )

        # Read train
        train_path = self._find(self._TRAIN_FILES)
        if not train_path:
            raise FileNotFoundError(f"No train file in {data_dir}")
        self.train = self.fc.read(str(train_path))

        # Read test
        test_path = self._find(self._TEST_FILES)
        self.test = self.fc.read(str(test_path)) if test_path else None

        # Read dev
        dev_path = self._find(self._DEV_FILES)
        self.dev = self.fc.read(str(dev_path)) if dev_path else None

        logger.info(
            "DataPreparer: %s  types=%s  train=%d  test=%d  dev=%d",
            dataset_name, self.fc.types,
            len(self.train),
            len(self.test) if self.test else 0,
            len(self.dev) if self.dev else 0,
        )

    def _find(self, candidates: List[str]) -> Optional[Path]:
        for name in candidates:
            p = self.data_dir / name
            if p.exists():
                return p
        return None

    def prepare_for(self, method: str) -> str:
        """Prepare data for one method. Returns output path."""
        key = method.upper().replace("-", "").replace("_", "")
        lookup = {k.upper().replace("-", "").replace("_", ""): k for k in self._DIR_NAMES}
        if key not in lookup:
            raise ValueError(f"Unknown method '{method}'. Options: {list(self._DIR_NAMES)}")
        method_name = lookup[key]
        fn = getattr(self, f"_prep_{method_name.lower()}")
        root = self.output_root / self._DIR_NAMES[method_name]
        return fn(root)

    def prepare_all(self) -> Dict[str, str]:
        """Prepare data for all 8 methods."""
        results = {}
        for method in self._DIR_NAMES:
            try:
                results[method] = self.prepare_for(method)
            except Exception as e:
                logger.error("✗ %s: %s", method, e)
                results[method] = f"ERROR: {e}"
        return results

    # ── Per-method helpers (use self.fc.write) ─────────────────────────

    def _write_dev_or_test(self, path: str, fmt: str, split: str = "test"):
        """Write dev set; fall back to test if no dev."""
        data = self.dev if self.dev else self.test
        if data:
            self.fc.write(data, path, fmt=fmt, split=split, source=f"{self.ds}_dev")

    def _prep_bond(self, root: Path) -> str:
        d = root / "dataset" / self.ds
        os.makedirs(d, exist_ok=True)
        self.fc.write(self.train, str(d / "train.json"), fmt="bond_json")
        if self.test:
            self.fc.write(self.test, str(d / "test.json"), fmt="bond_json")
        self._write_dev_or_test(str(d / "dev.json"), "bond_json")
        return str(d)

    def _prep_atsen(self, root: Path) -> str:
        d = root / "dataset" / self.ds
        os.makedirs(d, exist_ok=True)
        self.fc.write(self.train, str(d / f"{self.ds}_train.json"), fmt="bond_json")
        if self.test:
            self.fc.write(self.test, str(d / f"{self.ds}_test.json"), fmt="bond_json")
        self._write_dev_or_test(str(d / f"{self.ds}_dev.json"), "bond_json")
        return str(d)

    def _prep_scdl(self, root: Path) -> str:
        d = root / "dataset"
        os.makedirs(d, exist_ok=True)
        self.fc.write(self.train, str(d / f"{self.ds}_train.json"), fmt="bond_json")
        if self.test:
            self.fc.write(self.test, str(d / f"{self.ds}_test.json"), fmt="bond_json")
        self._write_dev_or_test(str(d / f"{self.ds}_dev.json"), "bond_json")
        return str(d)

    def _prep_desert(self, root: Path) -> str:
        d = root / "dataset" / self.ds
        os.makedirs(d, exist_ok=True)
        self.fc.write(self.train, str(d / f"{self.ds}_train.json"), fmt="bond_json")
        if self.test:
            self.fc.write(self.test, str(d / f"{self.ds}_test.json"), fmt="bond_json")
        self._write_dev_or_test(str(d / f"{self.ds}_dev.json"), "bond_json")
        return str(d)

    def _prep_roster(self, root: Path) -> str:
        d = str(root / "data" / self.ds)
        self.fc.write(self.train, d, fmt="roster", split="train")
        if self.test:
            self.fc.write(self.test, d, fmt="roster", split="test")
        return d

    def _prep_cupul(self, root: Path) -> str:
        d = root / "data" / self.ds
        os.makedirs(d, exist_ok=True)
        self.fc.write(self.train, str(d / "train.ALL.txt"), fmt="conll_dist")
        bio_train = ensure_bio(self.train, self.fc.types)
        self.fc.write(bio_train, str(d / "train.txt"), fmt="conll_bio")
        if self.test:
            self.fc.write(self.test, str(d / "test.txt"), fmt="conll_bio")
        if self.dev:
            self.fc.write(self.dev, str(d / "dev.txt"), fmt="conll_bio")
        return str(d)

    def _prep_autoner(self, root: Path) -> str:
        d = root / "data" / self.ds
        os.makedirs(d, exist_ok=True)
        self.fc.write(self.train, str(d / "raw_text.txt"), fmt="autoner_raw")
        if self.test:
            self.fc.write(self.test, str(d / "truth_test.ck"), fmt="autoner_ck")
        dev = self.dev if self.dev else self.test
        if dev:
            self.fc.write(dev, str(d / "truth_dev.ck"), fmt="autoner_ck")
        return str(d)

    def _prep_mproto(self, root: Path) -> str:
        d = root / "data" / self.ds
        os.makedirs(d, exist_ok=True)
        self.fc.write(self.train, str(d / "train.jsonl"), fmt="mproto_jsonl", source=f"{self.ds}_train")
        if self.test:
            self.fc.write(self.test, str(d / "test.jsonl"), fmt="mproto_jsonl", source=f"{self.ds}_test")
        if self.dev:
            self.fc.write(self.dev, str(d / "dev.jsonl"), fmt="mproto_jsonl", source=f"{self.ds}_dev")
        meta = {"entities": {t: {"short": t} for t in self.fc.types}}
        with open(d / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        return str(d)


# ═══════════════════════════════════════════════════════════════════════════
# Convenience shortcut for QTL
# ═══════════════════════════════════════════════════════════════════════════

def prepare_qtl(
    data_dir: str,
    output_root: str,
    methods: Optional[List[str]] = None,
    types: Optional[List[str]] = None,
) -> Dict[str, str]:
    """One-liner to prepare QTL data for DS-NER methods."""
    prep = DataPreparer(data_dir, output_root, "qtl", types)
    return prep.prepare_all() if not methods else {m: prep.prepare_for(m) for m in methods}


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="DS-NER data format converter",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # -- formats --
    sub.add_parser("formats", help="List supported formats")

    # -- convert --
    p_conv = sub.add_parser("convert", help="Convert between two formats")
    p_conv.add_argument("--input", required=True, help="Input file or directory")
    p_conv.add_argument("--output", required=True, help="Output file or directory")
    p_conv.add_argument("--src_fmt", default=None, help="Source format (auto-detect if omitted)")
    p_conv.add_argument("--dst_fmt", required=True, help="Target format")
    p_conv.add_argument("--types", nargs="*", help="Entity types")
    p_conv.add_argument("--types_file", default=None, help="Path to types.txt")
    p_conv.add_argument("--split", default="train", help="Split for roster format")

    # -- prepare --
    p_prep = sub.add_parser("prepare", help="Prepare dataset for all/some methods")
    p_prep.add_argument("--data_dir", required=True, help="Dir with train_ALL.txt, test.txt, types.txt")
    p_prep.add_argument("--output_root", required=True, help="Path to original_dsner")
    p_prep.add_argument("--dataset", default="qtl", help="Dataset name")
    p_prep.add_argument("--methods", nargs="*", help="Methods (default: all 8)")
    p_prep.add_argument("--types", nargs="*", help="Entity types")

    args = parser.parse_args()

    if args.command == "formats":
        print("\nSupported formats:")
        print("-" * 70)
        for name, desc in SUPPORTED_FORMATS.items():
            print(f"  {name:18s} {desc}")
        print("\nMethod format requirements:")
        print("-" * 70)
        for method, fmts in METHOD_FORMATS.items():
            print(f"  {method:10s} train={fmts['train']:18s} test={fmts['test']}")

    elif args.command == "convert":
        fc = FormatConverter(types=args.types, types_file=args.types_file)
        fc.convert(args.input, args.output, args.src_fmt, args.dst_fmt, args.split)

    elif args.command == "prepare":
        prep = DataPreparer(args.data_dir, args.output_root, args.dataset, args.types)
        if args.methods:
            for m in args.methods:
                prep.prepare_for(m)
        else:
            results = prep.prepare_all()
            print("\n" + "=" * 60)
            for method, path in results.items():
                ok = "✓" if not str(path).startswith("ERROR") else "✗"
                print(f"  {ok} {method:10s} → {path}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
