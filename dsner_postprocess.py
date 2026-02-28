"""
dsner_postprocess.py — Unified post-processing for all DS-NER method predictions.

All 8 methods produce predictions in different formats. This module:
  1. Collects predictions from any method's native output into a unified format
  2. Applies configurable post-processing rules
  3. Writes the cleaned predictions back out
  4. Evaluates (precision / recall / F1)

=== Unified Prediction Format ===

  TOKEN GOLD_TAG PRED_LABEL

  Where PRED_LABEL is an integer: 0=O, 1+=entity type index.
  Blank lines separate sentences.

  Example:
    shear B-Trait 1
    force I-Trait 1
    collected O 0

=== Usage ===

  from dsner_postprocess import PostProcessor, collect_predictions

  # Step 1: Collect predictions from any method
  pred_file = collect_predictions(
      method="BOND",
      project_root="./original_dsner",
      dataset="qtl",
      gold_file="./data/qtl/test.txt",        # gold BIO test file
  )

  # Step 2: Post-process
  pp = PostProcessor(pred_file)
  pp.run_all()                  # apply all rules in order
  pp.save("pred_final.txt")

  # Step 3: Evaluate
  pp.evaluate()

  # Or pick specific rules
  pp.run(rules=["span_consistency", "prep_bridging"])
  pp.save("pred_custom.txt")

=== CLI ===

  # Post-process a prediction file
  python dsner_postprocess.py process --input pred.txt --output pred_final.txt

  # Collect from a method then post-process
  python dsner_postprocess.py collect --method BOND --project_root ./original_dsner \\
      --dataset qtl --gold_file ./data/qtl/test.txt --output pred_unified.txt

  # Evaluate a prediction file
  python dsner_postprocess.py evaluate --input pred_final.txt
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("dsner_postprocess")


# ═══════════════════════════════════════════════════════════════════════════
# Prediction File I/O
# ═══════════════════════════════════════════════════════════════════════════

# Token: [word, gold_tag, pred_label, original_index]
# None = sentence boundary

Token = Optional[List]  # [str, str, int, int] or None


def read_predictions(path: str) -> List[Token]:
    """Read unified prediction file: TOKEN GOLD_TAG PRED_LABEL.

    Returns flat list where None = sentence boundary.
    """
    tokens: List[Token] = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if line == "":
                tokens.append(None)
            else:
                parts = line.split()
                if len(parts) >= 3:
                    tokens.append([parts[0], parts[1], int(parts[2]), idx])
                elif len(parts) == 2:
                    tokens.append([parts[0], parts[1], 0, idx])
    logger.info("Read %d lines from %s", len(tokens), path)
    return tokens


def write_predictions(tokens: List[Token], path: str):
    """Write unified prediction file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for tok in tokens:
            if tok is None:
                f.write("\n")
            else:
                f.write(f"{tok[0]} {tok[1]} {tok[2]}\n")
    logger.info("Wrote predictions to %s", path)


def read_gold_bio(path: str) -> List[List[Tuple[str, str]]]:
    """Read gold BIO file (TOKEN TAG). Returns list of sentences."""
    sents, cur = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("-DOCSTART"):
                if cur:
                    sents.append(cur)
                    cur = []
                continue
            parts = line.split()
            cur.append((parts[0], parts[1] if len(parts) >= 2 else "O"))
    if cur:
        sents.append(cur)
    return sents


# ═══════════════════════════════════════════════════════════════════════════
# Prediction Collectors — extract predictions from each method's output
# ═══════════════════════════════════════════════════════════════════════════

def _bio_pred_to_int(tag: str, types: List[str]) -> int:
    """Convert a BIO prediction tag to integer label."""
    if tag == "O":
        return 0
    etype = re.sub(r"^[BIES]-", "", tag)
    for i, t in enumerate(types):
        if t == etype:
            return i + 1
    return 1  # default to first entity type


def collect_from_conll_pred(
    pred_path: str, gold_path: str, types: List[str],
) -> str:
    """Collect from CoNLL-format prediction file (TOKEN GOLD PRED).

    Works for: CuPuL, RoSTER, and any method that outputs CoNLL format.
    The pred column can be BIO tags or integers.
    """
    output = os.path.splitext(pred_path)[0] + "_unified.txt"
    gold_sents = read_gold_bio(gold_path)

    pred_tokens = []
    with open(pred_path, "r", encoding="utf-8") as f:
        cur = []
        for line in f:
            line = line.strip()
            if not line:
                if cur:
                    pred_tokens.append(cur)
                    cur = []
                continue
            parts = line.split()
            cur.append(parts)
        if cur:
            pred_tokens.append(cur)

    with open(output, "w", encoding="utf-8") as f:
        for si, pred_sent in enumerate(pred_tokens):
            gold_sent = gold_sents[si] if si < len(gold_sents) else None
            for ti, parts in enumerate(pred_sent):
                token = parts[0]
                gold = gold_sent[ti][1] if gold_sent and ti < len(gold_sent) else "O"
                # Pred can be integer or BIO tag
                pred_raw = parts[-1]
                if pred_raw.lstrip("-").isdigit():
                    pred_int = int(pred_raw)
                else:
                    pred_int = _bio_pred_to_int(pred_raw, types)
                f.write(f"{token} {gold} {pred_int}\n")
            f.write("\n")

    logger.info("Collected predictions → %s", output)
    return output


def collect_from_bond_json(
    pred_path: str, gold_path: str, types: List[str],
) -> str:
    """Collect from BOND/ATSEN/SCDL/DeSERT JSON prediction output.

    These methods typically write test_predictions.txt in CoNLL format,
    or a JSON file with predicted tags.
    """
    # BOND writes test_predictions.txt as CoNLL format
    if pred_path.endswith(".txt"):
        return collect_from_conll_pred(pred_path, gold_path, types)

    # JSON format
    output = os.path.splitext(pred_path)[0] + "_unified.txt"
    with open(pred_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    gold_sents = read_gold_bio(gold_path)

    tag_list = ["O"]
    for t in types:
        tag_list.extend([f"B-{t}", f"I-{t}"])

    with open(output, "w", encoding="utf-8") as f:
        for si, item in enumerate(data):
            tokens = item.get("str_words", item.get("tokens", []))
            preds = item.get("preds", item.get("tags", []))
            gold_sent = gold_sents[si] if si < len(gold_sents) else None
            for ti, (token, pred_id) in enumerate(zip(tokens, preds)):
                gold = gold_sent[ti][1] if gold_sent and ti < len(gold_sent) else "O"
                if isinstance(pred_id, int):
                    pred_tag = tag_list[pred_id] if pred_id < len(tag_list) else "O"
                    pred_int = _bio_pred_to_int(pred_tag, types)
                else:
                    pred_int = _bio_pred_to_int(str(pred_id), types)
                f.write(f"{token} {gold} {pred_int}\n")
            f.write("\n")

    logger.info("Collected BOND-style predictions → %s", output)
    return output


def collect_from_mproto_jsonl(
    pred_path: str, gold_path: str, types: List[str],
) -> str:
    """Collect from MProto JSONL prediction output."""
    output = os.path.splitext(pred_path)[0] + "_unified.txt"
    gold_sents = read_gold_bio(gold_path)
    type2id = {t: i + 1 for i, t in enumerate(types)}

    with open(pred_path, "r", encoding="utf-8") as pf, \
         open(output, "w", encoding="utf-8") as of:
        for si, line in enumerate(pf):
            obj = json.loads(line.strip())
            tokens = obj["tokens"]
            preds = [0] * len(tokens)
            for ent in obj.get("entities", obj.get("pred_entities", [])):
                eid = type2id.get(ent["type"], 1)
                for j in range(ent["start"], min(ent["end"], len(tokens))):
                    preds[j] = eid
            gold_sent = gold_sents[si] if si < len(gold_sents) else None
            for ti, (token, pred) in enumerate(zip(tokens, preds)):
                gold = gold_sent[ti][1] if gold_sent and ti < len(gold_sent) else "O"
                of.write(f"{token} {gold} {pred}\n")
            of.write("\n")

    logger.info("Collected MProto predictions → %s", output)
    return output


def collect_predictions(
    method: str,
    project_root: str,
    dataset: str,
    gold_file: str,
    pred_file: Optional[str] = None,
    types: Optional[List[str]] = None,
    output: Optional[str] = None,
) -> str:
    """Auto-detect and collect predictions from any method into unified format.

    Parameters
    ----------
    method : str
        Method name (BOND, ATSEN, CuPuL, etc.).
    project_root : str
        Path to original_dsner.
    dataset : str
        Dataset name.
    gold_file : str
        Path to gold test BIO file.
    pred_file : str, optional
        Explicit path to prediction file. Auto-searched if None.
    types : list[str], optional
        Entity types. Auto-detected if None.
    output : str, optional
        Output path. Auto-generated if None.

    Returns
    -------
    str : path to unified prediction file.
    """
    if types is None:
        # Try to read types from gold file
        sents = read_gold_bio(gold_file)
        t = set()
        for s in sents:
            for _, tag in s:
                if tag != "O":
                    t.add(re.sub(r"^[BIES]-", "", tag))
        types = sorted(t) or ["Trait"]

    method_upper = method.upper().replace("-", "").replace("_", "")
    dir_names = {
        "ATSEN": "ATSEN", "AUTONER": "AutoNER", "BOND": "BOND",
        "CUPUL": "CuPuL", "DESERT": "DeSERT", "MPROTO": "mproto",
        "ROSTER": "RoSTER", "SCDL": "SCDL",
    }
    method_dir = Path(project_root) / dir_names.get(method_upper, method)

    if pred_file is None:
        pred_file = _find_prediction_file(method_upper, method_dir, dataset)

    if output is None:
        output = str(Path(pred_file).parent / f"pred_{dataset}_unified.txt")

    # Dispatch by method
    if method_upper in ("BOND", "ATSEN", "SCDL", "DESERT"):
        result = collect_from_bond_json(pred_file, gold_file, types)
    elif method_upper == "MPROTO":
        result = collect_from_mproto_jsonl(pred_file, gold_file, types)
    else:
        # CuPuL, RoSTER, AutoNER — all produce CoNLL-like output
        result = collect_from_conll_pred(pred_file, gold_file, types)

    # Copy to desired output location if different
    if output != result:
        import shutil
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        shutil.copy2(result, output)
        logger.info("Copied to %s", output)
        return output
    return result


def _find_prediction_file(method: str, method_dir: Path, dataset: str) -> str:
    """Search for prediction output file from a method."""
    # Common patterns per method
    search_patterns = {
        "BOND": [
            f"dataset/{dataset}/test_predictions.txt",
            f"output/{dataset}/test_predictions.txt",
            f"ptms/{dataset}/test_predictions.txt",
        ],
        "ATSEN": [
            f"ptms/{dataset}/test_predictions.txt",
            f"ptms/{dataset}/checkpoint-best/test_predictions.txt",
        ],
        "SCDL": [
            f"ptms/{dataset}/test_predictions.txt",
        ],
        "DESERT": [
            f"ptms/{dataset}/test_predictions.txt",
        ],
        "CUPUL": [
            f"data/{dataset}/output/predictions.txt",
            f"../data/{dataset}/output/predictions.txt",
        ],
        "ROSTER": [
            f"out_{dataset}/pred_test.txt",
            f"out_{dataset}/test_predictions.txt",
        ],
        "AUTONER": [
            f"models/{dataset}/decoded.txt",
        ],
        "MPROTO": [
            f"data/{dataset}/test_predictions.jsonl",
        ],
    }

    patterns = search_patterns.get(method, [])
    for pattern in patterns:
        path = method_dir / pattern
        if path.exists():
            logger.info("Found prediction file: %s", path)
            return str(path)

    # Fallback: search for any prediction-like file
    for f in method_dir.rglob("*predict*"):
        if f.is_file() and dataset.lower() in str(f).lower():
            logger.info("Found prediction file (fallback): %s", f)
            return str(f)
    for f in method_dir.rglob("*pred*test*"):
        if f.is_file():
            logger.info("Found prediction file (fallback): %s", f)
            return str(f)

    raise FileNotFoundError(
        f"No prediction file found for {method} in {method_dir}.\n"
        f"Searched: {patterns}\n"
        f"Provide --pred_file explicitly."
    )


# ═══════════════════════════════════════════════════════════════════════════
# Post-Processing Rules
# ═══════════════════════════════════════════════════════════════════════════

PREPOSITIONS = {
    "of", "in", "on", "by", "to", "with", "for", "from", "at", "against",
    "as", "under", "over", "into", "between",
}

# Rule names and their execution order
DEFAULT_RULE_ORDER = [
    "span_consistency",      # Rule 4
    "prep_bridging",         # Rule 1
    "span_consistency",      # Rule 4 (re-apply)
    "abbrev_resolution",     # Rule 2
    "span_consistency",      # Rule 4 (re-apply)
    "pos_filtering",         # Rule 0
]

AVAILABLE_RULES = {
    "span_consistency",
    "prep_bridging",
    "abbrev_resolution",
    "pos_filtering",
}


def rule_span_consistency(tokens: List[Token], **kwargs) -> List[Token]:
    """Rule 4: Document-level case-insensitive span consistency.

    If a multi-token span is predicted as entity somewhere in the document,
    mark all other occurrences of the same span (case-insensitive) as entity.
    """
    # Collect multi-token entity spans
    def get_spans(tl):
        spans, span = [], []
        for tok in tl:
            if tok is None:
                if span:
                    spans.append(span)
                    span = []
            else:
                if tok[2] >= 1:
                    span.append(tok)
                else:
                    if span:
                        spans.append(span)
                        span = []
        if span:
            spans.append(span)
        return [s for s in spans if len(s) > 1]

    spans = get_spans(tokens)
    span_words_list = [
        [tok[0].lower() for tok in span] for span in spans
    ]

    if not span_words_list:
        return tokens

    marked: Set[int] = set()
    i = 0
    while i < len(tokens):
        if tokens[i] is None:
            i += 1
            continue
        for span_words in span_words_list:
            span_len = len(span_words)
            if i + span_len <= len(tokens):
                if all(
                    tokens[i + j] is not None
                    and tokens[i + j][0].lower() == span_words[j]
                    and (i + j) not in marked
                    for j in range(span_len)
                ):
                    for j in range(span_len):
                        tokens[i + j][2] = 1
                        marked.add(i + j)
                    break
        i += 1
    return tokens


def rule_prep_bridging(
    tokens: List[Token],
    prepositions: Optional[Set[str]] = None,
    **kwargs,
) -> List[Token]:
    """Rule 1: Preposition bridging.

    If we see pattern [entity] [preposition] [entity], mark the
    preposition as entity too (bridges the gap).
    """
    if prepositions is None:
        prepositions = PREPOSITIONS
    for i in range(1, len(tokens) - 1):
        if tokens[i] is None:
            continue
        prev, curr, nxt = tokens[i - 1], tokens[i], tokens[i + 1]
        if prev and nxt:
            if prev[2] >= 1 and curr[2] == 0 and nxt[2] >= 1:
                if curr[0].lower() in prepositions:
                    curr[2] = prev[2]  # inherit type from left
    return tokens


def rule_abbrev_resolution(tokens: List[Token], **kwargs) -> List[Token]:
    """Rule 2: Abbreviation resolution.

    If we see FULL_FORM (ABBREV), propagate predictions between the
    full form and abbreviation.
    """
    i = 0
    while i < len(tokens) - 2:
        if tokens[i] and tokens[i][0] == "(":
            abbrev_tok = tokens[i + 1]
            close_tok = tokens[i + 2]
            if (
                abbrev_tok
                and close_tok
                and close_tok[0] == ")"
                and abbrev_tok[0].isupper()
                and abbrev_tok[0].isalpha()
                and len(abbrev_tok[0]) <= 10
            ):
                abbrev = abbrev_tok[0]
                # Look back for full form
                j = i - 1
                full_form = []
                while j >= 0 and tokens[j] and tokens[j][0].isalpha():
                    full_form.insert(0, tokens[j][0])
                    j -= 1

                max_len = 4
                min_len = len(abbrev)
                window = full_form[-max_len:]
                for k in range(len(window) - min_len + 1):
                    candidate = window[k : k + min_len]
                    if "".join(w[0].upper() for w in candidate) == abbrev:
                        start = i - len(window) + k
                        pred_outside = any(
                            tokens[m][2] >= 1 for m in range(start, i)
                        )
                        # Propagate: abbrev → full form
                        if abbrev_tok[2] >= 1 and not pred_outside:
                            for m in range(start, i):
                                tokens[m][2] = abbrev_tok[2]
                        # Propagate: full form → abbrev
                        if pred_outside and abbrev_tok[2] == 0:
                            abbrev_tok[2] = 1
                        break
            i += 2
        i += 1
    return tokens


def rule_pos_filtering(tokens: List[Token], noun_tags: Optional[Set[str]] = None, **kwargs) -> List[Token]:
    """Rule 0: POS filtering for single-token spans.

    Demote singleton entity predictions that are not nouns.
    Requires nltk.
    """
    try:
        import nltk

        # Ensure models are available
        try:
            nltk.data.find("taggers/averaged_perceptron_tagger")
        except LookupError:
            nltk.download("averaged_perceptron_tagger", quiet=True)
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
    except ImportError:
        logger.warning("nltk not installed. Skipping POS filtering rule.")
        return tokens

    if noun_tags is None:
        noun_tags = {"NN", "NNS", "NNP", "NNPS"}

    # Identify singleton entity spans
    is_singleton = [False] * len(tokens)
    for i in range(len(tokens)):
        if tokens[i] is None or tokens[i][2] == 0:
            continue
        prev_pred = tokens[i - 1][2] if i > 0 and tokens[i - 1] is not None else 0
        next_pred = (
            tokens[i + 1][2]
            if i < len(tokens) - 1 and tokens[i + 1] is not None
            else 0
        )
        if prev_pred == 0 and next_pred == 0:
            is_singleton[i] = True

    # Group into sentences for POS tagging
    sentences, sentence_indices = [], []
    cur, cur_idx = [], []
    for idx, tok in enumerate(tokens):
        if tok is None:
            if cur:
                sentences.append(cur)
                sentence_indices.append(cur_idx)
                cur, cur_idx = [], []
        else:
            cur.append(tok)
            cur_idx.append(idx)
    if cur:
        sentences.append(cur)
        sentence_indices.append(cur_idx)

    # Apply POS filter
    for sent, idxs in zip(sentences, sentence_indices):
        words = [tok[0] for tok in sent]
        try:
            pos_tags = nltk.pos_tag(words)
        except Exception:
            continue
        for tok, index, (_, pos) in zip(sent, idxs, pos_tags):
            if tokens[index][2] >= 1 and is_singleton[index] and pos not in noun_tags:
                tokens[index][2] = 0
    return tokens


# Rule dispatch
_RULE_FNS = {
    "span_consistency": rule_span_consistency,
    "prep_bridging": rule_prep_bridging,
    "abbrev_resolution": rule_abbrev_resolution,
    "pos_filtering": rule_pos_filtering,
}


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def _tokens_to_eval_format(
    tokens: List[Token], types: List[str],
) -> Tuple[List[List[str]], List[List[str]]]:
    """Convert flat token list into the format eval.py expects.

    Returns (truth_sentences, pred_sentences) where each sentence is
    a list of BIO tag strings.
    """
    id2type = {0: "O"}
    for i, t in enumerate(types):
        id2type[i + 1] = t

    truth_sents: List[List[str]] = []
    pred_sents: List[List[str]] = []
    cur_truth: List[str] = []
    cur_pred: List[str] = []
    for tok in tokens:
        if tok is None:
            if cur_truth:
                truth_sents.append(cur_truth)
                pred_sents.append(cur_pred)
                cur_truth, cur_pred = [], []
        else:
            cur_truth.append(tok[1])
            p = tok[2]
            cur_pred.append("I-" + id2type[p] if p != 0 else "O")
    if cur_truth:
        truth_sents.append(cur_truth)
        pred_sents.append(cur_pred)
    return truth_sents, pred_sents


def _check_equal(entity1: str, entity2: str, exact: bool = True) -> bool:
    """Check if two entity spans match (from eval.py)."""
    if exact:
        return entity1 == entity2
    t1, s1, e1 = entity1.split("_")
    t2, s2, e2 = entity2.split("_")
    s1, s2, e1, e2 = int(s1), int(s2), int(e1), int(e2)
    if t1 != t2:
        return False
    if s1 > e2 or s2 > e1:
        return False
    return True


def _get_entity(sentences: List[List[str]]) -> List[List[str]]:
    """Extract entity spans from BIO-tagged sentences (from eval.py).

    Returns list of entity strings per sentence: "TYPE_START_END".
    """
    res = []
    for sentence in sentences:
        tmp = []
        pre_type = "O"
        entity_start = 0
        for index, tag in enumerate(sentence + ["O"]):
            etype = tag.split("-")[-1]
            if pre_type == "O":
                if etype != "O":
                    entity_start = index
            else:
                if etype != pre_type:
                    entity_end = index - 1
                    tmp.append("_".join([str(i) for i in [pre_type, entity_start, entity_end]]))
                    entity_start = index
            pre_type = etype
        res.append(tmp)
    return res


def _compute_prf(
    truth_entities: List[List[str]],
    pred_entities: List[List[str]],
    exact: bool,
) -> Tuple[float, float, float]:
    """Compute P/R/F1 from entity lists (from eval.py).

    exact=True  → strict (exact span match)
    exact=False → relaxed (overlapping spans of same type count as match)
    """
    p_tp = t_tp = 0
    for tru, pre in zip(truth_entities, pred_entities):
        p_tmp, t_tmp = [], []
        for t in tru:
            for p in pre:
                if _check_equal(t, p, exact=exact):
                    p_tmp.append(p)
                    t_tmp.append(t)
        p_tp += len(set(p_tmp))
        t_tp += len(set(t_tmp))
    pe_num = sum(len(i) for i in pred_entities)
    te_num = sum(len(i) for i in truth_entities)
    precision = p_tp / pe_num if pe_num else 0
    recall = t_tp / te_num if te_num else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return precision, recall, f1


def evaluate_predictions(tokens: List[Token], types: Optional[List[str]] = None) -> Dict:
    """Evaluate predictions with both strict and relaxed entity matching.

    Uses the same logic as eval.py:
      - Strict:  exact span boundaries must match.
      - Relaxed: overlapping spans of the same type count as match.

    Returns dict with strict_* and relaxed_* metrics.
    """
    if types is None:
        types = sorted(set(
            re.sub(r"^[BIES]-", "", tok[1])
            for tok in tokens
            if tok is not None and tok[1] != "O"
        ))

    truth_sents, pred_sents = _tokens_to_eval_format(tokens, types)
    truth_entities = _get_entity(truth_sents)
    pred_entities = _get_entity(pred_sents)

    sp, sr, sf = _compute_prf(truth_entities, pred_entities, exact=True)
    rp, rr, rf = _compute_prf(truth_entities, pred_entities, exact=False)

    results = {
        "strict_precision": sp, "strict_recall": sr, "strict_f1": sf,
        "relaxed_precision": rp, "relaxed_recall": rr, "relaxed_f1": rf,
    }

    logger.info("Strict:  P=%.4f  R=%.4f  F1=%.4f", sp, sr, sf)
    logger.info("Relaxed: P=%.4f  R=%.4f  F1=%.4f", rp, rr, rf)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# PostProcessor — the main class
# ═══════════════════════════════════════════════════════════════════════════

class PostProcessor:
    """Apply post-processing rules to DS-NER predictions.

    Parameters
    ----------
    input_path : str
        Path to unified prediction file (TOKEN GOLD_TAG PRED_LABEL).
    prepositions : set[str], optional
        Words to bridge in prep_bridging rule.
    noun_tags : set[str], optional
        POS tags considered "noun-like" for pos_filtering rule.
    """

    def __init__(
        self,
        input_path: str,
        prepositions: Optional[Set[str]] = None,
        noun_tags: Optional[Set[str]] = None,
    ):
        self.tokens = read_predictions(input_path)
        self.input_path = input_path
        self.kwargs = {
            "prepositions": prepositions,
            "noun_tags": noun_tags,
        }
        # Count before
        n_pred = sum(1 for t in self.tokens if t is not None and t[2] >= 1)
        logger.info("Loaded %d tokens, %d predicted entities", len(self.tokens), n_pred)

    def run(self, rules: Optional[List[str]] = None):
        """Apply specified rules in order.

        Parameters
        ----------
        rules : list[str], optional
            Rule names to apply, in order. Default: DEFAULT_RULE_ORDER.
        """
        if rules is None:
            rules = DEFAULT_RULE_ORDER

        for rule_name in rules:
            if rule_name not in _RULE_FNS:
                logger.warning("Unknown rule '%s', skipping", rule_name)
                continue
            n_before = sum(1 for t in self.tokens if t is not None and t[2] >= 1)
            self.tokens = _RULE_FNS[rule_name](self.tokens, **self.kwargs)
            n_after = sum(1 for t in self.tokens if t is not None and t[2] >= 1)
            delta = n_after - n_before
            sign = "+" if delta >= 0 else ""
            logger.info("  %-25s %d → %d (%s%d tokens)", rule_name, n_before, n_after, sign, delta)

    def run_all(self):
        """Apply all rules in the default order: 4→1→4→2→4→0."""
        logger.info("Applying all post-processing rules...")
        self.run(DEFAULT_RULE_ORDER)

    def save(self, output_path: str):
        """Save post-processed predictions."""
        write_predictions(self.tokens, output_path)

    def evaluate(self, types: Optional[List[str]] = None) -> Dict:
        """Evaluate current predictions against gold labels."""
        return evaluate_predictions(self.tokens, types)

    def summary(self) -> str:
        """Return a summary of current prediction state."""
        n_total = sum(1 for t in self.tokens if t is not None)
        n_pred = sum(1 for t in self.tokens if t is not None and t[2] >= 1)
        n_sents = sum(1 for t in self.tokens if t is None)
        return f"Tokens: {n_total}, Predicted entities: {n_pred}, Sentences: {n_sents}"


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="DS-NER prediction post-processing",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # -- process --
    p_proc = sub.add_parser("process", help="Post-process a prediction file")
    p_proc.add_argument("--input", required=True, help="Input prediction file")
    p_proc.add_argument("--output", required=True, help="Output file")
    p_proc.add_argument(
        "--rules", nargs="*", default=None,
        help=f"Rules to apply (default: all). Available: {sorted(AVAILABLE_RULES)}",
    )
    p_proc.add_argument("--evaluate", action="store_true", help="Also evaluate after processing")

    # -- collect --
    p_coll = sub.add_parser("collect", help="Collect predictions from a method")
    p_coll.add_argument("--method", required=True, help="Method name")
    p_coll.add_argument("--project_root", required=True, help="Path to original_dsner")
    p_coll.add_argument("--dataset", required=True, help="Dataset name")
    p_coll.add_argument("--gold_file", required=True, help="Gold BIO test file")
    p_coll.add_argument("--pred_file", default=None, help="Explicit prediction file path")
    p_coll.add_argument("--output", required=True, help="Output unified file")
    p_coll.add_argument("--types", nargs="*", default=None, help="Entity types")

    # -- evaluate --
    p_eval = sub.add_parser("evaluate", help="Evaluate a prediction file")
    p_eval.add_argument("--input", required=True, help="Prediction file")
    p_eval.add_argument("--types", nargs="*", default=None, help="Entity types")

    # -- rules --
    sub.add_parser("rules", help="List available rules")

    args = parser.parse_args()

    if args.command == "process":
        pp = PostProcessor(args.input)
        pp.run(rules=args.rules)
        pp.save(args.output)
        if args.evaluate:
            pp.evaluate()

    elif args.command == "collect":
        collect_predictions(
            method=args.method,
            project_root=args.project_root,
            dataset=args.dataset,
            gold_file=args.gold_file,
            pred_file=args.pred_file,
            types=args.types,
            output=args.output,
        )

    elif args.command == "evaluate":
        tokens = read_predictions(args.input)
        evaluate_predictions(tokens, args.types)

    elif args.command == "rules":
        print("\nAvailable post-processing rules:")
        print("-" * 50)
        for rule in AVAILABLE_RULES:
            doc = _RULE_FNS[rule].__doc__
            first_line = doc.strip().split("\n")[0] if doc else ""
            print(f"  {rule:25s} {first_line}")
        print(f"\nDefault order: {' → '.join(DEFAULT_RULE_ORDER)}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
