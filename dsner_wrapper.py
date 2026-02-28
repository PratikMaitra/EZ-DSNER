"""
dsner_wrapper.py — Unified wrapper for 8 Distantly-Supervised NER methods.

Supported methods:
    ATSEN, AutoNER, BOND, CuPuL, DeSERT, MProto, RoSTER, SCDL

Usage:
    from dsner_wrapper import DSNERWrapper

    wrapper = DSNERWrapper(
        method="BOND",
        project_root="/path/to/original_dsner",
        dataset="conll03",
        gpu_ids="0",
    )
    wrapper.train()
    wrapper.evaluate()
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("dsner_wrapper")

# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BaseConfig:
    """Parameters shared across all methods."""
    dataset: str = "conll03"
    gpu_ids: str = "0"
    seed: int = 0
    max_seq_length: int = 128
    train_batch_size: int = 32
    eval_batch_size: int = 32
    learning_rate: float = 2e-5
    num_train_epochs: int = 50
    output_dir: str = "output"
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        extra = d.pop("extra", {})
        d.update(extra)
        return d


@dataclass
class ATSENConfig(BaseConfig):
    """ATSEN-specific configuration (AAAI 2023).

    Dual student-teacher framework with RoBERTa + DistilRoBERTa.
    Entry point: main.py (called via shell scripts like run_qtl.sh)

    Shell scripts take: $1=GPU_ID  $2=DATASET
    """
    # model
    tokenizer_name: str = "roberta-base"
    student1_model_name: str = "roberta-base"
    student2_model_name: str = "distilroberta-base"
    model_type: str = "roberta"
    # training
    learning_rate: float = 1e-5
    warmup_steps: int = 200
    begin_epoch: int = 1
    period: int = 6000            # self-learning update cycle (iterations)
    mean_alpha: float = 0.995     # EMA alpha
    threshold: float = 0.9        # confidence threshold delta
    train_batch_size: int = 8     # per_gpu_train_batch_size
    num_train_epochs: int = 50
    label_mode: str = "soft"      # "soft" or "hard" (use "hard" for Twitter)
    # ATSEN-specific: SVM ensemble distillation
    al: float = 0.8               # alpha for SVM nu computation
    bate: float = 1.0             # weight for distillation loss
    # optimizer
    weight_decay: float = 1e-4
    adam_epsilon: float = 1e-8
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    # logging / saving
    logging_steps: int = 100
    save_steps: int = 100000
    evaluate_during_training: bool = True
    overwrite_output_dir: bool = True


@dataclass
class AutoNERConfig(BaseConfig):
    """AutoNER-specific configuration (EMNLP 2018).

    LSTM-CRF model (NOT transformer-based). Multi-step pipeline:
      1. Compile C++ binary:  make
      2. Encode embeddings:   preprocess_partial_ner/save_emb.py
      3. Generate distant supervision: bin/generate <raw_text> <dict_core> <dict_full> <output>
      4. Encode dataset:      preprocess_partial_ner/encode_folder.py
      5. Train:               train_partial_ner.py
      6. Inference:           test_partial_ner.py

    Shell scripts: autoner_train.sh (full pipeline), autoner_test.sh (inference)
    """
    # data paths (relative to AutoNER dir)
    model_name: str = "BC5CDR"        # used for models/{model_name}/
    raw_text: str = ""                # e.g. data/BC5CDR/raw_text.txt
    dict_core: str = ""               # e.g. data/BC5CDR/dict_core.txt
    dict_full: str = ""               # e.g. data/BC5CDR/dict_full.txt
    embedding_path: str = ""          # e.g. embedding/bio_embedding.txt
    dev_set: str = ""                 # e.g. data/BC5CDR/truth_dev.ck
    test_set: str = ""                # e.g. data/BC5CDR/truth_test.ck
    # model architecture
    hid_dim: int = 300
    word_dim: int = 200
    char_dim: int = 30
    label_dim: int = 50
    layer_num: int = 2
    droprate: float = 0.5
    rnn_layer: str = "Basic"
    rnn_unit: str = "lstm"            # lstm, gru, rnn
    batch_norm: bool = False
    # training
    num_train_epochs: int = 50        # epoch
    learning_rate: float = 0.05       # lr
    optimizer: str = "SGD"            # Adam, Adagrad, Adadelta, SGD
    clip: float = 5.0
    sample_ratio: float = 1.0
    batch_token_number: int = 3000
    tolerance: int = 5
    lr_decay: float = 0.05
    interval: int = 30               # logging interval
    check: int = 1000                # evaluation interval
    # inference
    threshold: float = 0.0           # chunking threshold for test


@dataclass
class BONDConfig(BaseConfig):
    """BOND-specific configuration.

    Two entry points:
      - run_ner.py                  (Stage I: baseline distant training)
      - run_self_training_ner.py    (Stage II: self-training with mean teacher)

    Set self_training=True to use Stage II (default), False for Stage I only.
    """
    self_training: bool = True
    # model
    model_type: str = "roberta"  # roberta, bert, biobert, distilbert, xlmroberta, camembert
    model_name_or_path: str = "roberta-base"
    tokenizer_name: str = ""
    config_name: str = ""
    cache_dir: str = ""
    do_lower_case: bool = False
    # actions
    do_train: bool = True
    do_eval: bool = True
    do_predict: bool = False
    evaluate_during_training: bool = True
    # training
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    adam_epsilon: float = 1e-8
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    max_grad_norm: float = 1.0
    num_train_epochs: int = 3
    max_steps: int = -1
    warmup_steps: int = 0
    train_batch_size: int = 8    # per_gpu_train_batch_size
    eval_batch_size: int = 8     # per_gpu_eval_batch_size
    gradient_accumulation_steps: int = 1
    # logging / saving
    logging_steps: int = 50
    save_steps: int = 50
    overwrite_output_dir: bool = True
    overwrite_cache: bool = False
    # mean teacher
    mt: int = 0                    # enable mean teacher (1=on, 0=off)
    mt_updatefreq: int = 1
    mt_class: str = "kl"           # kl, smart, prob, logit, distill
    mt_lambda: float = 1.0
    mt_rampup: int = 300
    mt_alpha1: float = 0.99
    mt_alpha2: float = 0.995
    mt_beta: float = 10.0
    mt_avg: str = "exponential"    # exponential, simple, double_ema
    mt_loss_type: str = "logits"   # logits, embeds
    # virtual adversarial training
    vat: int = 0
    vat_eps: float = 1e-3
    vat_lambda: float = 1.0
    vat_beta: float = 1.0
    vat_loss_type: str = "logits"
    # self-training (Stage II only)
    self_training_reinit: int = 0
    self_training_begin_step: int = 900
    self_training_label_mode: str = "hard"  # hard, soft
    self_training_period: int = 878
    self_training_hp_label: float = 0.0
    self_training_ensemble_label: int = 0
    # weak data
    load_weak: bool = False
    remove_labels_from_weak: bool = False
    rep_train_against_weak: int = 1


@dataclass
class CuPuLConfig(BaseConfig):
    """CuPuL-specific configuration.

    Three entry points:
      1. dictionary_match.py   — preprocessing: dictionary-based distant labeling
      2. train.py              — training pipeline (noise-robust → curriculum → self-training)
      3. predict.py            — inference from saved checkpoint

    Training pipeline (train.py --do_train) runs sequentially:
      - num_models ensemble training rounds
      - Token hardness scoring + ensemble prediction
      - Curriculum training
      - Self-training
    """
    # model
    pretrained_model: str = "roberta-base"
    tag_scheme: str = "io"           # io or iob
    # directories
    temp_dir: str = "temp"
    # actions
    do_train: bool = True
    do_eval: bool = True
    eval_on: str = "test"            # valid or test
    train_on: str = "train"
    # loss
    loss_type: str = "MAE"           # MAE or CE
    # training: noise-robust phase
    learning_rate: float = 2e-5      # train_lr
    train_batch_size: int = 32
    eval_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    num_train_epochs: int = 1        # train_epochs
    noise_train_update_interval: int = 200
    self_train_update_interval: int = 100
    warmup_proportion: float = 0.1
    weight_decay: float = 0.01
    # training: curriculum phase
    curriculum_train_lr: float = 1e-5
    curriculum_train_epochs: int = 5
    curriculum_train_sub_epochs: int = 2
    num_models: int = 5              # ensemble count
    # training: self-training phase
    self_train_lr: float = 1e-5
    self_train_epochs: int = 10
    student1_lr: float = 1e-5
    student2_lr: float = 1e-5
    # dropout / thresholds
    drop_other: float = 0.1
    drop_entity: float = 0.1
    entity_threshold: float = 0.8
    ratio: float = 0.1
    m: float = 10.0
    # dictionary matching (preprocessing)
    dict_dir: str = "./dictionaries"
    dict_names: Dict[str, str] = field(default_factory=dict)  # e.g. {"Trait": "default_dict.txt"}
    tag2idx: Dict[str, int] = field(default_factory=dict)     # e.g. {"O": 0, "Trait": 1}


@dataclass
class DeSERTConfig(BaseConfig):
    """DeSERT-specific configuration.

    DeSERT uses a dual student-teacher framework with RoBERTa + DistilRoBERTa.
    Two programs are available:
      - run_script_debias_bin_basic.py        (main training)
      - run_script_debias_bin_basic_finetune.py (finetuning from checkpoint)

    Shell scripts (run_script_qtl.sh, run_script_conll.sh, etc.) take:
      $1 = GPU_ID, $2 = DATASET, $3 = PROGRAM (python script path)
    """
    # model
    tokenizer_name: str = "roberta-base"
    student1_model_name: str = "roberta-base"
    student2_model_name: str = "distilroberta-base"
    model_type: str = "roberta"
    # training
    learning_rate: float = 1e-5
    warmup_steps: int = 200
    begin_epoch: int = 1
    period: int = 6000          # self-learning update cycle (iterations)
    mean_alpha: float = 0.995   # EMA alpha
    threshold: float = 0.95     # confidence threshold delta
    train_batch_size: int = 16
    num_train_epochs: int = 50
    label_mode: str = "soft"
    wce_weight: float = 1.0
    begin_coguess: int = 6
    # optimizer
    weight_decay: float = 1e-4
    adam_epsilon: float = 1e-8
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    # logging / saving
    logging_steps: int = 100
    save_steps: int = 100000
    evaluate_during_training: bool = True
    overwrite_output_dir: bool = True
    # program selection
    program: str = "run_script_debias_bin_basic.py"  # or finetune variant
    finetune: bool = False  # if True, uses the finetune program instead


@dataclass
class MProtoConfig(BaseConfig):
    """MProto-specific configuration.

    Uses the `alchemy` framework with TOML config files.
    Entry point: python -m scripts.train_and_test <config.toml> [options]

    All model/training/data configuration lives in .toml files under cfg/.
    Examples:
      cfg/conll03-dict/mproto/train-p3-1.0.toml
      cfg/bc5cdr-dict/mproto/train-p3-1.0.toml

    The train_and_test script:
      1. Trains the model using alchemy.run()
      2. Automatically runs test using the best checkpoint
    """
    # Path to .toml config file (relative to mproto dir)
    config_path: str = ""             # e.g. cfg/conll03-dict/mproto/train-p3-1.0.toml
    # Device (GPU IDs as list, e.g. [0] or [0,1])
    device: str = ""                  # passed as --device; empty = auto
    # alchemy options
    user_dir: str = "src"
    desc: str = ""
    debug: bool = False


@dataclass
class RoSTERConfig(BaseConfig):
    """RoSTER-specific configuration (EMNLP 2021).

    Entry point: src/train.py
    Shell scripts (run_qtl.sh, run_conll.sh, etc.) set GPU via env var,
    corpus name is hardcoded per script.

    Training pipeline runs three phases sequentially:
      1. Noise-robust training
      2. Ensemble model training
      3. Self-training
    """
    # model
    pretrained_model: str = "roberta-base"
    tag_scheme: str = "io"           # io or iob
    # directories
    temp_dir: str = ""               # auto-set to tmp_{corpus}_{seed} if empty
    # actions
    do_train: bool = True
    do_eval: bool = True
    eval_on: str = "test"            # test or valid
    # training: phase 1 (noise-robust)
    noise_train_lr: float = 3e-5
    noise_train_epochs: int = 3
    noise_train_update_interval: int = 200
    q: float = 0.7
    tau: float = 0.7
    # training: phase 2 (ensemble)
    ensemble_train_lr: float = 1e-5
    ensemble_train_epochs: int = 2
    num_models: int = 5
    # training: phase 3 (self-training)
    self_train_lr: float = 5e-7
    self_train_epochs: int = 5
    self_train_update_interval: int = 100
    # shared training params
    train_batch_size: int = 32
    eval_batch_size: int = 64
    gradient_accumulation_steps: int = 1
    dropout: float = 0.1
    warmup_proportion: float = 0.1
    weight_decay: float = 0.01


@dataclass
class SCDLConfig(BaseConfig):
    """SCDL-specific configuration (EMNLP 2021).

    Dual student-teacher framework with RoBERTa + DistilRoBERTa.
    Entry point: run_script.py (called via run_script.sh)
    Shell script takes: $1=GPU_ID  $2=DATASET
    Based on BOND codebase.
    """
    # model
    tokenizer_name: str = "roberta-base"
    student1_model_name: str = "roberta-base"
    student2_model_name: str = "distilroberta-base"
    model_type: str = "roberta"
    # training
    learning_rate: float = 2e-5
    warmup_steps: int = 200
    begin_epoch: int = 6
    period: int = 3200            # self-learning update cycle (iterations)
    mean_alpha: float = 0.995     # EMA alpha
    threshold: float = 0.9        # confidence threshold delta
    train_batch_size: int = 16    # per_gpu_train_batch_size
    num_train_epochs: int = 50
    label_mode: str = "soft"      # "soft" or "hard" (use "hard" for Twitter)
    # optimizer
    weight_decay: float = 1e-4
    adam_epsilon: float = 1e-8
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    # logging / saving
    logging_steps: int = 100
    save_steps: int = 100000
    evaluate_during_training: bool = True
    overwrite_output_dir: bool = True


# Map from method name → config class
CONFIG_REGISTRY: Dict[str, type] = {
    "ATSEN": ATSENConfig,
    "AutoNER": AutoNERConfig,
    "BOND": BONDConfig,
    "CuPuL": CuPuLConfig,
    "DeSERT": DeSERTConfig,
    "MProto": MProtoConfig,
    "RoSTER": RoSTERConfig,
    "SCDL": SCDLConfig,
}

# ---------------------------------------------------------------------------
# Method runners (strategy pattern)
# ---------------------------------------------------------------------------

class BaseRunner(ABC):
    """Abstract runner that each method must implement."""

    def __init__(self, project_root: str, method_dir: str, config: BaseConfig):
        self.project_root = Path(project_root)
        self.method_dir = self.project_root / method_dir
        self.config = config
        self._validate_paths()

    def _validate_paths(self):
        if not self.method_dir.exists():
            raise FileNotFoundError(
                f"Method directory not found: {self.method_dir}\n"
                f"Available: {[p.name for p in self.project_root.iterdir() if p.is_dir()]}"
            )

    def _run_command(
        self,
        cmd: Union[str, List[str]],
        cwd: Optional[Path] = None,
        env_extra: Optional[Dict[str, str]] = None,
        shell: bool = True,
    ) -> subprocess.CompletedProcess:
        """Execute a shell command with logging."""
        cwd = cwd or self.method_dir
        env = os.environ.copy()
        if self.config.gpu_ids:
            env["CUDA_VISIBLE_DEVICES"] = self.config.gpu_ids
        if env_extra:
            env.update(env_extra)

        cmd_str = cmd if isinstance(cmd, str) else " ".join(cmd)
        logger.info("Running command:\n  cwd=%s\n  cmd=%s", cwd, cmd_str)

        result = subprocess.run(
            cmd_str,
            cwd=str(cwd),
            shell=shell,
            env=env,
            capture_output=False,
        )
        if result.returncode != 0:
            logger.error("Command failed with return code %d", result.returncode)
        return result

    def _resolve_dataset_name(self, subdir: str = "data") -> str:
        """Resolve dataset name to match the actual folder on disk.

        Methods expect --dataset_name to match the folder exactly (e.g. 'QTL'
        not 'qtl'). This does a case-insensitive lookup in both
        method_dir/subdir/ and project_root/subdir/.
        """
        target = self.config.dataset.lower()
        for base in [self.method_dir, self.project_root]:
            data_dir = base / subdir
            if data_dir.is_dir():
                for d in data_dir.iterdir():
                    if d.is_dir() and d.name.lower() == target:
                        return d.name
        return self.config.dataset

    @abstractmethod
    def train(self) -> subprocess.CompletedProcess:
        ...

    @abstractmethod
    def evaluate(self) -> subprocess.CompletedProcess:
        ...

    def predict(self, input_path: Optional[str] = None) -> subprocess.CompletedProcess:
        """Default predict falls back to evaluate. Override if method has separate predict."""
        logger.warning(
            "%s does not have a separate predict command; falling back to evaluate.",
            self.__class__.__name__,
        )
        return self.evaluate()


# ---- ATSEN ----------------------------------------------------------------

class ATSENRunner(BaseRunner):
    """Runner for ATSEN (AAAI 2023).

    Shell scripts (run_qtl.sh, run_conll03.sh, etc.) take: $1=GPU_ID  $2=DATASET
    Entry point: main.py
    """

    DATASET_SCRIPT_MAP = {
        "conll03": "run_conll03.sh",
        "conll": "run_conll03.sh",
        "ontonotes": "run_ontonotes5.sh",
        "twitter": "run_twitter.sh",
        "tweet": "run_twitter.sh",
        "webpage": "run_webpage_added.sh",
        "bc5cdr": "run_bc5cdr.sh",
        "qtl": "run_qtl.sh",
    }

    def _get_script(self) -> Optional[str]:
        ds = self.config.dataset.lower()
        for key, script in self.DATASET_SCRIPT_MAP.items():
            if key in ds:
                if (self.method_dir / script).exists():
                    return script
        return None

    def train(self):
        cfg: ATSENConfig = self.config  # type: ignore
        script = self._get_script()
        dataset = self._resolve_dataset_name()
        if script:
            cmd = f"bash {script} {cfg.gpu_ids} {dataset}"
            return self._run_command(cmd)
        else:
            return self._run_command(self._build_python_cmd())

    def evaluate(self):
        # ATSEN evaluates during training (--evaluate_during_training)
        return self.train()

    def _build_python_cmd(self) -> str:
        """Build the full python command matching the shell script structure."""
        cfg: ATSENConfig = self.config  # type: ignore
        project_root = str(self.method_dir)
        data_root = str(self.method_dir / "dataset")
        output_dir = str(self.method_dir / "ptms" / cfg.dataset)

        parts = [
            f"python3 -u main.py",
            f"--data_dir {data_root}",
            f"--student1_model_name_or_path {cfg.student1_model_name}",
            f"--student2_model_name_or_path {cfg.student2_model_name}",
            f"--output_dir {output_dir}",
            f"--tokenizer_name {cfg.tokenizer_name}",
            f"--cache_dir {project_root}/cached_models",
            f"--max_seq_length {cfg.max_seq_length}",
            f"--learning_rate {cfg.learning_rate}",
            f"--weight_decay {cfg.weight_decay}",
            f"--adam_epsilon {cfg.adam_epsilon}",
            f"--adam_beta1 {cfg.adam_beta1}",
            f"--adam_beta2 {cfg.adam_beta2}",
            f"--max_grad_norm {cfg.max_grad_norm}",
            f"--num_train_epochs {cfg.num_train_epochs}",
            f"--warmup_steps {cfg.warmup_steps}",
            f"--per_gpu_train_batch_size {cfg.train_batch_size}",
            f"--per_gpu_eval_batch_size {cfg.eval_batch_size}",
            f"--gradient_accumulation_steps {cfg.gradient_accumulation_steps}",
            f"--logging_steps {cfg.logging_steps}",
            f"--save_steps {cfg.save_steps}",
            f"--seed {cfg.seed}",
            f"--mean_alpha {cfg.mean_alpha}",
            f"--self_learning_label_mode {cfg.label_mode}",
            f"--self_learning_period {cfg.period}",
            f"--model_type {cfg.model_type}",
            f"--begin_epoch {cfg.begin_epoch}",
            f"--dataset {self._resolve_dataset_name()}",
            f"--threshold {cfg.threshold}",
            f"--al {cfg.al}",
            f"--bate {cfg.bate}",
            "--do_train",
        ]
        if cfg.evaluate_during_training:
            parts.append("--evaluate_during_training")
        if cfg.overwrite_output_dir:
            parts.append("--overwrite_output_dir")
        for k, v in cfg.extra.items():
            if isinstance(v, bool):
                if v:
                    parts.append(f"--{k}")
            else:
                parts.append(f"--{k} {v}")
        return " ".join(parts)


# ---- AutoNER --------------------------------------------------------------

class AutoNERRunner(BaseRunner):
    """Runner for AutoNER (EMNLP 2018).

    LSTM-CRF model with multi-step pipeline:
      autoner_train.sh: make → encode_emb → generate DS → encode_data → train
      autoner_test.sh:  encode_test → test

    Can also run individual steps via dedicated methods.
    """

    def _resolve_paths(self) -> dict:
        """Resolve data paths, auto-detecting from model_name if not set."""
        cfg: AutoNERConfig = self.config  # type: ignore
        mn = cfg.model_name
        base = str(self.method_dir)
        return {
            "model_root": f"{base}/models/{mn}",
            "raw_text": cfg.raw_text or f"data/{mn}/raw_text.txt",
            "dict_core": cfg.dict_core or f"data/{mn}/dict_core.txt",
            "dict_full": cfg.dict_full or f"data/{mn}/dict_full.txt",
            "embedding": cfg.embedding_path or f"embedding/bio_embedding.txt",
            "dev_set": cfg.dev_set or f"data/{mn}/truth_dev.ck",
            "test_set": cfg.test_set or f"data/{mn}/truth_test.ck",
        }

    def train(self):
        """Run full training pipeline (autoner_train.sh)."""
        cfg: AutoNERConfig = self.config  # type: ignore
        script = self.method_dir / "autoner_train.sh"
        if script.exists():
            return self._run_command("bash autoner_train.sh")
        else:
            # Build manual pipeline
            return self._run_train_pipeline()

    def evaluate(self):
        """Run inference pipeline (autoner_test.sh)."""
        cfg: AutoNERConfig = self.config  # type: ignore
        script = self.method_dir / "autoner_test.sh"
        if script.exists():
            return self._run_command("bash autoner_test.sh")
        else:
            return self._run_test_pipeline()

    def predict(self, input_path: Optional[str] = None):
        return self.evaluate()

    def _run_train_pipeline(self) -> subprocess.CompletedProcess:
        """Replicate autoner_train.sh as python commands."""
        cfg: AutoNERConfig = self.config  # type: ignore
        paths = self._resolve_paths()
        mr = paths["model_root"]

        # Step 1: Compile
        self._run_command("make")

        # Step 2: Encode embeddings (if not already done)
        emb_pk = f"{mr}/embedding.pk"
        self._run_command(
            f"python preprocess_partial_ner/save_emb.py "
            f"--input_embedding {paths['embedding']} --output_embedding {emb_pk}"
        )

        # Step 3: Generate distant supervision
        training_set = f"{mr}/annotations.ck"
        self._run_command(
            f"bin/generate {paths['raw_text']} {paths['dict_core']} {paths['dict_full']} {training_set}"
        )

        # Step 4: Encode dataset
        encoded_dir = f"{mr}/encoded_data"
        self._run_command(f"mkdir -p {encoded_dir}")
        self._run_command(
            f"python preprocess_partial_ner/encode_folder.py "
            f"--input_train {training_set} "
            f"--input_testa {paths['dev_set']} "
            f"--input_testb {paths['test_set']} "
            f"--pre_word_emb {emb_pk} "
            f"--output_folder {encoded_dir}/"
        )

        # Step 5: Train
        checkpoint_dir = f"{mr}/checkpoint/"
        checkpoint_name = "autoner"
        return self._run_command(self._build_train_cmd(
            checkpoint_dir, checkpoint_name,
            f"{encoded_dir}/test.pk", f"{encoded_dir}/train_0.pk"
        ))

    def _build_train_cmd(self, cp_root: str, cp_name: str, eval_ds: str, train_ds: str) -> str:
        cfg: AutoNERConfig = self.config  # type: ignore
        parts = [
            "python train_partial_ner.py",
            f"--cp_root {cp_root}",
            f"--checkpoint_name {cp_name}",
            f"--eval_dataset {eval_ds}",
            f"--train_dataset {train_ds}",
            f"--update {cfg.optimizer}",
            f"--lr {cfg.learning_rate}",
            f"--hid_dim {cfg.hid_dim}",
            f"--word_dim {cfg.word_dim}",
            f"--char_dim {cfg.char_dim}",
            f"--label_dim {cfg.label_dim}",
            f"--layer_num {cfg.layer_num}",
            f"--droprate {cfg.droprate}",
            f"--sample_ratio {cfg.sample_ratio}",
            f"--epoch {cfg.num_train_epochs}",
            f"--clip {cfg.clip}",
            f"--rnn_layer {cfg.rnn_layer}",
            f"--rnn_unit {cfg.rnn_unit}",
            f"--batch_token_number {cfg.batch_token_number}",
            f"--tolerance {cfg.tolerance}",
            f"--lr_decay {cfg.lr_decay}",
            f"--interval {cfg.interval}",
            f"--check {cfg.check}",
        ]
        if cfg.seed != 0:
            parts.append(f"--seed {cfg.seed}")
        if cfg.batch_norm:
            parts.append("--batch_norm")
        for k, v in cfg.extra.items():
            if isinstance(v, bool):
                if v:
                    parts.append(f"--{k}")
            else:
                parts.append(f"--{k} {v}")
        return " ".join(parts)

    def _run_test_pipeline(self) -> subprocess.CompletedProcess:
        """Replicate autoner_test.sh."""
        cfg: AutoNERConfig = self.config  # type: ignore
        paths = self._resolve_paths()
        mr = paths["model_root"]
        checkpoint = f"{mr}/checkpoint/autoner/"

        # Step 1: Encode test data
        self._run_command(
            f"python preprocess_partial_ner/encode_test.py "
            f"--input_data {paths['raw_text']} "
            f"--checkpoint_folder {checkpoint} "
            f"--output_file {mr}/encoded_test.pk"
        )

        # Step 2: Run inference
        return self._run_command(
            f"python test_partial_ner.py "
            f"--input_corpus {mr}/encoded_test.pk "
            f"--checkpoint_folder {checkpoint} "
            f"--output_text {mr}/decoded.txt "
            f"--hid_dim {cfg.hid_dim} "
            f"--word_dim {cfg.word_dim} "
            f"--droprate {cfg.droprate} "
            f"--threshold {cfg.threshold}"
        )


# ---- BOND ------------------------------------------------------------------

class BONDRunner(BaseRunner):
    """
    Runner for BOND (KDD 2020).

    Two entry points:
      - run_ner.py                  → Stage I (baseline)
      - run_self_training_ner.py    → Stage II (self-training)

    Shell scripts in scripts/ dir (e.g. conll_self_training.sh, conll_baseline.sh)
    are thin wrappers around these python files.
    """

    DATASET_SCRIPT_MAP = {
        "conll03": ("conll_self_training.sh", "conll_baseline.sh"),
        "conll": ("conll_self_training.sh", "conll_baseline.sh"),
        "tweet": ("tweet_self_training.sh", "tweet_baseline.sh"),
        "twitter": ("tweet_self_training.sh", "tweet_baseline.sh"),
        "ontonotes": ("onto_self_training.sh", "onto_baseline.sh"),
        "webpage": ("webpage_self_training.sh", "webpage_baseline.sh"),
        "wikigold": ("wikigold_self_training.sh", "wikigold_baseline.sh"),
    }

    def _get_scripts(self):
        ds = self.config.dataset.lower()
        for key, scripts in self.DATASET_SCRIPT_MAP.items():
            if key in ds:
                return scripts
        return None

    def _get_program(self) -> str:
        cfg: BONDConfig = self.config  # type: ignore
        if cfg.self_training:
            return "run_self_training_ner.py"
        return "run_ner.py"

    def train(self):
        cfg: BONDConfig = self.config  # type: ignore
        # Try shell scripts first
        scripts = self._get_scripts()
        if scripts:
            script = scripts[0] if cfg.self_training else scripts[1]
            for search_dir in ["scripts", "."]:
                script_path = self.method_dir / search_dir / script
                if script_path.exists():
                    return self._run_command(f"bash {search_dir}/{script}" if search_dir != "." else f"bash {script}")
        # Fallback: build python command from config
        return self._run_command(self._build_python_cmd(do_train=True, do_eval=cfg.do_eval))

    def evaluate(self):
        cfg: BONDConfig = self.config  # type: ignore
        return self._run_command(self._build_python_cmd(do_train=False, do_eval=True))

    def predict(self, input_path: Optional[str] = None):
        return self._run_command(self._build_python_cmd(do_train=False, do_eval=False, do_predict=True))

    def _build_python_cmd(self, do_train=True, do_eval=True, do_predict=False) -> str:
        cfg: BONDConfig = self.config  # type: ignore
        program = self._get_program()
        data_dir = str(self.method_dir / "dataset" / cfg.dataset)
        output_dir = cfg.output_dir if cfg.output_dir != "output" else str(self.method_dir / "output" / cfg.dataset)

        parts = [
            f"python {program}",
            f"--data_dir {data_dir}",
            f"--model_type {cfg.model_type}",
            f"--model_name_or_path {cfg.model_name_or_path}",
            f"--output_dir {output_dir}",
            f"--max_seq_length {cfg.max_seq_length}",
            f"--learning_rate {cfg.learning_rate}",
            f"--weight_decay {cfg.weight_decay}",
            f"--adam_epsilon {cfg.adam_epsilon}",
            f"--adam_beta1 {cfg.adam_beta1}",
            f"--adam_beta2 {cfg.adam_beta2}",
            f"--max_grad_norm {cfg.max_grad_norm}",
            f"--num_train_epochs {cfg.num_train_epochs}",
            f"--warmup_steps {cfg.warmup_steps}",
            f"--per_gpu_train_batch_size {cfg.train_batch_size}",
            f"--per_gpu_eval_batch_size {cfg.eval_batch_size}",
            f"--gradient_accumulation_steps {cfg.gradient_accumulation_steps}",
            f"--logging_steps {cfg.logging_steps}",
            f"--save_steps {cfg.save_steps}",
            f"--seed {cfg.seed}",
        ]
        # Optional string params
        if cfg.tokenizer_name:
            parts.append(f"--tokenizer_name {cfg.tokenizer_name}")
        if cfg.config_name:
            parts.append(f"--config_name {cfg.config_name}")
        if cfg.cache_dir:
            parts.append(f"--cache_dir {cfg.cache_dir}")
        if cfg.max_steps > 0:
            parts.append(f"--max_steps {cfg.max_steps}")
        # Boolean flags
        if do_train:
            parts.append("--do_train")
        if do_eval:
            parts.append("--do_eval")
        if do_predict:
            parts.append("--do_predict")
        if cfg.evaluate_during_training:
            parts.append("--evaluate_during_training")
        if cfg.do_lower_case:
            parts.append("--do_lower_case")
        if cfg.overwrite_output_dir:
            parts.append("--overwrite_output_dir")
        if cfg.overwrite_cache:
            parts.append("--overwrite_cache")
        # Mean teacher params
        if cfg.mt:
            parts += [
                f"--mt {cfg.mt}",
                f"--mt_updatefreq {cfg.mt_updatefreq}",
                f"--mt_class {cfg.mt_class}",
                f"--mt_lambda {cfg.mt_lambda}",
                f"--mt_rampup {cfg.mt_rampup}",
                f"--mt_alpha1 {cfg.mt_alpha1}",
                f"--mt_alpha2 {cfg.mt_alpha2}",
                f"--mt_beta {cfg.mt_beta}",
                f"--mt_avg {cfg.mt_avg}",
                f"--mt_loss_type {cfg.mt_loss_type}",
            ]
        # VAT params
        if cfg.vat:
            parts += [
                f"--vat {cfg.vat}",
                f"--vat_eps {cfg.vat_eps}",
                f"--vat_lambda {cfg.vat_lambda}",
                f"--vat_beta {cfg.vat_beta}",
                f"--vat_loss_type {cfg.vat_loss_type}",
            ]
        # Self-training params (only for Stage II)
        if cfg.self_training:
            parts += [
                f"--self_training_reinit {cfg.self_training_reinit}",
                f"--self_training_begin_step {cfg.self_training_begin_step}",
                f"--self_training_label_mode {cfg.self_training_label_mode}",
                f"--self_training_period {cfg.self_training_period}",
                f"--self_training_hp_label {cfg.self_training_hp_label}",
                f"--self_training_ensemble_label {cfg.self_training_ensemble_label}",
            ]
        # Weak data
        if cfg.load_weak:
            parts.append("--load_weak")
        if cfg.remove_labels_from_weak:
            parts.append("--remove_labels_from_weak")
        if cfg.rep_train_against_weak != 1:
            parts.append(f"--rep_train_against_weak {cfg.rep_train_against_weak}")
        # Merge any extra params
        for k, v in cfg.extra.items():
            if isinstance(v, bool):
                if v:
                    parts.append(f"--{k}")
            else:
                parts.append(f"--{k} {v}")
        return " ".join(parts)


# ---- CuPuL ----------------------------------------------------------------

class CuPuLRunner(BaseRunner):
    """
    Runner for CuPuL (COLING 2025).

    Three entry points:
      1. dictionary_match.py  — preprocessing (dictionary-based distant labeling)
      2. train.py             — full training pipeline
      3. predict.py           — inference from checkpoint

    train.py --do_train internally runs:
      - num_models rounds of ensemble training (noise-robust)
      - Token hardness scoring + ensemble prediction
      - Curriculum training
      - Self-training
    """

    def _build_common_args(self) -> List[str]:
        """Build CLI args shared between train.py and predict.py."""
        cfg: CuPuLConfig = self.config  # type: ignore
        dataset_name = self._resolve_dataset_name()
        parts = [
            f"--dataset_name {dataset_name}",
            f"--pretrained_model {cfg.pretrained_model}",
            f"--max_seq_length {cfg.max_seq_length}",
            f"--tag_scheme {cfg.tag_scheme}",
            f"--loss_type {cfg.loss_type}",
            f"--train_batch_size {cfg.train_batch_size}",
            f"--eval_batch_size {cfg.eval_batch_size}",
            f"--gradient_accumulation_steps {cfg.gradient_accumulation_steps}",
            f"--train_lr {cfg.learning_rate}",
            f"--curriculum_train_lr {cfg.curriculum_train_lr}",
            f"--train_epochs {cfg.num_train_epochs}",
            f"--curriculum_train_epochs {cfg.curriculum_train_epochs}",
            f"--curriculum_train_sub_epochs {cfg.curriculum_train_sub_epochs}",
            f"--num_models {cfg.num_models}",
            f"--warmup_proportion {cfg.warmup_proportion}",
            f"--weight_decay {cfg.weight_decay}",
            f"--drop_other {cfg.drop_other}",
            f"--drop_entity {cfg.drop_entity}",
            f"--seed {cfg.seed}",
            f"--self_train_lr {cfg.self_train_lr}",
            f"--self_train_epochs {cfg.self_train_epochs}",
            f"--student1_lr {cfg.student1_lr}",
            f"--student2_lr {cfg.student2_lr}",
            f"--entity_threshold {cfg.entity_threshold}",
            f"--ratio {cfg.ratio}",
            f"--m {cfg.m}",
            f"--noise_train_update_interval {cfg.noise_train_update_interval}",
            f"--self_train_update_interval {cfg.self_train_update_interval}",
        ]
        if cfg.output_dir != "output":
            parts.append(f"--output_dir {cfg.output_dir}")
        if cfg.temp_dir != "temp":
            parts.append(f"--temp_dir {cfg.temp_dir}")
        if cfg.eval_on != "test":
            parts.append(f"--eval_on {cfg.eval_on}")
        if cfg.train_on != "train":
            parts.append(f"--train_on {cfg.train_on}")
        # Merge any extra params
        for k, v in cfg.extra.items():
            if isinstance(v, bool):
                if v:
                    parts.append(f"--{k}")
            else:
                parts.append(f"--{k} {v}")
        return parts

    def dict_match(self) -> subprocess.CompletedProcess:
        """Run dictionary matching preprocessing (dictionary_match.py)."""
        cmd = "python dictionary_match.py"
        return self._run_command(cmd)

    def train(self):
        cfg: CuPuLConfig = self.config  # type: ignore
        parts = ["python train.py"]
        if cfg.do_train:
            parts.append("--do_train")
        if cfg.do_eval:
            parts.append("--do_eval")
        parts += self._build_common_args()
        return self._run_command(" ".join(parts))

    def evaluate(self):
        cfg: CuPuLConfig = self.config  # type: ignore
        parts = ["python train.py", "--do_eval"]
        parts += self._build_common_args()
        return self._run_command(" ".join(parts))

    def predict(self, input_path: Optional[str] = None):
        cfg: CuPuLConfig = self.config  # type: ignore
        parts = ["python predict.py"]
        parts += self._build_common_args()
        return self._run_command(" ".join(parts))


# ---- DeSERT ----------------------------------------------------------------

class DeSERTRunner(BaseRunner):
    """
    DeSERT runner.

    DeSERT uses shell scripts that take: $1=GPU_ID  $2=DATASET  $3=PROGRAM
    where PROGRAM is one of:
      - run_script_debias_bin_basic.py          (main training)
      - run_script_debias_bin_basic_finetune.py (finetuning)

    Available shell scripts:
      - run_script_qtl.sh
      - run_script_conll.sh
      - run_script_ontonotes.sh
      - run.sh  (generic)
    """

    DATASET_SCRIPT_MAP = {
        "qtl": "run_script_qtl.sh",
        "conll": "run_script_conll.sh",
        "ontonotes": "run_script_ontonotes.sh",
    }

    def _get_program(self) -> str:
        cfg: DeSERTConfig = self.config  # type: ignore
        if cfg.finetune:
            return "run_script_debias_bin_basic_finetune.py"
        return cfg.program

    def _get_shell_script(self) -> Optional[str]:
        ds = self.config.dataset.lower()
        for key, script in self.DATASET_SCRIPT_MAP.items():
            if key in ds:
                if (self.method_dir / script).exists():
                    return script
        # Try generic run.sh
        if (self.method_dir / "run.sh").exists():
            return "run.sh"
        return None

    def train(self):
        cfg: DeSERTConfig = self.config  # type: ignore
        program = self._get_program()
        script = self._get_shell_script()

        if script:
            # Shell script mode: bash <script> <GPU_ID> <DATASET> <PROGRAM>
            cmd = f"bash {script} {cfg.gpu_ids} {self._resolve_dataset_name()} {program}"
            return self._run_command(cmd)
        else:
            # Direct python command mode (build from config)
            return self._run_command(self._build_python_cmd(program))

    def evaluate(self):
        # DeSERT evaluates during training (--evaluate_during_training)
        return self.train()

    def _build_python_cmd(self, program: str) -> str:
        """Build the full python command matching the shell script's structure."""
        cfg: DeSERTConfig = self.config  # type: ignore
        project_root = str(self.method_dir)
        data_root = str(self.method_dir / "dataset")
        output_dir = str(self.method_dir / "ckpt" / cfg.dataset)

        parts = [
            f"python3 -u {program}",
            f"--data_dir {data_root}",
            f"--student1_model_name_or_path {cfg.student1_model_name}",
            f"--student2_model_name_or_path {cfg.student2_model_name}",
            f"--output_dir {output_dir}",
            f"--tokenizer_name {cfg.tokenizer_name}",
            f"--cache_dir {project_root}/cached_models",
            f"--max_seq_length {cfg.max_seq_length}",
            f"--learning_rate {cfg.learning_rate}",
            f"--weight_decay {cfg.weight_decay}",
            f"--adam_epsilon {cfg.adam_epsilon}",
            f"--adam_beta1 {cfg.adam_beta1}",
            f"--adam_beta2 {cfg.adam_beta2}",
            f"--max_grad_norm {cfg.max_grad_norm}",
            f"--num_train_epochs {cfg.num_train_epochs}",
            f"--warmup_steps {cfg.warmup_steps}",
            f"--per_gpu_train_batch_size {cfg.train_batch_size}",
            f"--per_gpu_eval_batch_size {cfg.eval_batch_size}",
            f"--gradient_accumulation_steps {cfg.gradient_accumulation_steps}",
            f"--logging_steps {cfg.logging_steps}",
            f"--save_steps {cfg.save_steps}",
            f"--seed {cfg.seed}",
            f"--mean_alpha {cfg.mean_alpha}",
            f"--self_learning_label_mode {cfg.label_mode}",
            f"--self_learning_period {cfg.period}",
            f"--model_type {cfg.model_type}",
            f"--begin_epoch {cfg.begin_epoch}",
            f"--dataset {self._resolve_dataset_name()}",
            f"--threshold {cfg.threshold}",
            f"--wce_weight {cfg.wce_weight}",
            f"--begin_coguess {cfg.begin_coguess}",
            "--do_train",
        ]
        if cfg.evaluate_during_training:
            parts.append("--evaluate_during_training")
        if cfg.overwrite_output_dir:
            parts.append("--overwrite_output_dir")
        # Merge any extra params
        for k, v in cfg.extra.items():
            parts.append(f"--{k} {v}")
        return " ".join(parts)


# ---- MProto ----------------------------------------------------------------

class MProtoRunner(BaseRunner):
    """Runner for MProto (prototypical networks for DS-NER).

    Uses the `alchemy` framework. All configuration is in .toml files.
    Entry point: python -m scripts.train_and_test <config.toml> [options]

    The train_and_test script automatically:
      1. Trains using alchemy.run()
      2. Tests using best checkpoint
    """

    def _resolve_config_path(self) -> str:
        """Find the .toml config, auto-detecting if not explicitly set."""
        cfg: MProtoConfig = self.config  # type: ignore
        if cfg.config_path:
            path = self.method_dir / cfg.config_path
            if path.exists():
                return cfg.config_path
            # Try as-is (might be absolute)
            if Path(cfg.config_path).exists():
                return cfg.config_path
            raise FileNotFoundError(f"Config not found: {cfg.config_path}")

        # Auto-detect: search cfg/ for toml files matching dataset
        cfg_dir = self.method_dir / "cfg"
        if not cfg_dir.exists():
            raise FileNotFoundError(
                f"No cfg/ directory found in {self.method_dir}. Set config.config_path manually."
            )
        ds = cfg.dataset.lower()
        candidates = []
        for toml_file in cfg_dir.rglob("*.toml"):
            if ds in str(toml_file).lower():
                candidates.append(toml_file)
        if not candidates:
            # Fallback: list all toml files
            all_tomls = list(cfg_dir.rglob("*.toml"))
            raise FileNotFoundError(
                f"No .toml config found matching dataset '{cfg.dataset}'.\n"
                f"Available configs: {[str(t.relative_to(self.method_dir)) for t in all_tomls]}\n"
                f"Set config.config_path explicitly."
            )
        # Prefer paths containing 'mproto' and 'train'
        best = candidates[0]
        for c in candidates:
            name = str(c).lower()
            if "mproto" in name and "train" in name:
                best = c
                break
        config_path = str(best.relative_to(self.method_dir))
        logger.info("Auto-detected MProto config: %s", config_path)
        return config_path

    def _build_cmd(self, config_path: str) -> str:
        cfg: MProtoConfig = self.config  # type: ignore
        parts = [
            f"python -m scripts.train_and_test {config_path}",
        ]
        if cfg.device:
            parts.append(f"--device {cfg.device}")
        elif cfg.gpu_ids:
            # Convert "0" or "0,1" to "[0]" or "[0,1]" format
            gpu_list = cfg.gpu_ids.replace(" ", "")
            parts.append(f"--device [{gpu_list}]")
        if cfg.user_dir != "src":
            parts.append(f"--user-dir {cfg.user_dir}")
        if cfg.desc:
            parts.append(f"--desc {cfg.desc}")
        if cfg.debug:
            parts.append("--debug")
        for k, v in cfg.extra.items():
            if isinstance(v, bool):
                if v:
                    parts.append(f"--{k}")
            else:
                parts.append(f"--{k} {v}")
        return " ".join(parts)

    def train(self):
        config_path = self._resolve_config_path()
        return self._run_command(self._build_cmd(config_path))

    def evaluate(self):
        # train_and_test automatically runs test after training
        return self.train()


# ---- RoSTER ----------------------------------------------------------------

class RoSTERRunner(BaseRunner):
    """Runner for RoSTER (EMNLP 2021).

    Entry point: src/train.py
    Shell scripts (run_qtl.sh, run_conll.sh, etc.) are self-contained
    (GPU set via CUDA_VISIBLE_DEVICES env var, corpus hardcoded).

    Available scripts:
      run_qtl.sh, run_conll.sh, run_ontonote.sh, run_twitter.sh,
      run_webpage.sh, run_wikigold.sh, run_bc5cdr.sh
    """

    DATASET_SCRIPT_MAP = {
        "qtl": "run_qtl.sh",
        "conll": "run_conll.sh",
        "ontonote": "run_ontonote.sh",
        "ontonotes": "run_ontonote.sh",
        "twitter": "run_twitter.sh",
        "tweet": "run_twitter.sh",
        "webpage": "run_webpage.sh",
        "wikigold": "run_wikigold.sh",
        "bc5cdr": "run_bc5cdr.sh",
    }

    def _get_shell_script(self) -> Optional[str]:
        ds = self.config.dataset.lower()
        for key, script in self.DATASET_SCRIPT_MAP.items():
            if key in ds:
                if (self.method_dir / script).exists():
                    return script
        return None

    def train(self):
        cfg: RoSTERConfig = self.config  # type: ignore
        script = self._get_shell_script()
        if script:
            # RoSTER scripts set GPU internally; override via env
            return self._run_command(f"bash {script}")
        else:
            return self._run_command(self._build_python_cmd(do_train=True, do_eval=cfg.do_eval))

    def evaluate(self):
        return self._run_command(self._build_python_cmd(do_train=False, do_eval=True))

    def _build_python_cmd(self, do_train=True, do_eval=True) -> str:
        cfg: RoSTERConfig = self.config  # type: ignore
        data_dir = str(self.method_dir / "data" / cfg.dataset)
        out_dir = cfg.output_dir if cfg.output_dir != "output" else f"out_{cfg.dataset}"
        temp_dir = cfg.temp_dir if cfg.temp_dir else f"tmp_{cfg.dataset}_{cfg.seed}"

        parts = [
            f"python -u src/train.py",
            f"--data_dir {data_dir}",
            f"--output_dir {out_dir}",
            f"--temp_dir {temp_dir}",
            f"--pretrained_model {cfg.pretrained_model}",
            f"--tag_scheme {cfg.tag_scheme}",
            f"--max_seq_length {cfg.max_seq_length}",
            f"--train_batch_size {cfg.train_batch_size}",
            f"--gradient_accumulation_steps {cfg.gradient_accumulation_steps}",
            f"--eval_batch_size {cfg.eval_batch_size}",
            f"--noise_train_lr {cfg.noise_train_lr}",
            f"--ensemble_train_lr {cfg.ensemble_train_lr}",
            f"--self_train_lr {cfg.self_train_lr}",
            f"--noise_train_epochs {cfg.noise_train_epochs}",
            f"--ensemble_train_epochs {cfg.ensemble_train_epochs}",
            f"--self_train_epochs {cfg.self_train_epochs}",
            f"--noise_train_update_interval {cfg.noise_train_update_interval}",
            f"--self_train_update_interval {cfg.self_train_update_interval}",
            f"--dropout {cfg.dropout}",
            f"--warmup_proportion {cfg.warmup_proportion}",
            f"--seed {cfg.seed}",
            f"--q {cfg.q}",
            f"--tau {cfg.tau}",
            f"--num_models {cfg.num_models}",
        ]
        if do_train:
            parts.append("--do_train")
        if do_eval:
            parts.append("--do_eval")
            parts.append(f'--eval_on "{cfg.eval_on}"')
        for k, v in cfg.extra.items():
            if isinstance(v, bool):
                if v:
                    parts.append(f"--{k}")
            else:
                parts.append(f"--{k} {v}")
        return " ".join(parts)


# ---- SCDL ------------------------------------------------------------------

class SCDLRunner(BaseRunner):
    """Runner for SCDL (EMNLP 2021).

    Entry point: run_script.py (called via run_script.sh)
    Shell script takes: $1=GPU_ID  $2=DATASET
    """

    def _find_shell_script(self) -> Optional[str]:
        """SCDL uses a single run_script.sh for all datasets."""
        if (self.method_dir / "run_script.sh").exists():
            return "run_script.sh"
        return None

    def train(self):
        cfg: SCDLConfig = self.config  # type: ignore
        script = self._find_shell_script()
        if script:
            return self._run_command(f"bash {script} {cfg.gpu_ids} {self._resolve_dataset_name()}")
        else:
            return self._run_command(self._build_python_cmd())

    def evaluate(self):
        # SCDL evaluates during training (--evaluate_during_training)
        return self.train()

    def _build_python_cmd(self) -> str:
        """Build full python command matching run_script.sh structure."""
        cfg: SCDLConfig = self.config  # type: ignore
        project_root = str(self.method_dir)
        data_root = str(self.method_dir / "dataset")
        output_dir = str(self.method_dir / "ptms" / cfg.dataset)

        parts = [
            f"python3 -u run_script.py",
            f"--data_dir {data_root}",
            f"--student1_model_name_or_path {cfg.student1_model_name}",
            f"--student2_model_name_or_path {cfg.student2_model_name}",
            f"--output_dir {output_dir}",
            f"--tokenizer_name {cfg.tokenizer_name}",
            f"--cache_dir {project_root}/cached_models",
            f"--max_seq_length {cfg.max_seq_length}",
            f"--learning_rate {cfg.learning_rate}",
            f"--weight_decay {cfg.weight_decay}",
            f"--adam_epsilon {cfg.adam_epsilon}",
            f"--adam_beta1 {cfg.adam_beta1}",
            f"--adam_beta2 {cfg.adam_beta2}",
            f"--max_grad_norm {cfg.max_grad_norm}",
            f"--num_train_epochs {cfg.num_train_epochs}",
            f"--warmup_steps {cfg.warmup_steps}",
            f"--per_gpu_train_batch_size {cfg.train_batch_size}",
            f"--per_gpu_eval_batch_size {cfg.eval_batch_size}",
            f"--gradient_accumulation_steps {cfg.gradient_accumulation_steps}",
            f"--logging_steps {cfg.logging_steps}",
            f"--save_steps {cfg.save_steps}",
            f"--seed {cfg.seed}",
            f"--mean_alpha {cfg.mean_alpha}",
            f"--self_learning_label_mode {cfg.label_mode}",
            f"--self_learning_period {cfg.period}",
            f"--model_type {cfg.model_type}",
            f"--begin_epoch {cfg.begin_epoch}",
            f"--dataset {self._resolve_dataset_name()}",
            f"--threshold {cfg.threshold}",
            "--do_train",
        ]
        if cfg.evaluate_during_training:
            parts.append("--evaluate_during_training")
        if cfg.overwrite_output_dir:
            parts.append("--overwrite_output_dir")
        for k, v in cfg.extra.items():
            if isinstance(v, bool):
                if v:
                    parts.append(f"--{k}")
            else:
                parts.append(f"--{k} {v}")
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Runner registry
# ---------------------------------------------------------------------------

RUNNER_REGISTRY: Dict[str, type] = {
    "ATSEN": ATSENRunner,
    "AutoNER": AutoNERRunner,
    "BOND": BONDRunner,
    "CuPuL": CuPuLRunner,
    "DeSERT": DeSERTRunner,
    "MProto": MProtoRunner,
    "RoSTER": RoSTERRunner,
    "SCDL": SCDLRunner,
}

# Method directory names as they appear on disk
METHOD_DIR_NAMES: Dict[str, str] = {
    "ATSEN": "ATSEN",
    "AutoNER": "AutoNER",
    "BOND": "BOND",
    "CuPuL": "CuPuL",
    "DeSERT": "DeSERT",
    "MProto": "mproto",
    "RoSTER": "RoSTER",
    "SCDL": "SCDL",
}


# ---------------------------------------------------------------------------
# Main wrapper class
# ---------------------------------------------------------------------------

class DSNERWrapper:
    """
    Unified interface for all 8 DS-NER methods.

    Example
    -------
    >>> wrapper = DSNERWrapper(
    ...     method="BOND",
    ...     project_root="/home/user/original_dsner",
    ...     dataset="conll03",
    ...     gpu_ids="0",
    ... )
    >>> wrapper.train()
    >>> wrapper.evaluate()
    """

    SUPPORTED_METHODS = list(RUNNER_REGISTRY.keys())

    def __init__(
        self,
        method: str,
        project_root: str,
        dataset: str = "conll03",
        gpu_ids: str = "0",
        config: Optional[BaseConfig] = None,
        **kwargs,
    ):
        method = self._normalize_method_name(method)
        self.method = method
        self.project_root = Path(project_root).resolve()

        # Build config
        if config is not None:
            self.config = config
        else:
            cfg_cls = CONFIG_REGISTRY[method]
            # Filter kwargs to only those accepted by the config class
            import dataclasses
            valid_fields = {f.name for f in dataclasses.fields(cfg_cls)}
            init_kwargs = {k: v for k, v in kwargs.items() if k in valid_fields}
            extra_kwargs = {k: v for k, v in kwargs.items() if k not in valid_fields}
            self.config = cfg_cls(dataset=dataset, gpu_ids=gpu_ids, **init_kwargs)
            self.config.extra.update(extra_kwargs)

        # Instantiate runner
        method_dir = METHOD_DIR_NAMES[method]
        runner_cls = RUNNER_REGISTRY[method]
        self.runner: BaseRunner = runner_cls(
            project_root=str(self.project_root),
            method_dir=method_dir,
            config=self.config,
        )

        logger.info(
            "Initialized DSNERWrapper: method=%s, dataset=%s, gpu=%s",
            method, dataset, gpu_ids,
        )

    @staticmethod
    def _normalize_method_name(name: str) -> str:
        """Case-insensitive lookup of method name."""
        lookup = {k.lower(): k for k in RUNNER_REGISTRY}
        key = name.lower().replace("-", "").replace("_", "")
        if key not in lookup:
            raise ValueError(
                f"Unknown method '{name}'. Supported: {list(RUNNER_REGISTRY.keys())}"
            )
        return lookup[key]

    def train(self) -> subprocess.CompletedProcess:
        """Train the selected DS-NER method."""
        logger.info("Starting training: %s on %s", self.method, self.config.dataset)
        return self.runner.train()

    def evaluate(self) -> subprocess.CompletedProcess:
        """Evaluate the selected DS-NER method."""
        logger.info("Starting evaluation: %s on %s", self.method, self.config.dataset)
        return self.runner.evaluate()

    def predict(self, input_path: Optional[str] = None) -> subprocess.CompletedProcess:
        """Run prediction / inference."""
        logger.info("Starting prediction: %s", self.method)
        return self.runner.predict(input_path)

    def get_config(self) -> Dict[str, Any]:
        """Return current configuration as a dict."""
        return self.config.to_dict()

    def set_config(self, **kwargs):
        """Update configuration fields."""
        for k, v in kwargs.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)
            else:
                self.config.extra[k] = v

    @classmethod
    def list_methods(cls) -> List[str]:
        """Return all supported method names."""
        return cls.SUPPORTED_METHODS

    def __repr__(self):
        return (
            f"DSNERWrapper(method='{self.method}', "
            f"dataset='{self.config.dataset}', "
            f"gpu_ids='{self.config.gpu_ids}')"
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Simple CLI: python dsner_wrapper.py --method BOND --dataset conll03 --action train"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Unified DS-NER Wrapper",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--method", required=True,
        help=f"Method name. Supported: {DSNERWrapper.SUPPORTED_METHODS}",
    )
    parser.add_argument("--project_root", required=True, help="Path to original_dsner directory")
    parser.add_argument("--dataset", default="conll03", help="Dataset name")
    parser.add_argument("--gpu_ids", default="0", help="Comma-separated GPU IDs")
    parser.add_argument(
        "--action", default="train", choices=["train", "evaluate", "predict"],
        help="Action to perform",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--num_train_epochs", type=int, default=50)
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--output_dir", default="output")
    # Extra pass-through args
    parser.add_argument(
        "--extra", nargs="*", default=[],
        help="Extra key=value pairs passed to method config",
    )

    args = parser.parse_args()

    extra = {}
    for item in args.extra:
        if "=" in item:
            k, v = item.split("=", 1)
            # Try to cast numeric values
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    pass
            extra[k] = v

    wrapper = DSNERWrapper(
        method=args.method,
        project_root=args.project_root,
        dataset=args.dataset,
        gpu_ids=args.gpu_ids,
        seed=args.seed,
        learning_rate=args.learning_rate,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        max_seq_length=args.max_seq_length,
        output_dir=args.output_dir,
        **extra,
    )

    action_fn = getattr(wrapper, args.action)
    result = action_fn()
    sys.exit(result.returncode if result else 0)


if __name__ == "__main__":
    main()
