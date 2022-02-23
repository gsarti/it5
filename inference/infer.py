import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm
import torch
from datasets import load_dataset

from transformers import (
    HfArgumentParser,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

from transformers.utils import check_min_version
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.15.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    from_flax: bool = field(
        default=False,
        metadata={
            "help": "If true, the model will be loaded from a saved Flax checkpoint."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    dataset_split: Optional[str] = field(
        default="test", metadata={"help": "The split of the dataset to use (via the datasets library)."}
    )
    source_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    target_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    batch_size: Optional[int] = field(
        default=8,
        metadata={"help": "Batch size used for inference."},
    )
    output_dir: Optional[str] = field(
        default=".",
        metadata={"help": "Output dir."},
    )

name_mapping = {
    "fst": ("formal", "informal"),
    "hg": ("text", "target"),
    "ns": ("source", "target"),
    "qa": ("source", "target"),
    "qg": ("text", "target"),
    "st_g2r": ("full_text", "headline"),
    "st_r2g": ("full_text", "headline"),
    "wits": ("source", "summary"),
}


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    model_shortname = model_args.model_name_or_path if "/" not in model_args.model_name_or_path else model_args.model_name_or_path.split("/")[-1]

    print(f"Loading model {model_args.model_name_or_path} and tokenizer from {model_args.tokenizer_name}...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_flax=model_args.from_flax,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=model_args.use_auth_token,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=model_args.use_auth_token,
    )
    model.resize_token_embeddings(len(tokenizer))
    print(f"Loading dataset {data_args.dataset_name} with config {data_args.dataset_config_name}")
    dataset = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        cache_dir=model_args.cache_dir,
        use_auth_token=model_args.use_auth_token
    )
    column_names = dataset[data_args.dataset_split].column_names

    # Get the column names for input/target.
    dataset_columns = name_mapping.get(data_args.dataset_config_name, None)
    if data_args.source_column is None:
        source_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        source_column = data_args.source_column
        if source_column not in column_names:
            raise ValueError(
                f"--source_column' value '{data_args.source_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.target_column is None:
        target_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        target_column = data_args.target_column
        if target_column not in column_names:
            raise ValueError(
                f"--target_column' value '{data_args.target_column}' needs to be one of: {', '.join(column_names)}"
            )
    
    def preprocess_function(examples):
        # remove pairs where at least one record is None
        inputs, targets = [], []
        for i in range(len(examples[source_column])):
            if examples[source_column][i] is not None and examples[target_column][i] is not None:
                inputs.append(examples[source_column][i])
                targets.append(examples[target_column][i])
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding="max_length", truncation=True)
        return model_inputs
    
    predict_dataset = dataset[data_args.dataset_split].map(
        preprocess_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on prediction dataset",
    )
    print(f"Example: {predict_dataset[0]}")
    predict_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    dataloader = torch.utils.data.DataLoader(predict_dataset, batch_size=data_args.batch_size)
    gen_kwargs = {
        "max_length": data_args.max_target_length,
        "num_beams": data_args.num_beams,
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)
    print(f"Inferencing...")
    predictions = []
    for i, batch in enumerate(tqdm(dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model.generate(**batch, **gen_kwargs)
        outputs = tokenizer.batch_decode(out.to("cpu"), skip_special_tokens=True)
        if i == 0:
            print(outputs[:2])
        predictions.extend(outputs)
    assert len(predictions) == len(predict_dataset)
    fname = f"{model_shortname}_{data_args.dataset_split}.txt"
    out_path = os.path.join(data_args.output_dir, fname)
    print(f"Writing predictions to {out_path}")
    with open(out_path, 'w') as f:
        for pred in predictions:
            f.write(f"{pred}\n")

if __name__ == "__main__":
    main()