header = """#!/bin/bash
#SBATCH --job-name=run-seq2seq-{model_name}-{dataset_name}
#SBATCH --time={time}:00:00
#SBATCH --partition=gpu
#SBATCH --mem=16GB
#SBATCH --gres=gpu:v100:1
#SBATCH --output=/data/p305238/slurm_logs/%x.%j.out
 
module purge
module load Python/3.8.6-GCCcore-10.2.0
module load CUDA/10.2.89-GCC-8.3.0

cd /data/$ME
source venv/bin/activate
 
python3 --version
which python3

export HF_HOME=/data/$ME/hf_cache/

"""

command = """python3 -u /home/$ME/scripts/run_seq2seq.py \\
    --model_name_or_path {model} \\
    --do_train \\
    --do_eval \\
    --use_auth_token <YOUR_AUTH_TOKEN> \\
    --max_source_length {source_len} \\
    --max_target_length {target_len} \\
    --evaluation_strategy "steps" \\
    --eval_steps 5000 \\
    --save_steps 2000 \\
    --learning_rate {lr} \\
    --num_train_epochs {epochs} \\
    --save_total_limit 1 \\
    --dataset_name it5/datasets \\
    --dataset_config {config} \\
    --output_dir /scratch/$ME/it5_experiments/runs/{model_name}-{config}-{lr} \\
    --per_device_train_batch_size={bs} \\
    --per_device_eval_batch_size={bs} \\
    --overwrite_output_dir \\
    --predict_with_generate

"""

footer = "deactivate"

models = [
    "google/mt5-small",
    "google/mt5-base",
    "gsarti/it5-small",
    "gsarti/it5-base",
    "gsarti/it5-large",
    "it5/it5-efficient-small-el32"
]

configs = [
    "fst",
    "hg",
    "ns",
    "qa",
    "qg",
    "st_g2r",
    "st_r2g",
    "wits",
]


def get_bsz(model):
    if model.startswith("gsarti/it5-small"):
        return 16
    elif model.startswith("gsarti/it5-base") or model.startswith("it5/it5-efficient-small"):
        return 8
    elif "mt5-base" in model or model.startswith("gsarti/it5-large"):
        return 4
    raise Exception(f"Unknown model: {model}")


def get_time(model):
    if model.startswith("gsarti/it5-small"):
        return 10
    elif model.startswith("gsarti/it5-base") or model.startswith("it5/it5-efficient-small"):
        return 15
    elif "mt5-base" in model or model.startswith("gsarti/it5-large"):
        return 24
    raise Exception(f"Unknown model: {model}")


def get_lr(model):
    if model.startswith("gsarti/it5-small"):
        return 8e-4
    elif model.startswith("gsarti/it5-base") or model.startswith("it5/it5-efficient-small"):
        return 3e-4
    elif "mt5-base" in model or model.startswith("gsarti/it5-large"):
        return 5e-5
    raise Exception(f"Unknown model: {model}")


def get_params(model, config):
    params = {
        "fst": {
            "source_len": 128,
            "target_len": 128,
            "epochs": 10,
        },
        "hg": {
            "source_len": 512,
            "target_len": 64,
            "epochs": 7,
        },
        "ns": {
            "source_len": 512,
            "target_len": 128,
            "epochs": 7,
        },
        "qa": {
            "source_len": 512,
            "target_len": 64,
            "epochs": 7,
        },
        "qg": {
            "source_len": 512,
            "target_len": 128,
            "epochs": 7,
        },
        "st_g2r": {
            "source_len": 512,
            "target_len": 64,
            "epochs": 10,
        },
        "st_r2g": {
            "source_len": 512,
            "target_len": 64,
            "epochs": 10,
        },
        "wits": {
            "source_len": 512,
            "target_len": 256,
            "epochs": 3,
        },
    }
    params[config]["bs"] = get_bsz(model)
    params[config]["time"] = get_time(model)
    params[config]["lr"] = get_lr(model)
    return params[config]
