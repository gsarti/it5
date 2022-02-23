import os
import shutil
from vars import header, command, footer, models, settings, get_params

out_path = "generated_scripts"
shutil.rmtree(out_path, ignore_errors=True)
os.makedirs(out_path, exist_ok=True)

for model in models:
    model_name = model.split("/")[-1]
    for setting in settings:
        params = get_params(model, setting)
        for split in params.get("splits", ["test"]):
            out_file = f"{out_path}/{model_name}-{setting}-{split}.sh"
            with open(out_file, "w") as f:
                f.write(header.format(model_name=model_name,dataset_name=setting,split=split))
                full_model_name = model + "-" + params["suffix"]
                f.write(command.format(
                    model=full_model_name,
                    config=params.get("config", setting),
                    bs=params["bs"],
                    source_len=params["source_len"],
                    target_len=params["target_len"],
                    source_column=params["source_column"],
                    target_column=params["target_column"],
                    split=split
                ))
                f.write(footer)