import os
import shutil
from vars import header, command, footer, models, configs, get_params

out_path = "generated_scripts"
shutil.rmtree(out_path, ignore_errors=True)
os.makedirs(out_path, exist_ok=True)

for model in models:
    model_name = model.split("/")[-1]
    for config in configs:
        params = get_params(model, config)
        out_file = f"{out_path}/{model_name}-{config}.sh"
        with open(out_file, "w") as f:
            f.write(header.format(model_name=model_name,dataset_name=config,time=params["time"]))
            f.write(command.format(
                model=model,
                model_name=model_name,
                config=config,
                bs=params["bs"],
                source_len=params["source_len"],
                target_len=params["target_len"],
                epochs=params["epochs"],
                lr=params["lr"],
            ))
            f.write(footer)