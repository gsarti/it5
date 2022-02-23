for script in generated_scripts/*.sh;
do
    echo "Running $script"
    sbatch $script
done