#!/bin/bash

# List of alpha values
alphas=(-2 -1 0 2 4 6 8)
devices=(0 1 2 3 4 5 6)

# Path to the Python script
script_path="subject_strengthen.py"

# Loop through the alpha values and launch the Python script on different CUDA devices
for i in "${!alphas[@]}"; do
    alpha="${alphas[$i]}"
    cuda_device="${devices[$i]}"

    echo "Launching script with alpha = $alpha on CUDA device $cuda_device"

    # Set CUDA_VISIBLE_DEVICES before running the script
    CUDA_VISIBLE_DEVICES="$cuda_device" PYTHONPATH=../../ python "$script_path" --model_size 7 --alpha "$alpha" &

    echo "--------------------------------------"
done
