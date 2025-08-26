#!/bin/bash

# List of alpha values
devices=(6 7)
starts=(0 500)
ends=(499 999)

# Path to the Python script
script_path="llama_on_wiki_movie.py"

# Loop through the alpha values and launch the Python script on different CUDA devices
for i in "${!devices[@]}"; do
    cuda_device="${devices[$i]}"
    start="${starts[$i]}"
    end="${ends[$i]}"

    echo "Launching script with start=$start,end=$end on CUDA device $cuda_device"

    # Set CUDA_VISIBLE_DEVICES before running the script
    CUDA_VISIBLE_DEVICES="$cuda_device" PYTHONPATH=../../../ python "$script_path" --model_size 7 --batch_size 2 --start $start --end $end

    echo "--------------------------------------"
done
