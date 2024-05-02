#!/bin/bash

# Define the path to the Singularity image and the overlay
SIF_IMAGE="/scratch/tl2546/my_env/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif"
OVERLAY="/scratch/tl2546/my_env/overlay-25GB-500K.ext3"

# Construct the command to be executed inside the Singularity container
INNER_COMMAND="source /ext3/env.sh && conda activate NLU_project && /bin/bash"

# Define the full Singularity command with proper quoting
CMD="singularity exec --nv --overlay ${OVERLAY}:rw ${SIF_IMAGE} /bin/bash -c '$INNER_COMMAND'"

# Execute the command
echo "Running the Singularity container..."
eval $CMD

# Note: Using 'eval' to correctly handle complex nested quotes in command execution.
echo "Singularity container has exited."
