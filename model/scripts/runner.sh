srun --account=cis431_531 \
    --job-name=project_gpu \
    --partition=preempt \
    --time=02:10:00 \
    --nodes=1 \
    --ntasks-per-node=1 \
    --gres=gpu:1 \
    --cpus-per-task=28 \
    --pty /bin/bash
