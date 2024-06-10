#!/bin/bash

#$ -N hellaswag-pythia-160m
#$ -o /exports/eddie/scratch/s2558433/eval-p160m_$JOB_ID.log
#$ -e /exports/eddie/scratch/s2558433/eval-p160m_$JOB_ID.err
#$ -cwd
#$ -pe sharedmem 16
#$ -l h_vmem=4G
#$ -l h_rt=48:00:00

source /exports/eddie/scratch/s2558433/miniconda3/etc/profile.d/conda.sh

cd /exports/eddie/scratch/s2558433/
#conda create -n eval python=3.9
conda activate eval

cd lm-evaluation-harness
pip install -e .


#test on the pythia 160

lm_eval --model hf \
    --model_args pretrained=EleutherAI/pythia-160m \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size 8 \
    --output_path results_pythia160m

conda deactivate
