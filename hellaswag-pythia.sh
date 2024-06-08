#!/bin/bash
#$ -N hellaswag-pythia-1.4b
#$ -o /exports/eddie/scratch/s2558433/pythia-1.4b_$JOB_ID.log
#$ -e /exports/eddie/scratch/s2558433/pythia-1.4b_$JOB_ID.err
#$ -cwd
#$ -pe sharedmem 16
#$ -l h_vmem=4G
#$ -l h_rt=48:00:00

source /exports/eddie/scratch/s2558433/miniconda3/etc/profile.d/conda.sh

cd /exports/eddie/scratch/s2558433/
conda create -n eval python=3.8
conda activate eval

cd base_extraction_implementaion

# Run the main script
#python main.py --N 1000 --batch-size 10 --model1 EleutherAI/pythia-2.8b --model2 EleutherAI/pythia-160m --corpus-path monology/pile-uncopyrighted
#python main.py --N 1000 --batch-size 10 --model1 EleutherAI/pythia-2.8b --model2 EleutherAI/pythia-410m --corpus-path monology/pile-uncopyrighted
#python main.py --N 1000 --batch-size 10 --model1 EleutherAI/pythia-2.8b --model2 EleutherAI/pythia-1b --corpus-path monology/pile-uncopyrighted
python main.py --N 1000 --batch-size 10 --model1 EleutherAI/pythia-2.8b --model2 EleutherAI/pythia-1.4b --corpus-path monology/pile-uncopyrighted
#python main.py --N 1000 --batch-size 10 --model1 EleutherAI/pythia-2.8b --model2 EleutherAI/pythia-2.8b --corpus-path monology/pile-uncopyrighted
