#!/bin/bash                                                                                                                                                                                               
#SBATCH --job-name=clip_extract                                                                                                                                                                           
#SBATCH --gres=gpu:v100:1                                                                                                                                                                                 
#SBATCH --cpus-per-task=8                                                                                                                                                                                 
#SBATCH --mem=32G                                                                                                                                                                                         
#SBATCH --time=03:00:00                                                                                                                                                                               
#SBATCH --array=0-441%16                                                                                                                                                                                     
#SBATCH --output=/ibex/scratch/alhumosm/SoccerNet/logs/clip_extract_%A_%a.out                                                                                                                             
#SBATCH --error=/ibex/scratch/alhumosm/SoccerNet/logs/clip_extract_%A_%a.err                                                                                                                              
                                                                                                                                                                                                        
mkdir -p /ibex/scratch/alhumosm/SoccerNet/logs                                                                                                                                                            
                                                                                                                                                                                                        
source ~/.bashrc                                                                                                                                                                                          
conda activate soccernet                                                                                                                                                                                   
                                                                                                                                                                                                        
cd /home/alhumosm/soccernet/clip                                                                                                                                                                      
                                                                                                                                                                                                        
python extract_clip_features.py \
    --dataset_root /ibex/scratch/alhumosm/SoccerNet/soccernet_videos \
    --fps 2.0 \
    --batch_size 256 \
    --clip_model ViT-L/14 \
    --device cuda \
    --job_id "$SLURM_ARRAY_TASK_ID" \
    --num_jobs "$SLURM_ARRAY_TASK_COUNT"
