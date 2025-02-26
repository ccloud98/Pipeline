#!/bin/bash

# baseline_train.sh 파일 생성

# 에러 발생 시 중단
set -e

# 학습 시작 시간 기록
start_time=$(date +"%Y-%m-%d %H:%M:%S")

# 학습할 baseline 모델 리스트
models=("MUSE" "LARP" "PISA")

for model in "${models[@]}"
do
    echo "================================================="
    echo "Starting training for $model at $(date +"%Y-%m-%d %H:%M:%S")"
    CUDA_VISIBLE_DEVICES=0 python src/run_baselines.py --model_name "$model"
    echo "Completed training for $model at $(date +"%Y-%m-%d %H:%M:%S")"
    echo "================================================="
done

echo "All baseline training completed!"
echo "Training started at: $start_time"
echo "Training finished at: $(date +"%Y-%m-%d %H:%M:%S")"