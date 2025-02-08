import numpy as np
import pandas as pd
from evaluator import Evaluator
from src.data_manager.data_manager import DataManager
import os
from datetime import datetime

# 추천 결과 파일 경로 설정
recos_paths = {
    'MF-Transformer': 'resources/recos/MF-Transformer.npy',
    'NN-Transformer': 'resources/recos/MF-Transformer.npy',
    'FM-Transformer': 'resources/recos/MF-Transformer.npy',
    'MF-GRU': 'resources/recos/MF-GRU.npy',
    'MF-AVG': 'resources/recos/MF-AVG.npy',
    
    'SKNN': 'resources/recos/VSKNN.npy',
    'STAN': 'resources/recos/VSKNN.npy',
    #  'VSTAN': 'resources/recos/VSKNN.npy',
    #  'VSKNN': 'resources/recos/VSKNN.npy',

    'MUSE' : 'resources/recos/MUSE.npy',
    'LARP' : 'resources/recos/LARP.npy',
    'PISA' : 'resources/recos/PISA.npy'
}

# 추천 결과 로드
recos_dict = {model_name: np.load(path, allow_pickle=True) for model_name, path in recos_paths.items()}

# 데이터 매니저 초기화 (테스트 데이터 로드를 위해 필요)
data_manager = DataManager()

# 평가자 초기화 (ground truth와 함께)
gt_test = [] 
for i in DataManager.N_SEED_SONGS:
    gt_test += data_manager.ground_truths["test"][i]
test_evaluator = Evaluator(data_manager, gt=np.array(gt_test), n_recos=500)

# 메트릭 계산 함수 정의
def compute_metrics(recos, evaluator):
    precisions = evaluator.compute_all_precisions(recos)
    recalls = evaluator.compute_all_recalls(recos)
    r_precisions = evaluator.compute_all_R_precisions(recos)
    ndcgs = evaluator.compute_all_ndcgs(recos)
    clicks = evaluator.compute_all_clicks(recos)
    return {
        'Precision': precisions.mean(),
        'Recall': recalls.mean(),
        'R-Precision': r_precisions.mean(),
        'NDCG': ndcgs.mean(),
        'Clicks': clicks.mean()
    }

# 각 모델에 대해 메트릭 계산 및 결과 저장
metrics_results = {model_name: compute_metrics(recos, test_evaluator) for model_name, recos in recos_dict.items()}

# 결과 출력
for model_name, metrics in metrics_results.items():
    print(f"Metrics for {model_name}:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value}")

# 결과를 데이터프레임에 저장
metrics_df = pd.DataFrame(metrics_results).T

# 데이터프레임 확인
print(metrics_df)

# CSV 파일로 저장 (폴더 경로 포함)
output_folder = 'metrics_results'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 현재 시간을 MMDDHHMM 형식으로 포맷팅
current_time = datetime.now().strftime('%m%d%H%M')
filename = f'couterpart_results_{current_time}.csv'

full_path = os.path.join(output_folder, filename)
metrics_df.to_csv(full_path)

print(f"Metrics saved to: {full_path}")