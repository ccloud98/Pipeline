import numpy as np
import pandas as pd
from evaluator import Evaluator
from src.data_manager.data_manager import DataManager
import os
from datetime import datetime

#############################
# 1) 추천 결과 파일 경로 설정
#############################
recos_paths = {
    'MF-Transformer': 'resources/recos/MF-Transformer.npy',
    'MUSE' : 'resources/recos/MUSE.npy',
    'LARP' : 'resources/recos/LARP.npy',
    'PISA' : 'resources/recos/PISA.npy'
}

#############################
# 2) 추천 결과 로드
#############################
# np.load(... allow_pickle=True)로 배열 읽기
recos_dict_raw = {
    model_name: np.load(path, allow_pickle=True)
    for model_name, path in recos_paths.items()
}

#############################
# 3) DataManager 초기화
#    (테스트 평가용 데이터+ground truth 로드)
#############################
data_manager = DataManager()

# Evaluator 준비
gt_test = []
for i in DataManager.N_SEED_SONGS:  # range(1..10)
    gt_test += data_manager.ground_truths["test"][i]
test_evaluator = Evaluator(data_manager, gt=np.array(gt_test), n_recos=500)

#############################
# 4) 안전한 클램핑 (item id 범위 밖 접근 방지)
#############################
def safe_clamp_recos(recos, n_tracks):
    # recos.shape: [num_test_playlists, n_recos]
    # clamp between 0 and (n_tracks-1)
    return np.clip(recos, 0, n_tracks-1)

# 실제 사용 recos_dict
recos_dict = {}
for model_name, arr in recos_dict_raw.items():
    recos_dict[model_name] = safe_clamp_recos(arr, data_manager.n_tracks-1)

#############################
# 5) 일반 평가 메트릭 (Precision, Recall 등) 계산 함수
#############################
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

#############################
# 6) cold-start 아이템 분석을 위한 함수
#############################
def analyze_coldstart_false_cases(recos, evaluator, coldstart_items):
    """
    recos: (num_test_playlists, n_recos) 추천 결과
    evaluator: Evaluator 객체 (evaluator.gt[i] = i번째 세션의 ground truth 세트)
    coldstart_items: set of item_ids considered cold-start
    """
    gt_all = evaluator.gt  # shape [N], 각 원소는 set(...)
    N = len(gt_all)

    total_FN = 0
    total_cold_FN = 0
    total_FP = 0
    total_cold_FP = 0

    for i in range(N):
        gt_set = gt_all[i]
        rec_set = set(recos[i])

        # False negative: 정답이지만 추천되지 못한 아이템
        FN = gt_set - rec_set
        # False positive: 추천했지만 정답이 아닌 아이템
        FP = rec_set - gt_set

        # cold-start 아이템만 골라보기
        cold_FN = FN.intersection(coldstart_items)
        cold_FP = FP.intersection(coldstart_items)

        total_FN += len(FN)
        total_cold_FN += len(cold_FN)
        total_FP += len(FP)
        total_cold_FP += len(cold_FP)

    # 분모가 0일 수도 있으므로 체크
    fn_ratio = total_cold_FN / total_FN if total_FN > 0 else 0.0
    fp_ratio = total_cold_FP / total_FP if total_FP > 0 else 0.0

    return {
        'FN_total': total_FN,
        'FN_cold': total_cold_FN,
        'FN_cold_ratio': fn_ratio,
        'FP_total': total_FP,
        'FP_cold': total_cold_FP,
        'FP_cold_ratio': fp_ratio
    }

#############################
# 7) cold-start 아이템 판별 (train 등장 아이템 집합 vs. 전체 아이템)
#############################
train_matrix = data_manager.binary_train_set  # shape: [train_sessions, n_tracks]
train_item_indices = set(train_matrix.indices)  # train 세션에 등장했던 아이템
all_items = set(range(data_manager.n_tracks))
coldstart_items = all_items - train_item_indices
print(f"Found {len(coldstart_items)} cold-start items out of {data_manager.n_tracks} total items.")

#############################
# 8) 각 모델에 대해: 
#    - 기본 메트릭 계산
#    - cold-start FN/FP 분석
#    - 결과 저장
#############################
analysis_results = {}
for model_name, recos in recos_dict.items():
    print("="*50)
    print(f"Analyzing Model: {model_name}")

    # (A) 기본 메트릭
    metrics = compute_metrics(recos, test_evaluator)

    # (B) cold-start 분석
    cs_analysis = analyze_coldstart_false_cases(recos, test_evaluator, coldstart_items)

    # 출력
    print("[Metrics]", metrics)
    print("[ColdStartAnalysis]", cs_analysis)

    # 병합하여 저장
    combined = {**metrics, **cs_analysis}
    analysis_results[model_name] = combined

#############################
# 9) DataFrame 만들고 CSV 저장
#############################
analysis_df = pd.DataFrame(analysis_results).T
print("\n=== Final Analysis DataFrame ===")
print(analysis_df)

output_folder = 'metrics_results'
os.makedirs(output_folder, exist_ok=True)

current_time = datetime.now().strftime('%m%d%H%M')
filename = f'coldstart_analysis_{current_time}.csv'
full_path = os.path.join(output_folder, filename)
analysis_df.to_csv(full_path)

print(f"\nAnalysis saved to: {full_path}")
