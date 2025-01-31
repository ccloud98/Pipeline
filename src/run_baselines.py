import numpy as np
from src.data_manager.data_manager import DataManager, SequentialTrainDataset, EvaluationDataset, pad_collate
from src.utils import array_mapping
from src.evaluator import Evaluator
import json, argparse
import pandas as pd

from src.baselines.vsknn import VMContextKNN
from src.baselines.sknn import ContextKNN
from src.baselines.vstan import VSKNN_STAN
from src.baselines.stan import STAN
from src.baselines.muse import MUSE
from src.baselines.larp import LARP
from src.baselines.pisa import PISA

import tqdm
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True,
                        help="Name of model to train")
    parser.add_argument("--data_path", type=str, required=False,
                        help="path to data", default="resources/data/baselines")
    parser.add_argument("--params_file", type=str, required=False,
                        help="file for parameters", default="resources/params/best_params_baselines.json")
    parser.add_argument("--models_path", type=str, required=False,
                        help="Path to save models", default="resources/recos")
    # (A) 샘플링 크기를 직접 지정할 수 있도록 인자 추가 (선택)
    parser.add_argument("--sample_size", type=int, default=1000,
                        help="Number of train playlists to sample for training (default=10000)")

    args = parser.parse_args()

    # 1) 하이퍼파라미터 로드
    with open(args.params_file, "r") as f:
        p = json.load(f)
    tr_params = p[args.model_name]

    # 2) DataManager 및 데이터 로드
    data_manager = DataManager()
    df_train = pd.read_hdf(f"{args.data_path}/df_train_for_test")
    savePath = f"{args.models_path}/{args.model_name}"

    # 3) 모델 생성
    if args.model_name == "VSKNN":
        knnModel = VMContextKNN(
            k=tr_params["k"],
            sample_size=tr_params["n_sample"],
            weighting=tr_params["w"],
            weighting_score=tr_params["w_score"],
            idf_weighting=tr_params["idf_w"]
        )
    elif args.model_name == "SKNN":
        knnModel = ContextKNN(
            k=tr_params["k"],
            sample_size=tr_params["n_sample"],
            similarity=tr_params["s"]
        )
    elif args.model_name == "VSTAN":
        df_train["Time"] = df_train["Time"] / 1000
        knnModel = VSKNN_STAN(
            k=tr_params["k"],
            sample_size=tr_params["n_sample"],
            lambda_spw=tr_params["sp_w"],
            lambda_snh=tr_params["sn_w"],
            lambda_inh=tr_params["in_w"]
        )
    elif args.model_name == "STAN":
        knnModel = STAN(
            k=tr_params["k"],
            sample_size=tr_params["n_sample"],
            lambda_spw=tr_params["sp_w"],
            lambda_snh=tr_params["sn_w"],
            lambda_inh=tr_params["in_w"]
        )
    elif args.model_name == "LARP":
        knnModel = LARP(
            k=tr_params["k"],
            sample_size=tr_params["n_sample"],
            embed_dim=tr_params.get("embed_dim", 256)
        )
    elif args.model_name == "MUSE":
        knnModel = MUSE(
            data_manager=data_manager,
            k=tr_params["k"],
            n_items=tr_params["n_items"],
            hidden_size=tr_params["hidden_size"],
            lr=tr_params["lr"],
            batch_size=tr_params["batch_size"],
            alpha=tr_params["alpha"],
            inv_coeff=tr_params["inv_coeff"],
            var_coeff=tr_params["var_coeff"],
            cov_coeff=tr_params["cov_coeff"],
            n_layers=tr_params["n_layers"],
            maxlen=tr_params["maxlen"],
            dropout=tr_params["dropout"],
            embedding_dim=tr_params["embedding_dim"],
            n_sample=tr_params["n_sample"],
            step=tr_params["step"]
        )
    elif args.model_name == "PISA":
        knnModel = PISA(
            k=tr_params["k"],
            sample_size=tr_params["n_sample"],
            embed_dim=tr_params.get("embed_dim", 256),
            queue_size=tr_params.get("queue_size", 57600),
            momentum=tr_params.get("momentum", 0.995),
            session_key=tr_params.get("session_key", "SessionId"),
            item_key=tr_params.get("item_key", "ItemId"),
            time_key=tr_params.get("time_key", "Time")
        )
    else:
        raise ValueError(f"Unknown model name: {args.model_name}")

    # 4) 여러 GPU 감지 → DataParallel 적용 (PyTorch 모델만)
    if isinstance(knnModel, nn.Module):
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"[Info] Detected {device_count} GPUs. Using DataParallel with device_ids=range({device_count}).")
            knnModel = nn.DataParallel(knnModel, device_ids=list(range(device_count)))
            knnModel.to("cuda")
        else:
            knnModel.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    else:
        pass  # KNN 모델 등은 CPU or single GPU
    
    # 5) 테스트 세션(마지막 아이템)
    last_item = (
        df_train[df_train.SessionId.isin(data_manager.test_indices)]
        .sort_values("Time", ascending=False)
        .groupby("SessionId", as_index=False)
        .first()
    )
    all_tids = np.arange(data_manager.n_tracks)
    unknown_tracks = list(set(all_tids) - set(df_train.ItemId.unique()))
    test_to_last = array_mapping(data_manager.test_indices, last_item.SessionId.values)

    # 6) 학습(run_training) - MUSE 모델 예시에서 샘플링 적용
    #    (KNN계열에선 run_training(train=df_train) 형태로 그대로 호출)
    start_fit = time.time()
    print("Start fitting", args.model_name, "model")

    if isinstance(knnModel, nn.DataParallel):
        # DataParallel
        knnModel.module.run_training(train=df_train, tuning=False, savePath=savePath, sample_size=10000)  # ← 여기서 sample_size 사용하도록 MUSE 내부 수정
    else:
        knnModel.run_training(train=df_train, tuning=False, savePath=savePath)          # ← 동일

    end_fit = time.time()
    print("Training done in %.2f seconds" % (end_fit - start_fit))

    # 7) 예측
    print("Start predicting", args.model_name, "model")
    recos_knn = np.zeros((10000, 500), dtype=np.int64)

    if isinstance(knnModel, nn.DataParallel):
        predict_fn = knnModel.module.predict_next
    else:
        predict_fn = knnModel.predict_next

    for i, (pid, tid, t) in tqdm.tqdm(
        enumerate(last_item[["SessionId", "ItemId", "Time"]].values),
        total=len(last_item)
    ):
        pl_tracks = df_train[df_train.SessionId == pid].ItemId.values
        scores = predict_fn(pid, tid, all_tids)
        if len(scores.shape) != 1:
            scores = scores.flatten()

        pl_tracks_valid = pl_tracks[pl_tracks < len(scores)]
        unknown_tracks_valid = [idx for idx in unknown_tracks if idx < len(scores)]
        scores[pl_tracks_valid] = 0
        scores[unknown_tracks_valid] = 0

        top_k_scores = np.argsort(-scores)[:500]
        if len(top_k_scores) < 500:
            top_k_scores = np.pad(top_k_scores, (0, 500 - len(top_k_scores)), 'constant', constant_values=0)

        recos_knn[i] = top_k_scores

    save_path = f"resources/recos/{args.model_name}.npy"
    np.save(save_path, recos_knn)
    print(f"Predictions saved to {save_path}")
