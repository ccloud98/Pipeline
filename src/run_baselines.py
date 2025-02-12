import numpy as np
from src.data_manager.data_manager import DataManager, SequentialTrainDataset, EvaluationDataset, pad_collate, pad_collate_eval
from src.utils import array_mapping
from src.evaluator import Evaluator
import json, argparse
import pandas as pd

import scipy.sparse as sp
from scipy.sparse import lil_matrix, save_npz
import os

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

def convert_df_to_npz(df_path, out_path):
    """
    df_path: df_train_for_test.h5 파일 경로
    out_path: 생성할 playlist_track_custom.npz 경로
    """
    # (1) DF 로드
    df = pd.read_hdf(df_path, "abc")  # 'abc'는 HDF 내부 key (코드에 따라 다를 수 있음)

    # (2) pos를 오름차순 정렬
    df.sort_values(["SessionId", "Pos"], ascending=[True, True], inplace=True)

    # (3) session / item 값이 이미 0부터 시작하는지 확인
    #     아니라면 매핑해서 0-based로 만든다
    unique_sess = df["SessionId"].unique()
    unique_item = df["ItemId"].unique()

    sess_map = {old: new for new, old in enumerate(np.sort(unique_sess))}
    item_map = {old: new for new, old in enumerate(np.sort(unique_item))}

    df["mapped_sess"] = df["SessionId"].map(sess_map)
    df["mapped_item"] = df["ItemId"].map(item_map)

    n_sess = len(unique_sess)  # 세션 개수
    n_item = len(unique_item)  # 아이템(트랙) 개수

    # (4) LIL 매트릭스 생성
    mat = lil_matrix((n_sess, n_item), dtype=np.int32)

    for (s, i, p) in df[["mapped_sess","mapped_item","Pos"]].itertuples(index=False):
        # pos가 1부터 커지는 오름차순이라고 가정
        mat[s, i] = p

    # (5) CSR로 변환 후 저장
    mat_csr = mat.tocsr()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_npz(out_path, mat_csr)

    print(f"[INFO] Saved {mat_csr.shape[0]}x{mat_csr.shape[1]} matrix to {out_path}.")
    print("Example: mat[0,0] =", mat_csr[0,0])


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

    # (A) 샘플링 크기를 직접 지정할 수 있도록 인자 추가 (negative sampling 시 세션 개수 줄이기 등)
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Number of train playlists to sample for training (default=10000)")

    parser.add_argument("--extra_task", type=str, required=False, default=None,
                        help="extra task for DF->npz conversion or so")

    args = parser.parse_args()

    # 1) 하이퍼파라미터 로드
    with open(args.params_file, "r") as f:
        all_params = json.load(f)
    tr_params = all_params[args.model_name]

    # (추가) 만약 extra_task == "convert_df" 라면 DF->npz 만 실행 후 종료
    if args.extra_task == "convert_df":
        custom_npz_path = "resources/data/rta_input/playlist_track_custom.npz"
        convert_df_to_npz(df_path=f"{args.data_path}/df_train_for_test", out_path=custom_npz_path)
        print("[DONE] DF->NPZ conversion.")
        exit(0)

    # 2) DataManager 로드
    data_manager = DataManager(
        foldername="resources/data/", 
        resplit=True  # or False, 만약 이미 split을 완료했다면
    )
    df_train = pd.read_hdf(f"{args.data_path}/df_train_for_test")  # 실제 데이터
    savePath = f"{args.models_path}/{args.model_name}"

    # 3) 모델 생성
    if args.model_name == "VSKNN":
        Model = VMContextKNN(
            k=tr_params["k"],
            sample_size=tr_params["n_sample"],
            weighting=tr_params["w"],
            weighting_score=tr_params["w_score"],
            idf_weighting=tr_params["idf_w"]
        )
    elif args.model_name == "SKNN":
        Model = ContextKNN(
            k=tr_params["k"],
            sample_size=tr_params["n_sample"],
            similarity=tr_params["s"]
        )
    elif args.model_name == "VSTAN":
        # time 보정
        df_train["Time"] = df_train["Time"] / 1000
        Model = VSKNN_STAN(
            k=tr_params["k"],
            sample_size=tr_params["n_sample"],
            lambda_spw=tr_params["sp_w"],
            lambda_snh=tr_params["sn_w"],
            lambda_inh=tr_params["in_w"]
        )
    elif args.model_name == "STAN":
        Model = STAN(
            k=tr_params["k"],
            sample_size=tr_params["n_sample"],
            lambda_spw=tr_params["sp_w"],
            lambda_snh=tr_params["sn_w"],
            lambda_inh=tr_params["in_w"]
        )
    elif args.model_name == "MUSE":
        # 수정된 MUSE (negative sampling + multi-target)
        # => MUSE 내부에 'compute_loss_batch(x_pos, x_neg)' 등 구현
        Model = MUSE(
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
        Model = PISA(
            data_manager=data_manager,
            n_sample=tr_params["n_sample"],
            sampling=tr_params["sampling"],
            embed_dim=tr_params["embed_dim"],
            queue_size=tr_params["queue_size"],
            momentum=tr_params["momentum"],
            session_key=tr_params["session_key"],
            item_key=tr_params["item_key"],
            time_key=tr_params["time_key"],
            device="cuda",
            training_params=tr_params["training_params"]
        )
    elif args.model_name == "LARP":
        Model = LARP(
            data_manager=data_manager,
            n_sample=tr_params["n_sample"],
            embed_dim=tr_params["embed_dim"],
            fusion_method=tr_params["fusion_method"],
            ablation_loss=tr_params["ablation_loss"],
            queue_size=tr_params["queue_size"],
            momentum=tr_params["momentum"],
            training_params=tr_params["training_params"]
        )
    else:
        raise ValueError(f"Unknown model name: {args.model_name}")

    # 4) 여러 GPU 감지 → DataParallel 적용 (PyTorch 모델만)
    if isinstance(Model, nn.Module):
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"[Info] Detected {device_count} GPUs. Using DataParallel...")
            Model = nn.DataParallel(Model)
            Model.cuda() 
        else:
            # single GPU (or CPU)
            Model.to("cuda" if torch.cuda.is_available() else "cpu")

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

    # 6) 학습(run_training)
    start_fit = time.time()
    print("Start fitting", args.model_name, "model")

    if hasattr(Model, "fit") and args.model_name in ("SKNN", "VSKNN", "STAN", "VSTAN"):
        print("Training KNN-based model.")
        Model.fit(df_train) 
    else:
        # MUSE, PISA 등은 run_training으로 학습
        print("Training CounterPart model.")
        # DataParallel인 경우 module로 접근
        if isinstance(Model, nn.DataParallel):
            Model.module.run_training(train=df_train, tuning=False, savePath=savePath, sample_size=args.sample_size)
        else:
            Model.run_training(train=df_train, tuning=False, savePath=savePath, sample_size=args.sample_size)
    end_fit = time.time()
    print(f"Training done in {end_fit - start_fit:.2f} seconds.")

    # 7) 예측
    Model.eval()
    test_dataset = EvaluationDataset(
        data_manager=data_manager,
        indices=data_manager.test_indices
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        collate_fn=pad_collate_eval
    )
    print("Start predicting", args.model_name, "model")
    
    # (3) Batch 추론
    if isinstance(Model, nn.DataParallel):
        recos = Model.module.compute_recos(test_dataloader, n_recos=500)
    else:
        recos = Model.compute_recos(test_dataloader, n_recos=500)

    # (5) 저장
    save_path = f"resources/recos/{args.model_name}.npy"
    np.save(save_path, recos)
    print(f"Predictions saved to {save_path}")



