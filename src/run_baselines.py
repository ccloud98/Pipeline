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
from torch.utils.data import DataLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type = str, required = True,
                    help = "Name of model to train")
    parser.add_argument("--data_path", type = str, required = False,
                    help = "path to data", default="resources/data/baselines")
    parser.add_argument("--params_file", type = str, required = False,
                    help = "file for parameters", default="resources/params/best_params_baselines.json")
    parser.add_argument("--models_path", type = str, required = False,
                    help = "Path to save models", default = "resources/recos")
    args = parser.parse_args()
    with open(args.params_file, "r") as f:
      p = json.load(f)

    tr_params = p[args.model_name]

    data_manager = DataManager()

    df_train = pd.read_hdf("%s/df_train_for_test" % args.data_path)
    
    savePath = "%s/%s" % (args.models_path, args.model_name)

    if args.model_name == "VSKNN":
        knnModel = VMContextKNN(k=tr_params["k"], sample_size=tr_params["n_sample"], weighting=tr_params["w"], weighting_score=tr_params["w_score"],  idf_weighting=tr_params["idf_w"])

    if args.model_name == "SKNN":
        knnModel = ContextKNN(k=tr_params["k"], sample_size=tr_params["n_sample"], similarity= tr_params["s"])

    if args.model_name == "VSTAN":
        df_train["Time"] = df_train["Time"] / 1000 # necessary to avoid overflow
        knnModel = VSKNN_STAN(k=tr_params["k"], sample_size=tr_params["n_sample"], lambda_spw=tr_params["sp_w"],  lambda_snh=tr_params["sn_w"], lambda_inh=tr_params["in_w"])

    if args.model_name == "STAN":
        knnModel = STAN(k=tr_params["k"], sample_size=tr_params["n_sample"], lambda_spw=tr_params["sp_w"],  lambda_snh=tr_params["sn_w"], lambda_inh=tr_params["in_w"])                                                                                     
    
    if args.model_name == "LARP":
        larpModel = LARP(k=tr_params["k"], sample_size=tr_params["n_sample"], embed_dim=tr_params.get("embed_dim", 256))
        knnModel = larpModel

    if args.model_name == "MUSE":
        knnModel = MUSE( data_manager, k=tr_params["k"], n_items=tr_params["n_items"], hidden_size=tr_params["hidden_size"], lr=tr_params["lr"], batch_size=tr_params["batch_size"], 
                         alpha=tr_params["alpha"], inv_coeff=tr_params["inv_coeff"], var_coeff=tr_params["var_coeff"], cov_coeff=tr_params["cov_coeff"],
                         n_layers=tr_params["n_layers"], maxlen=tr_params["maxlen"], dropout=tr_params["dropout"],
                         embedding_dim=tr_params["embedding_dim"], n_sample=tr_params["n_sample"], step=tr_params["step"] )
        
    if args.model_name == "PISA":
        pisaModel = PISA(
            k=tr_params["k"], sample_size=tr_params["n_sample"], embed_dim=tr_params.get("embed_dim", 256), queue_size=tr_params.get("queue_size", 57600), 
            momentum=tr_params.get("momentum", 0.995), session_key=tr_params.get("session_key", "SessionId"), item_key=tr_params.get("item_key", "ItemId"), time_key=tr_params.get("time_key", "Time")
        )
        knnModel = pisaModel

    last_item = df_train[df_train.SessionId.isin(data_manager.test_indices)].sort_values("Time", ascending=False).groupby("SessionId", as_index=False).first()
    all_tids = np.arange(data_manager.n_tracks)
    unknown_tracks = list(set(np.arange(data_manager.n_tracks)) - set(df_train.ItemId.unique()))

    gt_test = []
    for i in DataManager.N_SEED_SONGS:
      gt_test += data_manager.ground_truths["test"][i]

    n_recos = 500
    test_to_last = array_mapping(data_manager.test_indices, last_item.SessionId.values)

    start_fit = time.time()
    print("Start fitting" , args.model_name , "model")
    knnModel.run_training(train=df_train, tuning=False, savePath=savePath)
    end_fit = time.time()
    print("Training done in %.2f seconds" % (end_fit - start_fit))

    # Start predicting knn model
    print("Start predicting" , args.model_name , "model")
    recos_knn = np.zeros((10000, 500))

    for i, (pid, tid, t) in tqdm.tqdm(enumerate(last_item[["SessionId", "ItemId", "Time"]].values)):
        # 각 세션에 대한 트랙 목록 가져오기
        pl_tracks = df_train[df_train.SessionId == pid].ItemId.values
        
        # 모델을 통해 점수 예측
        scores = knnModel.predict_next(pid, tid, all_tids)

        # 예측값이 올바른 형태인지 확인
        if len(scores.shape) != 1:
            #print(f"Warning: scores expected to be 1-dimensional, but got shape {scores.shape}")
            # 다차원일 경우 1차원으로 변환
            scores = scores.flatten()

        # 유효한 인덱스 필터링 (scores의 길이 내에 있는 인덱스만 남김)
        pl_tracks_valid = pl_tracks[pl_tracks < len(scores)]
        unknown_tracks_valid = [idx for idx in unknown_tracks if idx < len(scores)]

        # 예측값 중 알려진 트랙 및 알려지지 않은 트랙의 점수를 0으로 설정하여 제외
        scores[pl_tracks_valid] = 0
        scores[unknown_tracks_valid] = 0

        # 상위 500개의 예측 결과를 저장
        top_k_scores = np.argsort(-scores)[:500]  # 상위 500개의 값 추출

        # 만약 top_k_scores가 500개보다 적다면, 나머지를 0으로 패딩
        if len(top_k_scores) < 500:
            top_k_scores = np.pad(top_k_scores, (0, 500 - len(top_k_scores)), 'constant', constant_values=0)

        # 최종적으로 500개의 값만 가지도록 저장
        recos_knn[i] = top_k_scores
        # 결과를 .npy 파일로 저장
    
    save_path = f"resources/recos/{args.model_name}.npy"
    np.save(save_path, recos_knn)
    print(f"Predictions saved to {save_path}")