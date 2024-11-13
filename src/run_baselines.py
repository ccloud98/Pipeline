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
    args = parser.parse_args()
    with open(args.params_file, "r") as f:
      p = json.load(f)

    tr_params = p[args.model_name]

    data_manager = DataManager()

    df_train = pd.read_hdf("%s/df_train_for_test" % args.data_path)
    if args.model_name == "VSKNN":
        knnModel = VMContextKNN(k=tr_params["k"], sample_size=tr_params["n_sample"], weighting=tr_params["w"], weighting_score=tr_params["w_score"],  idf_weighting=tr_params["idf_w"])

    if args.model_name == "SKNN":
        knnModel = ContextKNN(k=tr_params["k"], sample_size=tr_params["n_sample"], similarity= tr_params["s"])

    if args.model_name == "VSTAN":
        df_train["Time"] = df_train["Time"] / 1000 # necessary to avoid overflow
        knnModel = VSKNN_STAN(k=tr_params["k"], sample_size=tr_params["n_sample"], lambda_spw=tr_params["sp_w"],  lambda_snh=tr_params["sn_w"], lambda_inh=tr_params["in_w"])

    if args.model_name == "STAN":
        knnModel = STAN(k=tr_params["k"], sample_size=tr_params["n_sample"], lambda_spw=tr_params["sp_w"],  lambda_snh=tr_params["sn_w"], lambda_inh=tr_params["in_w"])

    if args.model_name == "MUSE":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # DataLoader 준비 (SequentialTrainDataset을 사용하여 데이터를 준비)
        sample_size = 10000
        train_dataset = SequentialTrainDataset(data_manager, data_manager.train_indices, sample_size=sample_size)
        train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=pad_collate, pin_memory=True) 
        

        # 데이터 로더 동작 검증 코드 추가
        print("데이터 로더 동작을 검증합니다. 첫 번째 배치를 가져옵니다...")
        for batch_idx, (inputs_padded, targets, len_str) in enumerate(train_dataloader):
            # GPU로 데이터 이동
            inputs_padded = inputs_padded.to(device)
            targets = targets.to(device)
            len_str = len_str.to(device)

            print(f"배치 {batch_idx + 1} 확인:")
            print(f"inputs_padded: {inputs_padded.shape}, device: {inputs_padded.device}")
            print(f"targets: {targets.shape}, device: {targets.device}")
            print(f"len_str: {len_str.shape}, device: {len_str.device}")
            # 첫 번째 배치만 확인 후 break
            break

        knnModel = MUSE(n_items=tr_params["n_items"], hidden_size=tr_params["hidden_size"], lr=tr_params["lr"], device=device) 

        epochs = tr_params.get("epochs", 10)

    knnModel.model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs} 시작...")
        epoch_loss = 0
        for batch_idx, (inputs_padded, targets, len_str) in enumerate(train_dataloader):
            knnModel.optimizer.zero_grad()

            # 입력 데이터를 DataParallel이 GPU에 적절히 분산할 수 있도록 처리
            inputs_padded = inputs_padded.to(device)
            targets = targets.to(device)
            len_str = len_str.to(device)

            # 모델의 forward 메서드 호출, 두 개의 값을 반환 (hidden, preds)
            hidden, preds = knnModel.model(inputs_padded, len_str)

            # 다중 타겟이 아니라 단일 타겟으로 변환
            targets = torch.argmax(targets, dim=1)

            # 손실 계산 및 역전파
            loss = knnModel.loss_func(preds, targets)
            loss.backward()
            knnModel.optimizer.step()
 
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_dataloader)}")



    last_item = df_train[df_train.SessionId.isin(data_manager.test_indices)].sort_values("Time", ascending=False).groupby("SessionId", as_index=False).first()
    all_tids = np.arange(data_manager.n_tracks)
    unknown_tracks = list(set(np.arange(data_manager.n_tracks)) - set(df_train.ItemId.unique()))

    gt_test = []
    for i in DataManager.N_SEED_SONGS:
      gt_test += data_manager.ground_truths["test"][i]

    n_recos = 500
    test_to_last = array_mapping(data_manager.test_indices, last_item.SessionId.values)

    start_fit = time.time()
    print("Start fitting knn model")
    if args.model_name == "MUSE":
        epochs = 10
        knnModel.fit(train_dataloader, epochs)  # DataLoader로 학습

    else:
        knnModel.fit(df_train)
    end_fit = time.time()
    print("Training done in %.2f seconds" % (end_fit - start_fit))

    # Start predicting knn model
    print("Start predicting knn model")
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
    np.save("resources/recos/MUSE.npy", recos_knn)
    print("Predictions saved to resources/MUSE.npy")