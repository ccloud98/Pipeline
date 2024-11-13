import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, optim
from tqdm import tqdm


class MUSE:
    def __init__(self, n_items, hidden_size, lr, device):
        self.n_items = n_items
        self.hidden_size = hidden_size
        self.device = device

        # 모델을 DataParallel로 감싸서 여러 GPU를 사용할 수 있도록 설정
        self.model = VICReg(self.n_items, self.hidden_size, self.device)

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)  # device_ids를 명시하지 않고 자동으로 사용
        self.model = self.model.to(self.device)  # 모델을 지정된 장치로 이동

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_func = nn.CrossEntropyLoss(reduction='mean')  # 손실 계산 방식을 'mean'으로 변경


    def fit(self, train_dataloader, epochs):
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_idx, (inputs_padded, targets, len_str) in enumerate(tqdm(train_dataloader)):
                self.optimizer.zero_grad()

                # 입력 데이터를 DataParallel이 GPU에 적절히 분산할 수 있도록 처리
                inputs_padded = inputs_padded.cuda()
                len_str = len_str.cuda()
                targets = targets.cuda()

                 # 모델의 forward 메서드 호출, 두 개의 값을 반환 (hidden, preds)
                hidden, preds = self.model(inputs_padded.to(self.device), len_str.to(self.device))

                targets = targets.to(self.device)

                # 다중 타겟이 아니라 단일 타겟으로 변환
                targets = torch.argmax(targets, dim=1)

                # 손실 계산 및 역전파
                loss = self.loss_func(preds, targets)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_dataloader)}")

    @torch.no_grad()
    def predict_next(self, session_id, input_item_id, predict_for_item_ids):
        self.model.eval()

        # 입력 배치 준비
        input_batch = self.prepare_input(session_id, input_item_id, predict_for_item_ids)

        # 모델을 통한 예측
        _, predictions = self.model(input_batch['orig_sess'], input_batch['lens'])

        # logits 계산 후 소프트맥스 확률 반환
        logits = self.predict(predictions)
        return logits.cpu().numpy()

    def calculate_loss(self, predictions, batch):
        # 아이템 임베딩을 활용한 예측 손실 계산
        if isinstance(self.model, nn.DataParallel):
            all_embs = self.model.module.backbone.item_embedding.weight
        else:
            all_embs = self.model.backbone.item_embedding.weight
        logits = torch.matmul(predictions, all_embs.transpose(0, 1))
        loss = self.loss_func(logits, batch['labels'])  # 타겟은 batch의 labels
        return loss


    def predict(self, predictions):
        # 모델의 예측 확률을 계산하여 반환
        if isinstance(self.model, nn.DataParallel):
            all_embs = self.model.module.backbone.item_embedding.weight
        else:
            all_embs = self.model.backbone.item_embedding.weight
        logits = torch.matmul(predictions, all_embs.transpose(0, 1))
        logits = F.softmax(logits, dim=1)  # 소프트맥스 확률 계산
        return logits


    def prepare_input(self, session_id, input_item_id, predict_for_item_ids):
        # 입력 데이터를 준비하는 메서드
        input_batch = {
            'orig_sess': torch.tensor([input_item_id]).unsqueeze(0).to(self.device),  # 세션 내 아이템
            'lens': torch.tensor([1]).to(self.device),  # 시퀀스 길이
            'labels': torch.tensor(predict_for_item_ids).to(self.device)  # 예측할 아이템들
        }
        return input_batch


class VICReg(nn.Module):
    def __init__(self, n_items, hidden_size, device):
        super().__init__()
        self.backbone = SRGNN(n_items, hidden_size, device)
    
    def forward(self, batch, len_str):
        # 배치에서 입력 시퀀스 및 길이 정보 추출
        hidden, preds = self.backbone(batch, len_str)
        return hidden, preds

    def compute_finegrained_matching_loss(self, batch, seq_hidden1, seq_hidden2, seq_pred1, seq_pred2, epoch):
        loss = 0.0
        mask1 = batch['orig_sess'].gt(0)
        mask2 = batch['aug1'].gt(0)
        mask = torch.cat([mask1.unsqueeze(0), mask2.unsqueeze(0)], dim=0)

        seq_hidden = torch.cat([seq_hidden1.unsqueeze(0),
                                seq_hidden2.unsqueeze(0)], dim=0)
        seq_pred = torch.cat([seq_pred1.unsqueeze(0), seg_pred2.unsqueeze(0)], dim=0)

        v1_position = torch.arange(self.args.maxlen).unsqueeze(0).repeat(
            batch['position_labels'].size(0), 1)
        v2_position = batch['position_labels'].masked_fill(
            batch['position_labels'] < 0, 0)
        locations = torch.cat([v1_position.unsqueeze(0),
                               v2_position.unsqueeze(0)], dim=0).to(self.device)

        # Global criterion
        if self.args.alpha < 1.0:
            inv_loss, var_loss, cov_loss = self.global_loss(seq_pred)
            loss = loss + (1 - self.args.alpha) * (inv_loss + var_loss + cov_loss)

        # Local criterion
        # Maps shape: B, C, H, W
        # With convnext actual maps shape is: B, H * W, C
        if self.args.alpha > 0.0:
            (maps_inv_loss, maps_var_loss, maps_cov_loss) = self.finegrained_matching_loss(seq_hidden, locations, mask, epoch)
            loss = loss + (self.args.alpha) * (
                maps_inv_loss + maps_var_loss + maps_cov_loss
            )
        return loss
        # 이 함수는 모델 학습에 사용됩니다.
        loss = 0.0
        # 추가적인 로직 필요 시 이곳에서 처리



class SRGNN(nn.Module):
    # SRGNN 구조 (간단화된 형태)
    def __init__(self, n_items, hidden_size, device):
        super(SRGNN, self).__init__()
        self.n_items = n_items
        self.hidden_size = hidden_size
        self.device = device

        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.gnn = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)

    def forward(self, batch, len_str):
        # item_seqs와 lengths 추출
        item_seqs = batch
        lengths = len_str

        item_seqs = torch.clamp(item_seqs, min=0, max=self.n_items - 1)  # 범위를 0부터 n_items-1로 클램핑

        # item_seqs 인덱스 범위 출력 (디버깅용)
        #print(f"item_seqs min: {item_seqs.min()}, item_seqs max: {item_seqs.max()}")

        # 인덱스 범위 검사
        if item_seqs.max() >= self.n_items:
            raise ValueError(f"item_seqs contains invalid index: {item_seqs.max()} >= {self.n_items}")

        # item_seqs를 임베딩
        embeddings = self.item_embedding(item_seqs)

        # lengths를 CPU 텐서로 변환
        lengths_cpu = lengths.cpu()  # 이 부분에서 CUDA에서 CPU로 변환

        # pack_padded_sequence 호출
        packed_input = nn.utils.rnn.pack_padded_sequence(embeddings, lengths_cpu, batch_first=True, enforce_sorted=False)
        
        # GNN에 입력 전달
        packed_output, _ = self.gnn(packed_input)

        # 패딩 해제
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # 마지막 스텝의 예측 값 반환
        last_step = lengths.view(-1, 1, 1).expand(output.size(0), 1, output.size(2)) - 1
        last_output = output.gather(1, last_step).squeeze(1)
        
        return output, last_output



#  # 예시 사용법
#  if __name__ == '__main__':
#      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#      n_items = 10000
#      hidden_size = 100
#      lr = 0.001
#      muse = MUSE(n_items=n_items, hidden_size=hidden_size, lr=lr, device=device)
    
#      # 학습 예시 (train_dataloader는 미리 준비된 데이터 로더)
#      # muse.fit(train_dataloader, epochs=10)

#      # 예측 예시
#      session_id = 1
#      input_item_id = 123
#      predict_for_item_ids = [111, 222, 333]
#      predictions = muse.predict_next(session_id, input_item_id, predict_for_item_ids)
#      print(predictions)