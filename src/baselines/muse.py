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
            for inputs_padded, targets, len_str in tqdm(train_dataloader):
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

                # 재구성 손실 및 정밀 매칭 손실 계산
                rec_loss = self.calculate_loss(preds, targets)
                matching_loss = self.model.compute_finegrained_matching_loss(inputs_padded, hidden, preds, epoch)
                loss = rec_loss + matching_loss

                # 손실 계산 및 역전파
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
        all_embs = self.model.backbone.item_embedding.weight
        logits = torch.matmul(predictions, all_embs.transpose(0, 1))
        loss = self.loss_func(logits, batch['labels'])  # 타겟은 batch의 labels
        return loss


    def predict(self, predictions):
        # 모델의 예측 확률을 계산하여 반환
        all_embs = self.model.module.backbone.item_embedding.weight
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
    def __init__(self, input_size, hidden_size, device, args):
        super().__init__()
        self.n_items = input_size
        self.args = args
        self.device = device

        self.num_features = hidden_size
        self.backbone = SRGNN(input_size, hidden_size, device)

        self.mask_default = self.mask_correlated_samples(batch_size=args.batch_size)
    
    def forward(self, batch, len_str):
        # 배치에서 입력 시퀀스 및 길이 정보 추출
        hidden, preds = self.backbone(batch, len_str)
        return hidden, preds
    
    def _inv_loss(self, x, y, loss_type):
        if loss_type.lower() == 'mse':
            repr_loss = F.mse_loss(x, y)
        elif loss_type.lower() == 'infonce':
            repr_logits, repr_labels = self.info_nce(x, y, self.args.temperature, x.size(0))
            repr_loss = F.cross_entropy(repr_logits, repr_labels)
        else:
            raise ValueError
        return repr_loss

    def _vicreg_loss(self, x, y):
        repr_loss = self.args.inv_coeff * F.mse_loss(x, y)

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = self.args.var_coeff * (
            torch.mean(F.relu(1.0 - std_x)) / 2 + torch.mean(F.relu(1.0 - std_y)) / 2
        )

        x = x.permute((1, 0, 2))
        y = y.permute((1, 0, 2))

        *_, sample_size, num_channels = x.shape
        non_diag_mask = ~torch.eye(num_channels, device=x.device, dtype=torch.bool)
        # Center features
        # centered.shape = NC
        x = x - x.mean(dim=-2, keepdim=True)
        y = y - y.mean(dim=-2, keepdim=True)

        cov_x = torch.einsum("...nc,...nd->...cd", x, x) / (sample_size - 1)
        cov_y = torch.einsum("...nc,...nd->...cd", y, y) / (sample_size - 1)
        cov_loss = (cov_x[..., non_diag_mask].pow(2).sum(-1) / num_channels) / 2 + (
            cov_y[..., non_diag_mask].pow(2).sum(-1) / num_channels
        ) / 2
        cov_loss = cov_loss.mean()
        cov_loss = self.args.cov_coeff * cov_loss

        return repr_loss, std_loss, cov_loss

    def _finegrained_matching_loss(self, maps_1, maps_2, location_1, location_2, mask1, mask2, j, epoch):
        inv_loss = 0.0
        var_loss = 0.0
        cov_loss = 0.0

        # item-based matching
        if epoch > self.args.warm_up_epoch:
            num_matches_on_l2 = self.args.num_matches

            maps_1_filtered, maps_1_nn = self.item_based_matching(
                maps_1, maps_2, num_matches=num_matches_on_l2[0], mask1=mask1, mask2=mask2
            )
            maps_2_filtered, maps_2_nn = self.item_based_matching(
                maps_2, maps_1, num_matches=num_matches_on_l2[1], mask1=mask1, mask2=mask2
            )

            inv_loss_1, var_loss_1, cov_loss_1 = self._vicreg_loss(maps_1_filtered, maps_1_nn)
            inv_loss_2, var_loss_2, cov_loss_2 = self._vicreg_loss(maps_2_filtered, maps_2_nn)
            var_loss = var_loss + (var_loss_1 / 2 + var_loss_2 / 2)
            cov_loss = cov_loss + (cov_loss_1 / 2 + cov_loss_2 / 2)
            inv_loss = inv_loss + (inv_loss_1 / 2 + inv_loss_2 / 2)

        maps_1_filtered, maps_1_nn = self.similarity_based_matching(
            location_1, location_2, maps_1, maps_2, mask1, mask2, j
        )

        inv_loss_1, var_loss_1, cov_loss_1 = self._vicreg_loss(maps_1_filtered, maps_1_nn)
        var_loss = var_loss + var_loss_1
        cov_loss = cov_loss + cov_loss_1
        inv_loss = inv_loss + inv_loss_1

        return inv_loss, var_loss, cov_loss

    def finegrained_matching_loss(self, maps_embedding, locations, mask, epoch):
        num_views = len(maps_embedding)
        inv_loss = 0.0
        var_loss = 0.0
        cov_loss = 0.0
        iter_ = 0
        for i in range(2):
            for j in np.delete(np.arange(np.sum(num_views)), i):
                inv_loss_this, var_loss_this, cov_loss_this = self._finegrained_matching_loss(
                    maps_embedding[i], maps_embedding[j], locations[i], locations[j], mask[i], mask[j], j, epoch
                )
                inv_loss = inv_loss + inv_loss_this
                var_loss = var_loss + var_loss_this
                cov_loss = cov_loss + cov_loss_this
                iter_ += 1

        inv_loss = inv_loss / iter_
        var_loss = var_loss / iter_
        cov_loss = cov_loss / iter_

        return inv_loss, var_loss, cov_loss

    def global_loss(self, embedding, maps=False):
        num_views = len(embedding)
        inv_loss = 0.0
        iter_ = 0
        for i in range(2):
            for j in np.delete(np.arange(np.sum(num_views)), i):
                inv_loss = inv_loss + F.mse_loss(embedding[i], embedding[j])
                iter_ = iter_ + 1
        inv_loss = self.args.inv_coeff * inv_loss / iter_

        var_loss = 0.0
        cov_loss = 0.0
        iter_ = 0
        for i in range(num_views):
            x = embedding[i]
            x = x - x.mean(dim=0)
            std_x = torch.sqrt(x.var(dim=0) + 0.0001)
            var_loss = var_loss + torch.mean(torch.relu(1.0 - std_x))
            cov_x = (x.T @ x) / (x.size(0) - 1)
            cov_loss = cov_loss + off_diagonal(cov_x).pow_(2).sum().div(
                self.args.embedding_dim
            )
            iter_ = iter_ + 1
        var_loss = self.args.var_coeff * var_loss / iter_
        cov_loss = self.args.cov_coeff * cov_loss / iter_

        return inv_loss, var_loss, cov_loss

    def compute_finegrained_matching_loss(self, batch, seq_hidden1, seq_hidden2, seq_pred1, seq_pred2, epoch):
        loss = 0.0
        mask1 = batch['orig_sess'].gt(0)
        mask2 = batch['aug1'].gt(0)
        mask = torch.cat([mask1.unsqueeze(0), mask2.unsqueeze(0)], dim=0)

        seq_hidden = torch.cat([seq_hidden1.unsqueeze(0),
                                seq_hidden2.unsqueeze(0)], dim=0)
        seq_pred = torch.cat([seq_pred1.unsqueeze(0), seq_pred2.unsqueeze(0)], dim=0)

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
        if self.args.alpha > 0.0:
            (maps_inv_loss, maps_var_loss, maps_cov_loss) = self.finegrained_matching_loss(seq_hidden, locations, mask, epoch)
            loss = loss + (self.args.alpha) * (
                maps_inv_loss + maps_var_loss + maps_cov_loss
            )
        return loss
        # 이 함수는 모델 학습에 사용됩니다.

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def info_nce(self, z_i, z_j, temperature, batch_size, sim='dot'):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * batch_size
    
        z = torch.cat((z_i, z_j), dim=0)
    
        if sim == 'cos':
            sim = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temperature
        elif sim == 'dot':
            sim = torch.mm(z, z.T) / temperature
    
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
    
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        if batch_size != self.args.batch_size:
            mask = self.mask_correlated_samples(batch_size)
        else:
            mask = self.mask_default
        negative_samples = sim[mask].reshape(N, -1)
    
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return logits, labels

    def nearest_neighbores(self, input_maps, candidate_maps, distances, num_matches):
        batch_size = input_maps.size(0)

        if num_matches is None or num_matches == -1:
            num_matches = input_maps.size(1)

        topk_values, topk_indices = distances.topk(k=1, largest=False)
        topk_values = topk_values.squeeze(-1)
        topk_indices = topk_indices.squeeze(-1)

        sorted_values, sorted_values_indices = torch.sort(topk_values, dim=1)
        sorted_indices, sorted_indices_indices = torch.sort(sorted_values_indices, dim=1)

        mask = torch.stack(
            [
                torch.where(sorted_indices_indices[i] < num_matches, True, False)
                for i in range(batch_size)
            ]
        )
        topk_indices_selected = topk_indices.masked_select(mask)
        topk_indices_selected = topk_indices_selected.reshape(batch_size, num_matches)

        indices = (
            torch.arange(0, topk_values.size(1))
            .unsqueeze(0)
            .repeat(batch_size, 1)
            .to(topk_values.device)
        )
        indices_selected = indices.masked_select(mask)
        indices_selected = indices_selected.reshape(batch_size, num_matches)

        filtered_input_maps = batched_index_select(input_maps, 1, indices_selected)
        filtered_candidate_maps = batched_index_select(
            candidate_maps, 1, topk_indices_selected
        )

        return filtered_input_maps, filtered_candidate_maps


    def item_based_matching(self, input_maps, candidate_maps, num_matches, mask1, mask2):
        """
        input_maps: (B, H * W, C)
        candidate_maps: (B, H * W, C)
        """
        distances = torch.cdist(input_maps, candidate_maps)
        mask_tensor1 = mask1.unsqueeze(1) * mask1.unsqueeze(-1)
        mask_tensor2 = mask2.unsqueeze(1) * mask2.unsqueeze(-1)
        mask_tensor = mask_tensor1 * mask_tensor2
        distances = distances.masked_fill(~mask_tensor, np.Inf)
        return self.nearest_neighbores(input_maps, candidate_maps, distances, num_matches)

    def similarity_based_matching(
            self, input_location, candidate_location, input_maps, candidate_maps, mask1, mask2, j
        ):
        # mask_tensor1 = mask1.unsqueeze(1) * mask1.unsqueeze(-1)
        # mask_tensor2 = mask2.unsqueeze(1) * mask2.unsqueeze(-1)
        if j == 1:
            perm_mat = candidate_location
            coverted_maps = candidate_maps
            mask = mask2
            # mask_tensor = mask_tensor2
        elif j == 0:
            perm_mat = input_location
            coverted_maps = input_maps
            mask = mask1
            # mask_tensor = mask_tensor1

        perm_mat = F.one_hot(perm_mat, num_classes=self.args.maxlen) * mask.unsqueeze(-1)
        zeros = torch.zeros_like(perm_mat).to(self.device)
        ones = (~mask).long()
        r = torch.arange(self.args.maxlen).to(self.device)
        zeros[:, r, r] = ones
        perm_mat += zeros

        candidate_maps = torch.matmul(coverted_maps.transpose(2, 1),
                                      perm_mat.float()).transpose(2, 1)
        return input_maps, candidate_maps



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

        # 범위를 벗어난 인덱스는 n_items - 1로 클램핑(clamp)합니다.
        item_seqs = torch.clamp(item_seqs, max=self.n_items - 1)

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
    
def batched_index_select(input, dim, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)



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
