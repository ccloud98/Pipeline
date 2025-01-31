import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.data_manager.data_manager import SequentialTrainDataset, pad_collate
from src.rta.utils import get_device

#########################################
# MUSE 클래스
#########################################
class MUSE(nn.Module):
    def __init__(self, data_manager,
                 k=1500, n_items=2252463, hidden_size=128, lr=0.001, batch_size=16,
                 alpha=0.5, inv_coeff=1.0, var_coeff=0.5, cov_coeff=0.25,
                 n_layers=1, maxlen=50, dropout=0.1, embedding_dim=256, n_sample=1000, step=1,
                 training_params=None):
        super().__init__()
        self.data_manager = data_manager
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 인스턴스 변수
        self.batch_size = batch_size
        self.n_sample = n_sample

        # config 딕셔너리에 필요한 파라미터 저장
        self.config = {
            "device": self.device,
            "hidden_size": hidden_size,
            "batch_size": batch_size,
            "alpha": alpha,
            "inv_coeff": inv_coeff,
            "var_coeff": var_coeff,
            "cov_coeff": cov_coeff,
            "n_layers": n_layers,
            "maxlen": maxlen
        }
        self.n_items = n_items
        self.lr = lr
        # VICReg, SRGNN 등에서 쓸 로스 계수
        self.alpha = alpha
        self.inv_coeff = inv_coeff
        self.var_coeff = var_coeff
        self.cov_coeff = cov_coeff

        # 학습 파라미터(training_params)
        if training_params is None:
            self.training_params = {
                'lr': self.lr,
                'wd': 1e-4,
                'mom': 0.9,
                'nesterov': True,
                'n_epochs': 10,
                'clip': 1.0,
                'patience': 3,
                'factor': 0.5,
                'max_size': 50,    # 수정
                'n_neg': 5,
                'batch_size': batch_size
            }
        else:
            self.training_params = training_params

        # 모델 및 Optimizer
        self.model = VICReg(self.n_items, self.config).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

    def run_training(self, train=None, tuning=False, savePath=None, sample_size=None):

        if tuning:
            train_indices = self.data_manager.train_indices
        else:
            train_indices = np.concatenate((self.data_manager.train_indices, self.data_manager.val_indices))

        dataset = SequentialTrainDataset(
            data_manager=self.data_manager,
            indices=train_indices,
            max_size=self.training_params["max_size"],
            sample_size=sample_size
        )

        train_dataloader = DataLoader(
            dataset,
            batch_size=self.training_params["batch_size"],
            shuffle=True,
            collate_fn=pad_collate,
            num_workers=0
        )

        if train is None:
            print("[Warning] No training data provided.")
            return

        # batch_size, max_size 설정
        self.training_params["batch_size"] = self.batch_size
        # n_sample은 "max_size"로도 쓰이지만,
        # 우리가 원하는 건 "샘플링할 세션 수"와 혼동이 있을 수 있음
        # 여기선 "sample_size"가 있다면 => Dataset에서 세션을 sample_size개로 제한
        if sample_size is not None:
            self.training_params["sample_size"] = sample_size
        else:
            self.training_params["sample_size"] = None

        # evaluator (optionally)
        if tuning:
            test_evaluator, test_dataloader = self.data_manager.get_test_data("val")
        else:
            test_evaluator, test_dataloader = self.data_manager.get_test_data("test")

        optimizer, scheduler, train_dataloader = self.prepare_training_objects(tuning)

        start_time = time.time()

        for epoch in range(self.training_params["n_epochs"]):
            print(f"=== Epoch {epoch+1}/{self.training_params['n_epochs']} ===")
            print(f"Elapsed time: {time.time() - start_time:.0f} seconds")

            avg_epoch_loss = self.train_epoch(train_dataloader, epoch)
            print(f"[Epoch {epoch+1}] Loss: {avg_epoch_loss:.4f}")

            # 스케줄러 스텝
            scheduler.step()

            # 에폭별 체크포인트 저장
            if savePath:
                torch.save(self.state_dict(), f"{savePath}_epoch{epoch+1}.pth")
                print(f"Model checkpoint saved at {savePath}_epoch{epoch+1}.pth")

        # 최종 모델 저장
        if savePath:
            torch.save(self.state_dict(), f"{savePath}.pth")
            print(f"Final model saved at {savePath}.pth")

    def prepare_training_objects(self, tuning):
        # train_indices
        if tuning:
            train_indices = self.data_manager.train_indices
        else:
            train_indices = np.concatenate((self.data_manager.train_indices, self.data_manager.val_indices))

        # sample_size
        sample_size = self.training_params.get("sample_size", None)

        # Dataset
        dataset = SequentialTrainDataset(
            self.data_manager,
            train_indices,
            max_size=self.training_params["max_size"],  # 시퀀스 길이 제한
            n_neg=self.training_params["n_neg"],
            sample_size=sample_size                     # <--- 세션 샘플링
        )

        loader = DataLoader(
            dataset,
            batch_size=self.training_params["batch_size"],
            shuffle=True,
            collate_fn=pad_collate,
            num_workers=0
        )

        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.training_params["lr"],
            weight_decay=self.training_params["wd"],
            momentum=self.training_params["mom"],
            nesterov=self.training_params["nesterov"]
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.training_params["patience"],
            gamma=self.training_params["factor"]
        )
        return optimizer, scheduler, loader

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"[Train Epoch {epoch+1}]")):
            self.optimizer.zero_grad()

            if "aug1" in batch:
                batch["aug1"] = batch["aug1"].to(self.device)
            else:
                batch["aug1"] = batch["orig_sess"]

            batch["orig_sess"] = batch["orig_sess"].to(self.device)
            batch["lens"] = batch["lens"].to(self.device)
            if "labels" in batch:
                batch["labels"] = batch["labels"].to(self.device)

            # Dual-view forward
            v1_hidden, v1_preds = self.model(batch, input_str='orig_sess', len_str='lens', get_last=True)
            v2_hidden, v2_preds = self.model(batch, input_str='aug1', len_str='lens', get_last=True)

            # Fine-grained matching loss
            matching_loss = self.model.compute_finegrained_matching_loss(
                batch, v1_hidden, v2_hidden, v1_preds, v2_preds, epoch
            )

            # Reconstruction loss
            rec_loss = self.calculate_loss(v1_preds, batch)

            # Global invariance/variance/covariance
            global_inv_loss, global_var_loss, global_cov_loss = self.model.global_loss([v1_preds, v2_preds])

            total_loss = (
                self.alpha * rec_loss.mean()
                + (1 - self.alpha) * matching_loss
                + global_inv_loss
                + global_var_loss
                + global_cov_loss
            )

            total_loss.backward()
            if self.training_params["clip"] > 0:
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.training_params["clip"])
            self.optimizer.step()

            epoch_loss += total_loss.item()

        avg_loss = epoch_loss / len(dataloader)
        return avg_loss

    def calculate_loss(self, preds, batch):
        if "labels" not in batch:
            # 라벨이 없으면 손실 0
            return torch.tensor(0.0, device=self.device)

        # 1) logits = [B, n_items]
        all_embs = self.model.backbone.item_embedding.weight  # [n_items, hidden_size]
        logits = torch.matmul(preds, all_embs.transpose(0, 1))  # [B, n_items]

        labels = batch["labels"]  # [B] shape
        # 2) 유효 범위 마스크 만들기
        valid_mask = (labels >= 0) & (labels < self.n_items)
        valid_indices = valid_mask.nonzero(as_tuple=False).view(-1)  # shape [#valid]

        # 유효 라벨이 하나도 없으면 손실 0 (또는 해당 미니배치를 스킵)
        if valid_indices.numel() == 0:
            return torch.tensor(0.0, device=self.device)

        # 3) logits/labels를 유효 index만큼 슬라이싱
        # logits: [B, n_items] → [#valid, n_items]
        # labels: [B] → [#valid]
        logits = logits[valid_indices, :]
        labels = labels[valid_indices]

        # 4) CrossEntropy
        loss = self.loss_func(logits, labels)

        return loss

    def predict_next(self, session_id, input_item_id, predict_for_item_ids):
        self.model.eval()
        input_batch = self.prepare_input(session_id, input_item_id)

        with torch.no_grad():
            _, preds = self.model(input_batch)
            all_embs = self.model.backbone.item_embedding.weight
            logits = torch.matmul(preds, all_embs.transpose(0, 1)).squeeze(0)
            scores = torch.softmax(logits, dim=-1).cpu().numpy()

        final_scores = np.zeros(len(predict_for_item_ids), dtype=np.float32)
        for i, gid in enumerate(predict_for_item_ids):
            if 0 <= gid < self.n_items:
                final_scores[i] = scores[gid]
        return final_scores

    def prepare_input(self, session_id, input_item_id):
        batch = {
            "orig_sess": torch.tensor([[input_item_id]], device=self.device),
            "lens": torch.tensor([1], device=self.device),
        }
        return batch

########################################
# GNN / SRGNN / VICReg
########################################

class GNN(nn.Module):
    """Gated GNN"""
    def __init__(self, embedding_dim, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = embedding_dim
        self.input_size = embedding_dim * 2
        self.gate_size = 3 * embedding_dim

        self.w_ih = nn.Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = nn.Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = nn.Parameter(torch.Tensor(self.gate_size))
        self.b_hh = nn.Parameter(torch.Tensor(self.gate_size))
        self.b_iah = nn.Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = nn.Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def forward(self, A, hidden):
        for _ in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden

    def GNNCell(self, A, hidden):
        B, n_nodes, _ = hidden.shape
        in_count = A[:, :, :n_nodes]
        out_count = A[:, :, n_nodes: 2*n_nodes]

        input_in = torch.matmul(in_count, self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(out_count, self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], dim=2)

        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, dim=2)
        h_r, h_i, h_n = gh.chunk(3, dim=2)

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy


class SRGNN(nn.Module):
    def __init__(self, input_size, args):
        super(SRGNN, self).__init__()
        self.n_items = input_size
        self.args = args
        self.device = args["device"]
        self.hidden_size = args["hidden_size"]
        self.n_layers = args["n_layers"]

        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.gnn = GNN(self.hidden_size, step=self.n_layers)
        self.linear1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear3 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)

        self._init_weights()

    def _init_weights(self):
        stdv = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-stdv, stdv)

    def forward(self, batch, input_str='orig_sess', len_str='lens', get_last=True):
        seqs = batch[input_str]  # [B, max_len]
        lengths_t = batch[len_str]  # [B]

        alias_inputs, A, items, mask = self._get_slice(seqs)

        hidden = self.item_embedding(items)  # [B, n_nodes, hidden]
        hidden = self.gnn(A, hidden)

        if alias_inputs.numel() > 0:
            alias_inputs = alias_inputs.view(alias_inputs.size(0), -1, 1).expand(-1, -1, self.hidden_size)
        else:
            alias_inputs = torch.zeros(1, 1, self.hidden_size, device=self.device)

        alias_inputs = alias_inputs.long()
        seq_hidden = torch.gather(hidden, 1, alias_inputs)
        ht = self.get_last_item(seq_hidden, lengths_t)

        q1 = self.linear1(ht).view(ht.size(0), 1, ht.size(1))
        q2 = self.linear2(seq_hidden)
        alp = self.linear3(torch.sigmoid(q1 + q2))

        a = torch.sum(alp * seq_hidden * mask.unsqueeze(-1).float(), dim=1)
        seq_output = self.linear_transform(torch.cat([a, ht], dim=1))
        return seq_hidden, seq_output

    def _get_slice(self, seqs):
        mask = seqs.gt(0)  # [B, seq_len]
        max_n_nodes = seqs.size(1)
        seqs_np = seqs.cpu().numpy()

        alias_inputs = []
        A_list = []
        items_list = []

        max_alias_len = 0

        for seq in seqs_np:
            valid_idx = np.where(seq > 0)[0]
            if len(valid_idx) == 0:
                # 빈 세션
                items_pad = [0]*max_n_nodes
                u_A_ = np.zeros((max_n_nodes, 2*max_n_nodes), dtype=np.float32)
                A_list.append(u_A_)
                items_list.append(items_pad)
                alias_inputs.append([0])
                continue

            node = np.unique(seq[valid_idx])
            node = np.clip(node, 0, self.n_items-1)

            u_A = np.zeros((max_n_nodes, max_n_nodes), dtype=np.float32)
            for i in range(len(valid_idx)-1):
                cur_item = seq[valid_idx[i]]
                nxt_item = seq[valid_idx[i+1]]

                u_candidates = np.where(node == cur_item)[0]
                v_candidates = np.where(node == nxt_item)[0]
                if len(u_candidates)==0 or len(v_candidates)==0:
                    continue
                u = u_candidates[0]
                v = v_candidates[0]
                u_A[u][v] = 1

            u_sum_in = np.sum(u_A, axis=0)
            u_sum_in[u_sum_in==0] = 1
            u_A_in = u_A / u_sum_in

            u_sum_out = np.sum(u_A, axis=1)
            u_sum_out[u_sum_out==0] = 1
            u_A_out = (u_A.T / u_sum_out).T

            u_A_ = np.concatenate([u_A_in, u_A_out], axis=0).T
            items_pad = list(node) + [0]*(max_n_nodes - len(node))

            alias = []
            for idx_ in valid_idx:
                item_ = seq[idx_]
                cand = np.where(node == item_)[0]
                if len(cand)>0:
                    alias.append(cand[0])
                else:
                    alias.append(0)

            if len(alias) > max_alias_len:
                max_alias_len = len(alias)

            A_list.append(u_A_)
            items_list.append(items_pad)
            alias_inputs.append(alias)

        for i in range(len(alias_inputs)):
            if len(alias_inputs[i]) < max_alias_len:
                alias_inputs[i] += [0]*(max_alias_len - len(alias_inputs[i]))

        alias_inputs = torch.LongTensor(alias_inputs).to(self.device)
        A = torch.FloatTensor(np.array(A_list)).to(self.device)
        items = torch.LongTensor(np.array(items_list)).to(self.device)

        mask = mask.to(self.device)

        return alias_inputs, A, items, mask

    def get_last_item(self, seq_hidden, lengths_t):
        idx = (lengths_t - 1).view(-1,1,1).expand(-1,1,seq_hidden.size(2))
        return seq_hidden.gather(1, idx).squeeze(1)

class VICReg(nn.Module):
    def __init__(self, input_size, config):
        super().__init__()
        self.n_items = input_size
        self.config = config
        self.device = config["device"]
        self.hidden_size = config["hidden_size"]
        self.alpha = config["alpha"]
        self.inv_coeff = config["inv_coeff"]
        self.var_coeff = config["var_coeff"]
        self.cov_coeff = config["cov_coeff"]

        self.backbone = SRGNN(input_size, config)
        self.mask_default = self.mask_correlated_samples(batch_size=config["batch_size"])

    def forward(self, batch, input_str='orig_sess', len_str='lens', get_last=True):
        hidden, preds = self.backbone(batch, input_str, len_str, get_last)
        return hidden, preds

    def compute_finegrained_matching_loss(self, batch, seq_hidden1, seq_hidden2, seq_pred1, seq_pred2, epoch):
        return self.inv_coeff * F.mse_loss(seq_hidden1, seq_hidden2)

    def global_loss(self, embeddings):
        inv_loss = 0.0
        count = 0
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                inv_loss += F.mse_loss(embeddings[i], embeddings[j])
                count += 1
        if count > 0:
            inv_loss /= count
        inv_loss = self.inv_coeff * inv_loss

        var_loss = 0.0
        cov_loss = 0.0
        for emb in embeddings:
            x = emb - emb.mean(dim=0)
            std_x = torch.sqrt(x.var(dim=0) + 1e-4)
            var_loss += torch.mean(F.relu(1.0 - std_x))
            cov_x = (x.T @ x) / (x.size(0)-1)
            cov_loss += self.off_diagonal(cov_x).pow(2).sum().div(self.hidden_size)

        n_view = len(embeddings)
        var_loss = self.var_coeff * (var_loss / n_view)
        cov_loss = self.cov_coeff * (cov_loss / n_view)
        return inv_loss, var_loss, cov_loss

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n-1, n+1)[:, 1:].flatten()

    def mask_correlated_samples(self, batch_size):
        N = 2*batch_size
        mask = torch.ones((N,N), dtype=bool)
        mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i,batch_size+i] = 0
            mask[batch_size+i,i] = 0
        return mask
