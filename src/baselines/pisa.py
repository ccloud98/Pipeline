import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import tqdm
import time

from src.data_manager.data_manager import SequentialTrainDataset, pad_collate
from src.data_manager.data_manager import negative_sampler
from src.evaluator import Evaluator


class PISA(nn.Module):
    def __init__(
        self,
        data_manager,
        n_sample=5000,
        sampling="random",
        embed_dim=128,
        queue_size=4096,
        momentum=0.9,
        session_key="SessionId",
        item_key="ItemId",
        time_key="Time",
        device="cuda",
        training_params=None
    ):
        super(PISA, self).__init__()

        self.data_manager = data_manager
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        if training_params is None:
            training_params = {}
        self.training_params = training_params
        self.n_epochs   = training_params.get("n_epochs", 30)
        self.batch_size = training_params.get("batch_size", 16)
        self.lr         = training_params.get("lr", 0.001)
        self.wd         = training_params.get("wd", 1e-5)
        self.mom        = training_params.get("mom", 0.9)
        self.nesterov   = training_params.get("nesterov", True)
        self.n_neg      = training_params.get("n_neg", 10)
        self.max_size   = training_params.get("max_size", 50)
        self.clip       = training_params.get("clip", 5.0)

        self.patience   = self.training_params.get("patience", 3) 
        self.factor     = self.training_params.get("factor", 0.5) 
        self.step_size  = self.training_params.get("step_size", 1) 
        self.step_every = self.training_params.get("step_every", None)

        # 임베딩 차원
        self.embed_dim = embed_dim
        # 전체 아이템 개수 (+2 해서 padding_idx 용)
        self.n_items   = self.data_manager.n_tracks + 2  

        self.item_embedding = nn.Embedding(
            num_embeddings=self.n_items,
            embedding_dim=self.embed_dim,
            padding_idx=0
        )
        self.gru = nn.GRU(
            input_size=self.embed_dim,
            hidden_size=self.embed_dim,
            num_layers=1,
            batch_first=True
        )
        self.optimizer = None
        self.scheduler = None

    def forward(self, x):
        # x: [B, L]
        emb = self.item_embedding(x)   # [B, L, embed_dim]
        output, hidden = self.gru(emb)
        return output
    
    def chose_negative_examples(self, X_agg, x_neg, pad_mask):
        n_easy = self.n_neg // 2
        n_hard = self.n_neg - n_easy

        X_neg_rep_full = self.item_embedding(x_neg)  # [B, n_neg, D]
        easy_neg_rep   = X_neg_rep_full[:, :n_easy, :]  # [B, n_easy, D]

        # session representation => pad 기반 평균
        # X_agg : [B, seq_len, D]
        # pad_mask: [B, seq_len], True=유효
        X_agg_mean = padded_avg(X_agg, pad_mask)  # [B, D]

        # neg_prods = batch-wise 내적 => [B, n_neg]
        neg_prods = torch.einsum("bnd,bd->bn", X_neg_rep_full, X_agg_mean)
        # topk
        top_neg_indices = torch.topk(neg_prods, k=n_hard, dim=1)[1]  # [B, n_hard]
        # gather => item id
        # x_neg: [B, n_neg]
        hard_indices = torch.gather(x_neg, 1, top_neg_indices)
        hard_neg_rep = self.item_embedding(hard_indices)

        X_neg_final = torch.cat([easy_neg_rep, hard_neg_rep], dim=1)  # [B, n_neg, D]
        return X_neg_final

    def compute_loss_batch(self, x_pos, x_neg):
            x_pos = x_pos.to(self.device)
            x_neg = x_neg.to(self.device)
            # pad_mask: True=유효
            pad_mask = (x_pos != 0)

            # forward => [B, seq_len, D]
            rep_all = self.forward(x_pos.to(self.device))

            # split
            X_agg = rep_all[:, :-1, :]  # [B, seq_len-1, D]
            Y_pos = rep_all[:, 1:,  :]  # [B, seq_len-1, D]
            # pad_mask도 동일하게 shift
            pad_mask_agg = pad_mask[:, :-1]
            pad_mask_pos = pad_mask[:, 1:]

            # negative
            # easy+hard => [B, n_neg, D]
            X_neg_rep = self.chose_negative_examples(X_agg, x_neg, pad_mask_agg)

            # 2) pos_loss
            #   pos_prod: (X_agg * Y_pos).sum(-1) => [B, seq_len-1]
            pos_prod = torch.sum(X_agg * Y_pos, dim=2)  # [B, seq_len-1]
            # RTA style => padded_avg(-logsigmoid(pos_prod), pad_mask_pos)
            pos_loss = padded_avg(-F.logsigmoid(pos_prod), pad_mask_pos).mean()

            # 3) neg_loss
            #   aggregator mean => [B, D]
            X_agg_mean = padded_avg(X_agg, pad_mask_agg)  # [B, D]
            # neg_prod => [B, n_neg] = X_neg_rep dot X_agg_mean
            neg_prod = torch.einsum("bnd,bd->bn", X_neg_rep, X_agg_mean)
            neg_loss = torch.mean(-F.logsigmoid(-neg_prod))

            loss = pos_loss + neg_loss
            return loss

    def run_training(self, train, tuning=False, savePath=None, sample_size=None):
        print(f"[PISA] start training (tuning={tuning}), sample_size={sample_size}")

        if tuning:
            used_indices = self.data_manager.train_indices
        else:
            used_indices = np.concatenate((self.data_manager.train_indices,
                                           self.data_manager.val_indices))

        train_dataset = SequentialTrainDataset(
            data_manager=self.data_manager,
            indices=used_indices,
            max_size=self.max_size,
            n_neg=self.n_neg,
            sample_size=sample_size 
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=pad_collate,
            num_workers=0,
            pin_memory=True
        )
        val_indices = self.data_manager.val_indices
        val_dataset = SequentialTrainDataset(
            data_manager=self.data_manager,
            indices=val_indices,
            max_size=self.max_size,
            n_neg=self.n_neg,
            sample_size=None  # 검증은 전체 val 이용
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=pad_collate,
            num_workers=0,
            pin_memory=True
        )
        self.optimizer, self.scheduler = self.prepare_optimizer()
        self.to(self.device)
        self.train()

        best_val_loss = float("inf")
        best_epoch    = 0
        wait          = 0  # patience 관리를 위한 카운터

        start_time = time.time()
        batch_count = 0

        for epoch in range(self.n_epochs):
            print(f"[PISA] Epoch {epoch+1}/{self.n_epochs}, elapsed={time.time()-start_time:.1f}s")

            # === Training Loop ===
            total_loss = 0.0
            self.train()
            for xx_pad, yy_pad_neg, x_lens in tqdm.tqdm(train_loader):
                self.optimizer.zero_grad()

                loss = self.compute_loss_batch(xx_pad, yy_pad_neg)
                loss.backward()

                if self.clip is not None and self.clip > 0:
                    clip_grad_norm_(self.parameters(), max_norm=self.clip)

                self.optimizer.step()
                total_loss += loss.item()

                # (선택) step 마다 scheduler 갱신 & 중간 로그
                if self.step_every is not None and batch_count % self.step_every == 0:
                    self.scheduler.step()
                batch_count += 1

            # 한 epoch 끝난 뒤 평균 train_loss
            avg_train_loss = total_loss / len(train_loader)

            # === Validation Loop ===
            val_loss = self.evaluate(val_loader)  # 아래 evaluate() 참조

            # 스케줄러는 epoch 단위로도 한 번 step
            self.scheduler.step()

            # 로그 출력
            print(f"   >> train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}")

            # === Early Stopping ===
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch    = epoch + 1
                wait = 0
                # 모델 저장
                if savePath:
                    torch.save(self.state_dict(), savePath)
            else:
                wait += 1
                if wait >= self.patience:
                    print(f"[PISA] Early stopping triggered at epoch={epoch+1}")
                    break

        print(f"[PISA] Done training. total_time={time.time()-start_time:.1f}s")
        print(f"     Best val_loss={best_val_loss:.4f} at epoch={best_epoch}")
        return

    @torch.no_grad()
    def evaluate(self, val_loader):
        self.eval()
        losses = []
        for xx_pad, yy_pad_neg, x_lens in val_loader:
            loss = self.compute_loss_batch(xx_pad, yy_pad_neg)
            losses.append(loss.item())
        avg_loss = np.mean(losses)
        return avg_loss
    
    def prepare_optimizer(self):
        """Prepare optimizer and scheduler based on training parameters"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.training_params.get("lr", 0.001),
            weight_decay=self.training_params.get("wd", 1e-5)
        )
        
        if self.training_params.get("use_cosine", False):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.training_params.get("cosine_tmax", self.n_epochs),
                eta_min=self.training_params.get("cosine_emin", 1e-6)
            )
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, 
                step_size=self.training_params.get("step_size", 3), 
                gamma=self.training_params.get("gamma", 0.1)
            )
            
        return optimizer, scheduler

    def predict_next(self, session_id, input_item_id, all_items):
        # 1) 임시로 batch_size=1 형태로 tensor 구성
        x = torch.LongTensor([[input_item_id+1]])
        x = x.to(self.device)

        # 2) RNN forward
        sess_repr_all = self.forward(x)  # shape [1, L, D]
        # L=1이므로 sess_repr = sess_repr_all[:, -1, :] 가능
        sess_repr = sess_repr_all[:, -1, :]  # [1, D]

        # 3) 아이템 임베딩
        max_id = min(self.n_items, len(all_items))
        items_t = torch.LongTensor(all_items[:max_id] + 1).to(self.device)
        item_embs = self.item_embedding(items_t)  # [n_items, D]

        # 4) 점수=내적
        scores = torch.matmul(sess_repr, item_embs.transpose(0,1))  # [1, n_items]
        scores = scores.squeeze(0).detach().cpu().numpy()
        return scores

def padded_avg(tensor, mask):
    # tensor이 2D라면 [B, seq_len, 1]로 reshape
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(-1)  # => [B, seq_len, 1]

    mask_f = mask.float()  # [B, seq_len]
    denom  = torch.sum(mask_f, dim=1, keepdim=True).clamp_min(1e-9)
    # [B, seq_len, D] × [B, seq_len, 1]
    masked_tensor = tensor * mask_f.unsqueeze(-1)
    summed = masked_tensor.sum(dim=1)  # [B, D]
    avg   = summed / denom
    return avg