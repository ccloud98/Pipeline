import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from src.data_manager.data_manager import SequentialTrainDataset, pad_collate

class LARP(nn.Module):
    def __init__(
        self,
        data_manager,
        n_sample=None,
        k=1500,
        hidden_size=256,
        n_layers=4,
        num_heads=4,
        intermediate_size=1024,
        method="average",
        queue_size=2000,
        momentum=0.995,
        lr=0.001,               # 필요하다면 기본값
        alpha=0.5,              # 필요하다면 기본값
        batch_size=128,          # 필요하다면 기본값
        training_params=None
    ):
        super().__init__()
        self.data_manager = data_manager
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 우선순위 1) 생성자 인자로 직접 받은 lr, alpha, batch_size
        # 우선순위 2) training_params 안에 있으면 그것으로 덮어씌울 수도 있음
        if training_params is None:
            training_params = {}
        # 예: 만약 training_params를 우선시하려면 get() 사용
        self.n_sample = n_sample
        self.lr         = training_params.get("lr", lr)
        self.alpha      = training_params.get("alpha", alpha)
        self.batch_size = training_params.get("batch_size", batch_size)
        
        # 남은 학습 하이퍼파라미터 기본값 설정
        self.training_params = {
            "lr":         self.lr,
            "wd":         training_params.get("wd", 1e-4),
            "n_epochs":   training_params.get("n_epochs", 30),
            "clip":       training_params.get("clip", 1.0),
            "early_patience": training_params.get("early_patience", 3),
            "factor":     training_params.get("factor", 0.5),
            "batch_size": self.batch_size,
            "sample_size":training_params.get("sample_size", None),
            "use_cosine": training_params.get("use_cosine", True),
            "cosine_tmax":training_params.get("cosine_tmax", 10),
            "cosine_emin":training_params.get("cosine_emin", 1e-5),
            "max_size":   training_params.get("max_size", 10),
        }

        # 실제 백본 (BERT + Momentum + Queue + CIP)
        self.model = LARPBackbone(
            num_tracks=k,               # 예: k를 num_tracks로 사용할 수도, 혹은 별도 인자
            hidden_size=hidden_size,
            method=method,
            queue_size=queue_size,
            momentum=momentum
        ).to(self.device)

    def run_training(self, train=None, tuning=False, savePath=None, sample_size=None):
        import numpy as np
        start_time = time.time()

        if sample_size is not None:
            self.training_params["sample_size"] = sample_size
        else:
        # 만약 None이면 전체 세션
            self.training_params["sample_size"] = None

        # train_indices/val_indices split
        if tuning:
            train_indices = self.data_manager.train_indices
            val_indices   = self.data_manager.val_indices
        else:
            train_indices = np.concatenate([self.data_manager.train_indices, self.data_manager.val_indices])
            val_indices   = self.data_manager.val_indices

        train_loader = self.make_dataloader(train_indices, sample_size=self.training_params["sample_size"])
        val_loader   = self.make_dataloader(val_indices, sample_size=None)

        optimizer, scheduler = self.prepare_optimizer()

        best_val_loss = float("inf")
        no_improve_count = 0
        best_epoch = 0
        n_epochs = self.training_params["n_epochs"]

        for epoch in range(1, n_epochs+1):
            print(f"=== Epoch {epoch}/{n_epochs} ===")
            print(f"Elapsed time: {time.time()-start_time:.0f} seconds")

            train_loss = self.train_epoch(train_loader, epoch, optimizer)
            val_loss   = self.val_epoch(val_loader)

            print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            if self.training_params.get("use_cosine", False):
                scheduler.step()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve_count = 0
                best_epoch = epoch
                if savePath:
                    torch.save(self.state_dict(), f"{savePath}_best.pth")
                    print(f"[EARLY STOP] Best model updated at epoch {epoch}, val_loss={val_loss:.4f}")
            else:
                no_improve_count += 1
                print(f"[EARLY STOP] No improvement count: {no_improve_count}/{self.training_params['early_patience']}")
                if no_improve_count >= self.training_params["early_patience"]:
                    print(f"[EARLY STOP] Stop training at epoch {epoch}, best epoch was {best_epoch}")
                    break

        if savePath:
            self.load_state_dict(torch.load(f"{savePath}_best.pth"))
            print(f"Training finished. Best model from epoch {best_epoch} loaded.")
        else:
            print(f"Training finished. Best epoch={best_epoch}")

    def make_dataloader(self, indices, sample_size=None):
        # MUSE와 유사
        dataset = SequentialTrainDataset(
            data_manager=self.data_manager,
            indices=indices,
            max_size=self.training_params["max_size"],
            n_neg=50,
            sample_size=sample_size
        )
        loader = DataLoader(
            dataset,
            batch_size=self.training_params["batch_size"],
            shuffle=True,
            collate_fn=pad_collate,
            num_workers=0
        )
        return loader

    def prepare_optimizer(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.training_params["lr"],
            weight_decay=self.training_params["wd"]
        )
        if self.training_params.get("use_cosine", False):
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.training_params["cosine_tmax"],
                eta_min=self.training_params["cosine_emin"]
            )
        else:
            from torch.optim.lr_scheduler import StepLR
            scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
        return optimizer, scheduler

    def train_epoch(self, dataloader, epoch, optimizer):
        self.model.train()
        epoch_loss = 0.0

        for batch_idx, (xx_pad, _, x_lens) in enumerate(tqdm(dataloader, desc=f"[Train Epoch {epoch}]")):
            optimizer.zero_grad()

            # (예시) dummy CIP inputs
            B = xx_pad.size(0)
            cls_input= torch.randn(B, self.model.hidden_size, device=self.device)
            track_feats= torch.randn(B, 5, self.model.hidden_size, device=self.device)
            track_idxs=  torch.zeros(B, 5, dtype=torch.long, device=self.device)
            track_masks= torch.zeros(B,5,dtype=torch.bool, device=self.device)
            track_lengths= torch.tensor([5]*B, device=self.device)

            batch_dict = {
                "cls_input": cls_input,
                "track_idxs": track_idxs,
                "track_feats": track_feats,
                "track_masks": track_masks,
                "track_lengths": track_lengths
            }
            out = self.model(
                batch_dict,
                alpha=self.alpha,
                loss_type={"wtc","tpc"},
                update_momentum=True,
                update_features=True
            )
            loss = out["loss"]
            loss.backward()

            if self.training_params["clip"] > 0:
                nn.utils.clip_grad_norm_(self.parameters(), self.training_params["clip"])
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(dataloader)
        return epoch_loss

    def val_epoch(self, dataloader):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (xx_pad, _, x_lens) in enumerate(dataloader):
                B = xx_pad.size(0)
                cls_input= torch.randn(B, self.model.hidden_size, device=self.device)
                track_feats= torch.randn(B, 5, self.model.hidden_size, device=self.device)
                track_idxs=  torch.zeros(B,5,dtype=torch.long, device=self.device)
                track_masks= torch.zeros(B,5,dtype=torch.bool, device=self.device)
                track_lengths= torch.tensor([5]*B, device=self.device)

                batch_dict = {
                    "cls_input": cls_input,
                    "track_idxs": track_idxs,
                    "track_feats": track_feats,
                    "track_masks": track_masks,
                    "track_lengths": track_lengths
                }
                out = self.model(
                    batch_dict,
                    alpha=self.alpha,
                    loss_type={"wtc","tpc"},
                    update_momentum=False,
                    update_features=False
                )
                val_loss += out["loss"].item()
        val_loss /= len(dataloader)
        return val_loss

    def compute_recos(self, test_dataloader, n_recos=500):
        """
        PISA/MUSE style batch inference
        """
        device= self.device
        n_p= len(test_dataloader.dataset)
        recos= np.zeros((n_p, n_recos), dtype=np.int64)
        idx_start= 0

        self.model.eval()
        with torch.no_grad():
            for batch_idx, xx_pad in enumerate(test_dataloader):
                B= xx_pad.size(0)
                # dummy
                scores= torch.rand(B, self.model.num_tracks, device=device)
                _, top_inds= torch.topk(scores, k=n_recos, dim=1)
                top_inds= top_inds.cpu().numpy()
                recos[idx_start: idx_start+B]= top_inds
                idx_start+= B
        return recos

    def predict_next(self, session_id, input_item_id, all_items):
        """
        MUSE/PISA와 유사한 online 예측
        """
        scores= np.random.rand(len(all_items))
        return scores

###################################################
# BERT-like 부분(간소화) + Momentum/Queue + CIP
###################################################
class PlaylistConstructor(nn.Module):
    """
    간단한 CIP: average, soft_weight, etc.
    최종적으로 [B, D] 형태를 만들어 줘야
    einsum("bd,bd->b")가 가능.
    """
    def __init__(self, num_tracks=5000, method="average", hidden_size=256):
        super().__init__()
        self.method = method
        self.hidden_size = hidden_size
        if method == "soft_weight":
            self.soft_weights = nn.Embedding(num_tracks, 1)
            nn.init.ones_(self.soft_weights.weight)

    def forward(self, track_idxs, track_feats, track_masks, track_lengths):
        """
        track_idxs:   [B, N]
        track_feats:  [B, N, D]
        track_masks:  [B, N] (True=padding)
        track_lengths:[B]
        => return [B, D]
        """
        # mask: True=padding => 0.0 for aggregator
        track_feats = track_feats.masked_fill(track_masks.unsqueeze(-1), 0.0)
        # average
        feat_sum = track_feats.sum(dim=1) # [B, D]
        denom = track_lengths.unsqueeze(-1).clamp_min(1e-9).float()
        final_feat = feat_sum / denom     # [B, D]
        return final_feat


class LARPBackbone(nn.Module):
    """
    - BERT-like encoder (간소화)
    - Momentum encoder
    - CIP buffer, Queue 등
    """
    def __init__(self, num_tracks=5000, hidden_size=256, method="average", queue_size=57600, momentum=0.995):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_tracks  = num_tracks
        self.queue_size  = queue_size
        self.momentum    = momentum

        # (1) BERT (간략)
        self.bert_main = nn.Linear(hidden_size, hidden_size) # dummy
        self.bert_m    = nn.Linear(hidden_size, hidden_size, bias=False)
        # Projection
        self.proj_main = nn.Linear(hidden_size, hidden_size)
        self.proj_m    = nn.Linear(hidden_size, hidden_size)

        # Momentum pairs
        self.model_pairs = [
            (self.bert_main, self.bert_m),
            (self.proj_main, self.proj_m)
        ]

        # (2) Queue
        self.register_buffer("text_queue", torch.randn(hidden_size, queue_size))
        self.register_buffer("queue_ptr",  torch.zeros(1, dtype=torch.long))
        nn.init.normal_(self.text_queue, 0, 0.01)

        # (3) CIP buffer => track마다 임베딩 저장
        self.register_buffer("text_features_all", torch.zeros(num_tracks, hidden_size))

        # CIP constructor
        self.cip_agg = PlaylistConstructor(num_tracks=num_tracks, method=method, hidden_size=hidden_size)

        # Momentum init
        for (m_main, m_m) in self.model_pairs:
            for p_m, p_s in zip(m_m.parameters(), m_main.parameters()):
                p_m.data.copy_(p_s.data)
                p_m.requires_grad=False

        # learnable temp
        self.temp = nn.Parameter(torch.tensor(0.07))

    def forward(self, batch_dict, alpha=0.5, loss_type={"wtc","tpc"}, update_momentum=True, update_features=True):
        """
        batch_dict: {
          "cls_input": [B, D],   # dummy
          "track_idxs":[B, N],
          "track_feats":[B, N, D],
          "track_masks":[B, N],
          "track_lengths":[B]
        }
        """
        cls_input    = batch_dict["cls_input"]        # [B, D]
        track_idxs   = batch_dict["track_idxs"]       # [B, N]
        track_feats  = batch_dict["track_feats"]      # [B, N, D]
        track_masks  = batch_dict["track_masks"]      # [B, N]
        track_lengths= batch_dict["track_lengths"]     # [B]

        # (1) main BERT
        x_main = self.bert_main(cls_input)   # [B, D]
        x_main = F.relu(x_main)
        cls_main= self.proj_main(x_main)     # [B, D]
        cls_main= F.normalize(cls_main, dim=-1)

        # CIP buffer 업데이트
        if update_features:
            # 간단히 track_idxs[:,0] 부분만 업데이트
            self.text_features_all[ track_idxs[:,0] ] = cls_main.detach()

        # momentum
        if update_momentum:
            self._momentum_update()
        with torch.no_grad():
            x_m = self.bert_m(cls_input)
            x_m = F.relu(x_m)
            cls_m= self.proj_m(x_m)
            cls_m= F.normalize(cls_m, dim=-1)
            # queue
            all_m= torch.cat([cls_m.t(), self.text_queue.clone().detach()], dim=1) # [D, B+queue]
            sim_m= torch.einsum("bd,dk->bk", cls_m, all_m) / self.temp
            sim_targets= torch.zeros_like(sim_m)
            idx_arange= torch.arange(cls_m.size(0), device=cls_m.device)
            sim_targets.scatter_(1, idx_arange.unsqueeze(-1), 1.0)
            sim_m_tg= alpha * F.softmax(sim_m, dim=1) + (1-alpha)*sim_targets

        sim_main= torch.einsum("bd,dk->bk", cls_main, all_m)/self.temp

        losses = {}
        total_loss = torch.tensor(0.0, device=cls_main.device)

        # (2) wtc
        if "wtc" in loss_type:
            wtc_loss = -torch.sum(F.log_softmax(sim_main, dim=1)*sim_m_tg, dim=1).mean()
            losses["wtc"]= wtc_loss
            total_loss += wtc_loss

        if update_momentum:
            self._dequeue_and_enqueue(cls_m)

        # (3) tpc => CIP aggregator => [B, D]
        if "tpc" in loss_type:
            # aggregator
            cip_vec= self.cip_agg(track_idxs, track_feats, track_masks, track_lengths)  # [B, D]
            # dot => [B]
            sim_cip= torch.einsum("bd,bd->b", cls_main, cip_vec)/self.temp
            tpc_loss= -torch.mean(F.logsigmoid(sim_cip))
            losses["tpc"]= tpc_loss
            total_loss += tpc_loss

        losses["loss"] = total_loss
        return losses

    @torch.no_grad()
    def _momentum_update(self):
        for (m_main, m_m) in self.model_pairs:
            for p_m, p_s in zip(m_m.parameters(), m_main.parameters()):
                p_m.data = p_m.data*self.momentum + p_s.data*(1-self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, cls_m):
        B, D = cls_m.shape
        ptr = int(self.queue_ptr.item())
        end_ptr = ptr + B
        if end_ptr > self.queue_size:
            remain = self.queue_size - ptr
            self.text_queue[:, ptr:] = cls_m[:remain].T
            ptr = 0
            cls_m = cls_m[remain:]
            end_ptr = B - remain

        # 남은 부분 enqueue
        if cls_m.size(0) > 0:
            self.text_queue[:, ptr:end_ptr] = cls_m.T
            ptr = end_ptr % self.queue_size
        self.queue_ptr[0] = ptr


###################################################
# LARP: MUSE/PISA와 비슷한 구조
###################################################

