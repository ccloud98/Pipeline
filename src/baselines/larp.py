import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader


###################################################
# 0) BERT 구조: config, embedding, self-attn 등
###################################################
class BertConfig:
    """
    간단한 BERT 설정. 실제 Hugging Face BertConfig와 달리
    필요한 필드만 최소한으로 정의합니다.
    """
    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=512,
        type_vocab_size=2
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size


class BertEmbeddings(nn.Module):
    """
    - token_embeddings: [vocab_size, hidden_size]
    - position_embeddings: [max_position_embeddings, hidden_size]
    - token_type_embeddings: [type_vocab_size, hidden_size]
    - layernorm + dropout
    """
    def __init__(self, config: BertConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)
        self.config = config

    def forward(self, input_ids, token_type_ids=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words_emb = self.word_embeddings(input_ids)
        pos_emb   = self.position_embeddings(position_ids)
        toktype_emb = self.token_type_embeddings(token_type_ids)

        embeddings = words_emb + pos_emb + toktype_emb
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    """
    Self-Attention의 핵심 부분 (Q,K,V projection + Scaled Dot-Product)
    """
    def __init__(self, config: BertConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.all_head_dim = self.num_heads * self.head_dim

        self.query = nn.Linear(config.hidden_size, self.all_head_dim)
        self.key   = nn.Linear(config.hidden_size, self.all_head_dim)
        self.value = nn.Linear(config.hidden_size, self.all_head_dim)
        self.out   = nn.Linear(config.hidden_size, config.hidden_size)

    def transpose_for_scores(self, x):
        """
        x: [B, seq_len, all_head_dim]
        return: [B, num_heads, seq_len, head_dim]
        """
        B, L, D = x.size()
        x = x.view(B, L, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        Q = self.query(hidden_states)
        K = self.key(hidden_states)
        V = self.value(hidden_states)

        # reshape
        Q = self.transpose_for_scores(Q)
        K = self.transpose_for_scores(K)
        V = self.transpose_for_scores(V)

        # scaled dot-product
        attn_scores = torch.matmul(Q, K.transpose(-1, -2))
        attn_scores = attn_scores / math.sqrt(self.head_dim)

        # attention mask (padding)
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_probs = F.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_probs, V)  # [B, num_heads, seq_len, head_dim]

        # transpose back
        context = context.permute(0, 2, 1, 3).contiguous()
        new_shape = context.size()[:-2] + (self.all_head_dim,)
        context = context.view(*new_shape)

        # output projection
        out = self.out(context)
        return out


class BertAttention(nn.Module):
    """
    Self-attention + LayerNorm + Dropout
    """
    def __init__(self, config: BertConfig):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = nn.Linear(config.hidden_size, config.hidden_size)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states, attention_mask=None):
        attn_out = self.self(hidden_states, attention_mask)
        attn_out = self.dropout(attn_out)
        hidden_states = self.layernorm(hidden_states + attn_out)
        return hidden_states


class BertIntermediate(nn.Module):
    """
    FFN 전단계 (hidden -> intermediate)
    """
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    """
    FFN 후단계 (intermediate -> hidden)
    """
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layernorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    """
    1 Transformer Layer: Self-Attention + FFN
    """
    def __init__(self, config: BertConfig):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        # 1) self-attn
        attn_out = self.attention(hidden_states, attention_mask)
        # 2) ffn
        inter_out = self.intermediate(attn_out)
        layer_out = self.output(inter_out, attn_out)
        return layer_out


class BertEncoder(nn.Module):
    """
    n_layers 쌓아올린 encoder
    """
    def __init__(self, config: BertConfig):
        super().__init__()
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states


class BertModelCustom(nn.Module):
    """
    전체 BERT 모델 (Embeddings + Encoder).
    Pooling or CLS-vector만 따로 구성할 수도 있음.
    """
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        # (선택) Pooler가 필요하면 추가

        # 임의로 초기화
        self.apply(self._init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        # 0 or 1 mask -> additive mask
        if attention_mask is not None:
            # mask: [B, L], 1=유효, 0=padding
            # -> shape [B,1,1,L], 값=0 or -inf
            extended_mask = (1.0 - attention_mask) * -10000.0
            extended_mask = extended_mask.unsqueeze(1).unsqueeze(2)
        else:
            extended_mask = None

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoder_output = self.encoder(embedding_output, extended_mask)
        # encoder_output: [B, L, hidden]
        return encoder_output

    def _init_bert_weights(self, module):
        """
        임의 초기화 (xavier_uniform 등).
        실제 BERT는 더 복잡한 초기화를 사용합니다.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_uniform_(module.weight)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


###################################################
# 1) CIP & Playlist_Constructor
###################################################
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embedding_size=768):
        super().__init__()
        self.w_q = nn.Linear(embedding_size, embedding_size, bias=False)
        self.w_k = nn.Linear(embedding_size, embedding_size, bias=False)
        self.w_v = nn.Linear(embedding_size, embedding_size, bias=False)
        self.ln  = nn.LayerNorm(embedding_size)

    def forward(self, features, mask):
        # features: [B, N, D]
        # mask: [B, N], True=padding
        # 간단히 self-attn 예시
        B, N, D = features.shape
        features = self.ln(features)
        Q = self.w_q(features)
        K = self.w_k(features)
        V = self.w_v(features)

        scale = D**-0.5
        attn_scores = torch.einsum("bnd,bmd->bnm", Q, K) * scale
        # mask 처리
        # mask: True=padding => -1e9로
        # => shape [B,N] -> [B,1,N]
        big_mask = mask.unsqueeze(1)
        attn_scores = attn_scores.masked_fill(big_mask, float("-inf"))
        attn_probs = F.softmax(attn_scores, dim=-1)
        ctx = torch.einsum("bnm,bmd->bnd", attn_probs, V)
        return ctx


class Playlist_Constructor(nn.Module):
    """
    간단 CIP 예시:
      - average / soft_weight / self_attn
    """
    def __init__(self, num_tracks=10000, method="average", embedding_size=768):
        super().__init__()
        self.method = method
        self.embedding_size = embedding_size
        if method == "soft_weight":
            self.soft_weights = nn.Embedding(num_tracks, 1)
            nn.init.ones_(self.soft_weights.weight)
        elif method == "self_attn":
            self.self_attn = TransformerEncoderLayer(embedding_size)

    def forward(self, track_idx_seq, track_feat_seq, mask, length):
        """
        track_idx_seq: [B, N]
        track_feat_seq: [B, N, D]
        mask: [B, N], True=padding
        length: [B]
        """
        if self.method == "average":
            # 패딩 위치 0으로
            track_feat_seq = track_feat_seq.masked_fill(mask.unsqueeze(-1), 0.0)
            # 평균
            feat = track_feat_seq.sum(dim=1) / (length.unsqueeze(-1).clamp_min(1))
        elif self.method == "soft_weight":
            w = self.soft_weights(track_idx_seq)  # [B,N,1]
            wfeat = track_feat_seq * w
            wfeat = wfeat.masked_fill(mask.unsqueeze(-1), 0.0)
            feat = wfeat.sum(dim=1) / (length.unsqueeze(-1).clamp_min(1))
        elif self.method == "self_attn":
            ctx = self.self_attn(track_feat_seq, mask)  # [B, N, D]
            ctx = ctx.masked_fill(mask.unsqueeze(-1), 0.0)
            feat = ctx.sum(dim=1) / (length.unsqueeze(-1).clamp_min(1))
        else:
            # fallback => average
            track_feat_seq = track_feat_seq.masked_fill(mask.unsqueeze(-1), 0.0)
            feat = track_feat_seq.sum(dim=1) / (length.unsqueeze(-1).clamp_min(1))

        return feat


###################################################
# 2) LARPModel: Momentum + Queue + CIP with BERT
###################################################
class LARPModel(nn.Module):
    """
    - BERTCustom을 이용해 텍스트 임베딩
    - Momentum 인코더 + Queue (contrastive)
    - CIP (Playlist Constructor)
    """
    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        n_layers=4,               # 간소화로 4레이어만
        num_heads=8,
        intermediate_size=1024,
        max_position=512,
        device="cuda",
        num_tracks=10000,
        method="average",         # CIP 방식
        queue_size=10000,
        momentum=0.995
    ):
        super().__init__()
        self.device = torch.device(device)
        self.num_tracks = num_tracks
        self.queue_size = queue_size
        self.momentum = momentum

        # --- 1) BERT config & model
        self.config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=n_layers,
            num_attention_heads=num_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position
        )
        self.bert_main = BertModelCustom(self.config)
        self.bert_m    = BertModelCustom(self.config)
        # momentum update를 위해 묶음
        self.model_pairs = [(self.bert_main, self.bert_m)]

        # --- 2) Projection layer (optional)
        #  CLS 벡터를 임베딩으로 쓴다고 가정
        self.proj_main = nn.Linear(hidden_size, hidden_size)
        self.proj_m    = nn.Linear(hidden_size, hidden_size)
        self.model_pairs.append((self.proj_main, self.proj_m))

        # -- 3) Queue
        self.register_buffer("text_queue", torch.randn(hidden_size, queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        nn.init.normal_(self.text_queue, mean=0, std=0.01)

        # -- 4) CIP buffer
        self.register_buffer("text_features_all", torch.zeros(num_tracks, hidden_size))

        # -- 5) CIP constructor
        self.playlist_constructor = Playlist_Constructor(
            num_tracks=num_tracks, method=method, embedding_size=hidden_size
        )

        # momentum 초기 동기화
        for (main_module, m_module) in self.model_pairs:
            for p_m, p_s in zip(m_module.parameters(), main_module.parameters()):
                p_m.data.copy_(p_s.data)
                p_m.requires_grad = False

        # 온도 파라미터
        self.temp = nn.Parameter(torch.tensor(0.07))

    def forward(self, input_ids, token_type_ids, attention_mask,
                track_idxs, track_masks, track_lengths,
                alpha=0.5, loss_type={"wtc", "tpc"},
                update_momentum=True, update_features=True):
        """
        input_ids: [B, L]
        track_idxs: [B, N]
        track_masks: [B, N], True=padding
        track_lengths: [B]
        """
        # (1) BERT main
        out_main = self.bert_main(input_ids, token_type_ids, attention_mask)
        # out_main: [B, L, hidden], CLS=out_main[:,0,:]
        cls_main = out_main[:, 0, :]
        cls_main = self.proj_main(cls_main)  # [B, hidden]
        cls_main = F.normalize(cls_main, dim=-1)

        # CIP buffer 업데이트
        if update_features:
            # track_idxs: [B], ex: CIP buffer => text_features_all
            self.text_features_all[track_idxs[:, 0]] = cls_main.detach()
            # (예시) 단, 실제론 track_idxs shape가 [B,N]인 경우
            #       대표값만 사용하거나, 세분화된 로직이 필요.

        # (2) BERT momentum
        if update_momentum:
            self._momentum_update()

        with torch.no_grad():
            out_m = self.bert_m(input_ids, token_type_ids, attention_mask)
            cls_m = out_m[:, 0, :]
            cls_m = self.proj_m(cls_m)
            cls_m = F.normalize(cls_m, dim=-1)

            # queue와 합침
            all_m = torch.cat([cls_m.t(), self.text_queue.clone().detach()], dim=1)
            # sim
            sim_m = torch.einsum("bd,dk->bk", cls_m, all_m) / self.temp
            sim_targets = torch.zeros_like(sim_m)
            sim_targets.scatter_(1, torch.arange(cls_m.size(0), device=self.device).unsqueeze(1), 1.0)

            # alpha로 softmax vs one-hot
            sim_m_tg = alpha * F.softmax(sim_m, dim=1) + (1-alpha)*sim_targets

        sim_main = torch.einsum("bd,dk->bk", cls_main, all_m) / self.temp

        losses = {}
        total_loss = torch.tensor(0.0, device=self.device)

        # (3-1) wtc
        if "wtc" in loss_type:
            wtc_loss = -torch.sum(F.log_softmax(sim_main, dim=1) * sim_m_tg, dim=1).mean()
            losses["wtc"] = wtc_loss
            total_loss += wtc_loss

        # update queue
        if update_momentum:
            self._dequeue_and_enqueue(cls_m)

        # (3-2) tpc (CIP)
        if "tpc" in loss_type:
            # playlist_constructor
            # track_feat_seq shape => [B, N, hidden]
            # text_features_all[track_idxs] => [B,N, hidden]
            #  (단, track_idxs shape가 [B,N] 필요)
            track_feats = self.text_features_all[track_idxs]  # [B,N, hidden]
            track_feats = track_feats.to(cls_main.device)
            # CIP
            CIP_feat = self.playlist_constructor(track_idxs, track_feats, track_masks, track_lengths)
            sim_cip = torch.einsum("bd,bd->b", cls_main, CIP_feat) / self.temp
            # 대각선 1
            diag_targets = torch.ones_like(sim_cip)
            tpc_loss = -torch.mean(F.logsigmoid(sim_cip))  # 간단 처리
            losses["tpc"] = tpc_loss
            total_loss += tpc_loss

        losses["loss"] = total_loss
        return losses

    @torch.no_grad()
    def _momentum_update(self):
        # EMA
        for (main_module, m_module) in self.model_pairs:
            for p_s, p_m in zip(main_module.parameters(), m_module.parameters()):
                p_m.data = p_m.data * self.momentum + p_s.data * (1.0 - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, cls_m):
        # cls_m: [B, hidden]
        # queue shape [hidden, queue_size]
        B, D = cls_m.shape
        ptr = int(self.queue_ptr.item())
        assert self.queue_size % B == 0, "queue_size % B != 0"

        end_ptr = ptr + B
        if end_ptr > self.queue_size:
            remain = self.queue_size - ptr
            self.text_queue[:, ptr:] = cls_m[:remain].T
            ptr = 0
            cls_m = cls_m[remain:]
            end_ptr = B - remain

        self.text_queue[:, ptr:end_ptr] = cls_m.T
        ptr = end_ptr % self.queue_size
        self.queue_ptr[0] = ptr


###################################################
# 3) Trainer (LARP) + Dataset
###################################################
class TextOnlyDataset(Dataset):
    """
    (데모) 단순히 랜덤 input_ids를 생성하고, 임의 track index를 만들어 CIP 테스트
    """
    def __init__(self, size=1000, seq_len=16, vocab_size=30522, max_tracks=10000):
        super().__init__()
        self.size = size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.max_tracks = max_tracks

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        # 임의 input_ids, token_type_ids, attention_mask
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        token_type_ids = torch.zeros(self.seq_len, dtype=torch.long)
        attention_mask = torch.ones(self.seq_len, dtype=torch.float)

        # CIP용 track_idxs, track_masks, track_lengths
        # 예: [B,N], N=5
        N = 5
        track_idxs = torch.randint(0, self.max_tracks, (N,))
        track_masks = torch.zeros(N, dtype=torch.bool)  # False=유효
        track_lengths = torch.tensor([N])  # N개 모두 유효

        return input_ids, token_type_ids, attention_mask, track_idxs, track_masks, track_lengths


class LARP(nn.Module):
    """
    간단한 Trainer 예시
    """
    def __init__(self, data_manager, n_sample: 1000, model: LARPModel, lr=1e-3, wd=1e-5, device="cuda"):
        super().__init__()
        self.data_manager = data_manager
        self.n_sample = n_sample,
        self.model = model.to(device)
        self.device = torch.device(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        self.loss_history = []

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        for batch in loader:
            # batch => input_ids, token_type_ids, attn_mask, track_idxs, track_masks, track_lengths
            # shape 정리
            for i in range(len(batch)):
                batch[i] = batch[i].to(self.device)

            input_ids, token_type_ids, attn_mask, track_idxs, track_masks, track_lengths = batch
            B = input_ids.size(0)

            self.optimizer.zero_grad()
            out = self.model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attn_mask,
                track_idxs=track_idxs.unsqueeze(1),  # [B,1,N]? or [B,N]? 예시로 수정 필요
                track_masks=track_masks.unsqueeze(1), 
                track_lengths=track_lengths,
                alpha=0.5,
                loss_type={"wtc", "tpc"},
                update_momentum=True,
                update_features=True
            )
            loss = out["loss"]
            loss.backward()
            clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(loader)

    @torch.no_grad()
    def eval_epoch(self, loader):
        self.model.eval()
        total_loss = 0.0
        for batch in loader:
            for i in range(len(batch)):
                batch[i] = batch[i].to(self.device)
            input_ids, token_type_ids, attn_mask, track_idxs, track_masks, track_lengths = batch
            out = self.model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attn_mask,
                track_idxs=track_idxs.unsqueeze(1),
                track_masks=track_masks.unsqueeze(1),
                track_lengths=track_lengths,
                alpha=0.5,
                loss_type={"wtc", "tpc"},
                update_momentum=False,
                update_features=False
            )
            total_loss += out["loss"].item()

        return total_loss / len(loader)

    def run_training(self, train_loader, val_loader, n_epochs=5, patience=3):
        best_val = float("inf")
        wait = 0
        for epoch in range(1, n_epochs+1):
            tr_loss = self.train_epoch(train_loader)
            val_loss = self.eval_epoch(val_loader)
            print(f"Epoch {epoch}/{n_epochs} | train_loss={tr_loss:.4f}, val_loss={val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print("EarlyStopping triggered.")
                    break


###################################################
# 실행 예시 (main)
###################################################
if __name__ == "__main__":
    # 1) 모델 생성 (임의 BERT 4 레이어, hidden=256 등 조정)
    model = LARPModel(
        vocab_size=30522,
        hidden_size=256,  # 줄여서 데모
        n_layers=4,
        num_heads=4,
        intermediate_size=512,
        max_position=128,
        device="cpu",
        num_tracks=5000,
        method="average",
        queue_size=2000,
        momentum=0.99
    )
    trainer = LARP(model, lr=1e-3, wd=1e-5, device="cpu")

    # 2) Dataset/DataLoader
    train_data = TextOnlyDataset(size=200, seq_len=16, vocab_size=30522, max_tracks=5000)
    val_data   = TextOnlyDataset(size=50,  seq_len=16, vocab_size=30522, max_tracks=5000)
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=8, shuffle=False)

    # 3) 학습
    trainer.run_training(train_loader, val_loader, n_epochs=3, patience=2)

    print("Done.")
