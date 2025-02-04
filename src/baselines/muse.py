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
# MUSE 클래스 (2D, 최종 시점만 사용, pos/neg)
#########################################
class MUSE(nn.Module):
    def __init__(self, data_manager,
                 k=1500, n_items=2262292, hidden_size=128, lr=0.01, batch_size=64,
                 alpha=0.5, inv_coeff=1.0, var_coeff=0.5, cov_coeff=0.25,
                 n_layers=2, maxlen=10, dropout=0.1, embedding_dim=256,
                 n_sample=10000, step=1, training_params=None):
        super().__init__()
        self.data_manager = data_manager
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.batch_size = batch_size
        self.n_sample = n_sample

        # config
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
        self.alpha = alpha
        self.inv_coeff = inv_coeff
        self.var_coeff = var_coeff
        self.cov_coeff = cov_coeff

        if training_params is None:
            self.training_params = {
                'lr': 0.001,
                'wd': 1e-4,
                'n_epochs': 30,
                'clip': 1.0,
                'early_patience': 3,
                'factor': 0.5,
                'max_size': 10,
                'batch_size': batch_size,
                'sample_size': None,
                # COSINE LR args
                'use_cosine': True,
                'cosine_tmax': 10,
                'cosine_emin': 1e-5
            }
        else:
            self.training_params = training_params

        # VICReg 백본 (SRGNN)
        self.model = VICReg(self.n_items, self.config).to(self.device)
        # 추가: aggregator hidden dimension vs item embedding dimension 체크
        aggregator_hidden_dim = self.model.backbone.hidden_size
        item_emb_dim = self.model.backbone.item_embedding.embedding_dim
        assert aggregator_hidden_dim == item_emb_dim, (
            f"[ERROR] aggregator hidden_size ({aggregator_hidden_dim}) != "
            f"item embedding_dim ({item_emb_dim})"
        )
        print(f"[INFO] aggregator hidden_size={aggregator_hidden_dim}, item_embedding_dim={item_emb_dim}")

    def run_training(self, train=None, tuning=False, savePath=None, sample_size=None):
        if sample_size is not None:
            self.training_params["sample_size"] = sample_size

        if tuning:
            train_indices = self.data_manager.train_indices
            val_indices   = self.data_manager.val_indices
        else:
            train_indices = np.concatenate([self.data_manager.train_indices, self.data_manager.val_indices])
            val_indices   = self.data_manager.val_indices

        train_dataloader = self.make_dataloader(train_indices, sample_size=self.training_params["sample_size"])
        val_dataloader   = self.make_dataloader(val_indices,   sample_size=None)

        optimizer, scheduler = self.prepare_optimizer()

        best_val_loss = float("inf")
        no_improve_count = 0
        best_epoch = 0

        n_epochs = self.training_params["n_epochs"]
        start_time = time.time()

        for epoch in range(1, n_epochs+1):
            print(f"=== Epoch {epoch}/{n_epochs} ===")
            print(f"Elapsed time: {time.time()-start_time:.0f} seconds")

            train_loss = self.train_epoch(train_dataloader, epoch, optimizer)
            val_loss   = self.val_epoch(val_dataloader)

            print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            if self.training_params.get("use_cosine", False):
                scheduler.step()

            if val_loss<best_val_loss:
                best_val_loss = val_loss
                no_improve_count = 0
                best_epoch = epoch
                if savePath:
                    torch.save(self.state_dict(), f"{savePath}_best.pth")
                    print(f"[EARLY STOP] Best model updated at epoch {epoch}, val_loss={val_loss:.4f}")
            else:
                no_improve_count += 1
                print(f"[EARLY STOP] No improvement count: {no_improve_count}/{self.training_params['early_patience']}")
                if no_improve_count>=self.training_params['early_patience']:
                    print(f"[EARLY STOP] Stop training at epoch {epoch}. Best epoch was {best_epoch}")
                    break

        if savePath:
            self.load_state_dict(torch.load(f"{savePath}_best.pth"))
            print(f"Training finished. Best model from epoch {best_epoch} loaded.")
        else:
            print(f"Training finished. Best epoch={best_epoch}")

    def make_dataloader(self, indices, sample_size=None):
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
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
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

        for batch_idx, (xx_pad, yy_pad_neg, x_lens) in enumerate(tqdm(dataloader, desc=f"[Train Epoch {epoch}]")):
            optimizer.zero_grad()

            xx_pad = xx_pad.to(self.device)        # [B, seq_len]
            yy_pad_neg = yy_pad_neg.to(self.device)# [B, n_neg]
            x_lens = x_lens.to(self.device)

            loss = self.compute_loss_batch(xx_pad, yy_pad_neg)
            loss.backward()

            if self.training_params["clip"]>0:
                nn.utils.clip_grad_norm_(self.parameters(), self.training_params["clip"])
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(dataloader)
        return epoch_loss

    def val_epoch(self, dataloader):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xx_pad, yy_pad_neg, x_lens in dataloader:
                xx_pad = xx_pad.to(self.device)
                yy_pad_neg = yy_pad_neg.to(self.device)
                x_lens = x_lens.to(self.device)

                loss = self.compute_loss_batch(xx_pad, yy_pad_neg)
                val_loss += loss.item()
        val_loss /= len(dataloader)
        return val_loss

    def compute_loss_batch(self, x_pos, x_neg):
        pad_mask = (x_pos==0).to(self.device)
        lengths  = torch.sum(~pad_mask, dim=1)  # [B], 이 값은 항상 >=2

        # aggregator forward
        hidden, preds = self.model({
            "orig_sess": x_pos,
            "lens": lengths
        }, get_last=True)  # preds -> [B, hidden]

        # pos_id
        # => sum(~pad_mask, dim=1)-1 >= 1
        pos_id = x_pos[range(x_pos.size(0)), lengths-1] 

        B, d = preds.shape

        # neg_emb => [B, n_neg, d]
        item_emb = self.model.backbone.item_embedding
        neg_emb = item_emb(x_neg)
        BN, n_neg, d_emb = neg_emb.shape
        if BN != B or d_emb != d:
            raise RuntimeError("neg_emb shape mismatch with aggregator output")

        # pos_emb => [B, d]
        pos_emb = item_emb(pos_id)

        # Positive dot
        pos_dot  = torch.sum(preds * pos_emb, dim=1, keepdim=True)
        pos_loss = -F.logsigmoid(pos_dot).mean()

        # Negative dot
        neg_dot  = torch.einsum("bd,bnd->bn", preds, neg_emb)
        neg_loss = -F.logsigmoid(-neg_dot).mean()

        total_loss = pos_loss + neg_loss
        return total_loss

    def predict_next(self, session_id, input_item_id, predict_for_item_ids):
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor([[input_item_id + 1]], device=self.device)
            lens = torch.tensor([1], device=self.device)
            hidden, preds = self.model({"orig_sess": x, "lens": lens}, get_last=True)
            # item_emb: [n_items+2, hidden] (n_items+2로 변경!)
            item_emb = self.model.backbone.item_embedding.weight
            logits   = torch.matmul(preds, item_emb.T).squeeze(0)  # [n_items+2]
            scores   = torch.softmax(logits, dim=-1).cpu().numpy()

        final_scores = np.zeros(len(predict_for_item_ids), dtype=np.float32)
        for i, gid in enumerate(predict_for_item_ids):
            # gid가 0 이상 n_items 미만이면, +1 한 위치의 score를 취함
            if 0 <= gid < self.n_items:
                final_scores[i] = scores[gid+1]
        return final_scores


########################################
# GNN / SRGNN / VICReg
########################################
class GNN(nn.Module):
    def __init__(self, embedding_dim, step=1):
        super().__init__()
        self.step = step
        self.hidden_size = embedding_dim
        self.input_size  = embedding_dim*2
        self.gate_size   = 3*embedding_dim

        self.w_ih = nn.Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = nn.Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = nn.Parameter(torch.Tensor(self.gate_size))
        self.b_hh = nn.Parameter(torch.Tensor(self.gate_size))
        self.b_iah= nn.Parameter(torch.Tensor(self.hidden_size))
        self.b_oah= nn.Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in  = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f   = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def forward(self, A, hidden):
        for _ in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden

    def GNNCell(self, A, hidden):
        B, n_nodes, _ = hidden.shape
        in_count  = A[:, :, :n_nodes]
        out_count = A[:, :, n_nodes:2*n_nodes]
        input_in  = torch.matmul(in_count, self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(out_count, self.linear_edge_out(hidden)) + self.b_oah
        inputs    = torch.cat([input_in, input_out], dim=2)

        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, dim=2)
        h_r, h_i, h_n = gh.chunk(3, dim=2)

        resetgate = torch.sigmoid(i_r+h_r)
        inputgate = torch.sigmoid(i_i+h_i)
        newgate   = torch.tanh(i_n + resetgate*h_n)
        hy = newgate + inputgate*(hidden-newgate)
        return hy


class SRGNN(nn.Module):
    def __init__(self, input_size, args):
        super().__init__()
        self.n_items = input_size
        self.args = args
        self.device = args["device"]
        self.hidden_size = args["hidden_size"]
        self.n_layers = args["n_layers"]

        self.item_embedding = nn.Embedding(self.n_items + 2, self.hidden_size, padding_idx=0)
        self.gnn = GNN(self.hidden_size, step=self.n_layers)
        self.linear1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear3 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)

        self._init_weights()

    def _init_weights(self):
        stdv = 1.0/np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-stdv,stdv)

    def forward(self, batch, input_str='orig_sess', len_str='lens', get_last=True):
        seqs = batch[input_str] # [B, seq_len]
        lengths_t = batch[len_str] # [B]

        alias_inputs, A, items, mask = self._get_slice(seqs)
        hidden = self.gnn(A, self.item_embedding(items))   # [B,n_nodes,hidden]

        if alias_inputs.numel()>0:
            alias_inputs = alias_inputs.view(alias_inputs.size(0),-1,1).expand(-1,-1,self.hidden_size)
        else:
            alias_inputs = torch.zeros(1,1,self.hidden_size, device=self.device)

        alias_inputs = alias_inputs.long()
        seq_hidden   = torch.gather(hidden,1, alias_inputs) # [B, seq_len, hidden]

        if get_last:
            # => [B, hidden]
            ht = self.get_last_item(seq_hidden, lengths_t)
        else:
            # => [B, seq_len, hidden]
            ht = seq_hidden

        q1 = self.linear1(ht).unsqueeze(1) # shape [B,1,hidden]
        q2 = self.linear2(seq_hidden)       # [B,seq_len,hidden]
        alp= self.linear3(torch.sigmoid(q1+q2))

        a = torch.sum(alp * seq_hidden * mask.unsqueeze(-1).float(), dim=1)
        seq_output= self.linear_transform(torch.cat([a, ht], dim=1)) # if get_last: shape [B,hidden]

        return seq_hidden, seq_output

    def _get_slice(self, seqs):
        mask = seqs.gt(0)
        max_n_nodes= seqs.size(1)
        seqs_np= seqs.cpu().numpy()

        alias_inputs= []
        A_list= []
        items_list= []
        max_alias_len=0

        for seq in seqs_np:
            valid_idx= np.where(seq>0)[0]
            if len(valid_idx)==0:
                items_pad= [0]*max_n_nodes
                u_A_= np.zeros((max_n_nodes,2*max_n_nodes),dtype=np.float32)
                A_list.append(u_A_)
                items_list.append(items_pad)
                alias_inputs.append([0])
                continue

            node= np.unique(seq[valid_idx])
            node= np.clip(node,0,self.n_items + 1)
            u_A= np.zeros((max_n_nodes,max_n_nodes),dtype=np.float32)

            for i in range(len(valid_idx)-1):
                cur_item= seq[valid_idx[i]]
                nxt_item= seq[valid_idx[i+1]]
                u_candidates= np.where(node==cur_item)[0]
                v_candidates= np.where(node==nxt_item)[0]
                if len(u_candidates)==0 or len(v_candidates)==0:
                    continue
                u_= u_candidates[0]
                v_= v_candidates[0]
                u_A[u_][v_]=1

            u_sum_in= np.sum(u_A,axis=0); u_sum_in[u_sum_in==0]=1
            u_A_in= u_A/u_sum_in
            u_sum_out= np.sum(u_A,axis=1); u_sum_out[u_sum_out==0]=1
            u_A_out= (u_A.T/u_sum_out).T

            u_A_= np.concatenate([u_A_in,u_A_out],axis=0).T
            items_pad= list(node)+[0]*(max_n_nodes-len(node))

            alias= []
            for idx_ in valid_idx:
                item_ = seq[idx_]
                cand= np.where(node==item_)[0]
                alias.append(cand[0] if len(cand)>0 else 0)

            if len(alias)>max_alias_len:
                max_alias_len= len(alias)

            A_list.append(u_A_)
            items_list.append(items_pad)
            alias_inputs.append(alias)

        for i in range(len(alias_inputs)):
            if len(alias_inputs[i])<max_alias_len:
                alias_inputs[i]+=[0]*(max_alias_len-len(alias_inputs[i]))

        alias_inputs_t= torch.LongTensor(alias_inputs).to(self.device)
        A_t= torch.FloatTensor(np.array(A_list)).to(self.device)
        items_t= torch.LongTensor(np.array(items_list)).to(self.device)
        mask= mask.to(self.device)
        return alias_inputs_t,A_t,items_t,mask

    def get_last_item(self, seq_hidden, lengths_t):
        # seq_hidden: [B, seq_len, hidden]
        idx= (lengths_t-1).view(-1,1,1).expand(-1,1,seq_hidden.size(2))
        return seq_hidden.gather(1, idx).squeeze(1)


class VICReg(nn.Module):
    def __init__(self, input_size, config):
        super().__init__()
        self.n_items= input_size
        self.config= config
        self.device= config["device"]
        self.hidden_size= config["hidden_size"]

        self.backbone= SRGNN(input_size, config)

    def forward(self, batch, input_str='orig_sess', len_str='lens', get_last=True):
        hidden, preds= self.backbone(batch, input_str, len_str, get_last)
        return hidden, preds

    def global_loss(self, embeddings):
        inv_loss=0.0
        count=0
        for i in range(len(embeddings)):
            for j in range(i+1,len(embeddings)):
                inv_loss+=F.mse_loss(embeddings[i], embeddings[j])
                count+=1
        if count>0:
            inv_loss/=count

        var_loss=0.0
        cov_loss=0.0
        return inv_loss*self.config.get("inv_coeff",1.0), var_loss, cov_loss
