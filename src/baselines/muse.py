import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.data_manager.data_manager import SequentialTrainDataset, pad_collate
from torch.utils.data import DataLoader
import time
from src.rta.utils import padded_avg, get_device

class MUSE(nn.Module):
    def __init__(self,data_manager,training_params = {}, k=1500, n_items=2252463, hidden_size=128, lr=0.001, batch_size=64, alpha=0.5, inv_coeff=1.0, var_coeff=0.5, cov_coeff=0.25, n_layers=1, maxlen=50, dropout=0.1, embedding_dim=256, n_sample=10000, step=1):
        super(MUSE, self).__init__()
        self.k = k
        self.n_items = n_items
        self.hidden_size = hidden_size
        self.lr = lr
        self.batch_size = batch_size
        self.alpha = alpha
        self.inv_coeff = inv_coeff
        self.var_coeff = var_coeff
        self.cov_coeff = cov_coeff
        self.n_layers = n_layers
        self.maxlen = maxlen
        self.dropout = dropout
        self.embedding_dim = embedding_dim
        self.n_sample = n_sample
        self.step = step
        self.data_manager = data_manager
        self.training_params = training_params
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.training_params = {
            'lr': lr,                  # Learning rate
            'wd': 1e-4,                # Weight decay (L2 regularization)
            'mom': 0.9,                # Momentum for optimizer
            'nesterov': True,          # Whether to use Nesterov momentum
            'n_epochs': 50,            # Number of epochs
            'clip': 1.0,               # Gradient clipping max norm
            'patience': 5,             # For learning rate scheduler
            'factor': 0.5,             # Learning rate decay factor
            'max_size': 10000,         # Maximum size of the dataset
            'n_neg': 5,                # Number of negative samples
            'batch_size': batch_size   # Batch size
        }

        self.load_model()

    def load_model(self):
        self.model = VICReg(self.n_items, self).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

    def run_training(self, tuning=False, savePath=False):
        # Test 데이터 로딩
        if tuning:
            test_evaluator, test_dataloader = self.data_manager.get_test_data("val")
        else:
            test_evaluator, test_dataloader = self.data_manager.get_test_data("test")
        
        # Optimizer, scheduler 및 train 데이터 준비
        optimizer, scheduler, train_dataloader = self.prepare_training_objects(tuning)
        
        batch_ct = 0
        print_every = False
        if "step_every" in self.training_params.keys():
            print_every = True
        
        start = time.time()
        
        # 모델 저장 경로가 제공되면 모델을 저장
        if savePath:
            torch.save(self, savePath)
        
        # 학습 시작
        for epoch in range(self.training_params['n_epochs']):
            print("Epoch %d/%d" % (epoch + 1, self.training_params['n_epochs']))
            print("Elapsed time : %.0f seconds" % (time.time() - start))
            
            epoch_loss = 0  # 에포크 손실 초기화
            
            for xx_pad, yy_pad_neg, x_lens in tqdm(train_dataloader):
                optimizer.zero_grad()
                
                # 배치 손실 계산
                loss = self.compute_loss_batch(xx_pad.to(get_device()), yy_pad_neg.to(get_device()))
                loss.backward()
                
                if self.training_params['clip']:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.training_params['clip'], norm_type=2)
                
                optimizer.step()
                
                epoch_loss += loss.item()
                
                if print_every:
                    if batch_ct % self.training_params["step_every"] == 0:
                        scheduler.step()
                        print(f"Batch {batch_ct}, Loss: {loss.item():.4f}")
                        
                        # 검증 지표 계산
                        recos = self.compute_recos(test_dataloader)
                        r_prec = test_evaluator.compute_all_R_precisions(recos)
                        ndcg = test_evaluator.compute_all_ndcgs(recos)
                        click = test_evaluator.compute_all_clicks(recos)
                        
                        print(f"rprec: {r_prec.mean():.3f}, ndcg: {ndcg.mean():.3f}, click: {click.mean():.3f}")
                
                batch_ct += 1
            # 에포크 손실 출력
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            print(f"Epoch {epoch + 1} Loss: {avg_epoch_loss:.4f}")
            
            # 모델 저장
            if savePath:
                torch.save(self, savePath)
        
        return
    
    def prepare_training_objects(self, tuning=False):
        # Prepare the optimizer, the scheduler and the data_loader that will be used for training
        optimizer = torch.optim.SGD(self.parameters(), lr= self.training_params['lr'], weight_decay=self.training_params['wd'], momentum=self.training_params['mom'], nesterov=self.training_params['nesterov'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.training_params['patience'], gamma=self.training_params['factor'], last_epoch=- 1, verbose=False)
        if tuning:
            train_indices = self.data_manager.train_indices
        else:
            train_indices = np.concatenate((self.data_manager.train_indices, self.data_manager.val_indices))
        train_dataset = SequentialTrainDataset(self.data_manager, train_indices, max_size=self.training_params['max_size'], n_neg=self.training_params['n_neg'])
        train_dataloader = DataLoader(train_dataset, batch_size = self.training_params['batch_size'], shuffle=True, collate_fn=pad_collate, num_workers=0)
        return optimizer, scheduler, train_dataloader
    
    def compute_pos_loss_batch(self, X_agg, Y_pos_rep, pad_mask):
        # The part of the loss that concerns positive examples
        pos_prod = torch.sum(X_agg * Y_pos_rep, axis = 2).unsqueeze(2)
        pos_loss = padded_avg(-F.logsigmoid(pos_prod), ~pad_mask).mean()
        return pos_loss
    
    def compute_loss_batch(self, x_pos, x_neg):
        # 패딩 마스크 생성 (0인 위치가 패딩)
        pad_mask = (x_pos == 0).to(self.device)
        
        # 입력과 타겟 시퀀스 분리
        input_rep = x_pos[:, :-1]  # 마지막 요소 제외
        Y_pos_rep = x_pos[:, 1:]   # 첫 요소 제외
        
        # Positive 손실 계산
        pos_loss = self.calculate_loss(input_rep, Y_pos_rep)
        
        # Negative 손실 계산 (패딩 마스크 조정)
        neg_loss = self.calculate_neg_loss(input_rep, x_neg, pad_mask[:, :-1])
        
        return pos_loss + neg_loss

    def calculate_session_similarity(self, current_session, neighbor_sessions, method='cosine'):
        similarities = {}
        for session_id, session_items in neighbor_sessions.items():
            if method == 'cosine':
                similarity = F.cosine_similarity(current_session, session_items, dim=0)
            elif method == 'jaccard':
                intersection = len(set(current_session) & set(session_items))
                union = len(set(current_session) | set(session_items))
                similarity = intersection / union
            similarities[session_id] = similarity
        return similarities
    
    def time_decay(self, current_time, neighbor_time, lambda_time=86400):  # 하루(초 단위) 기준
        time_difference = current_time - neighbor_time
        decay = torch.exp(-time_difference / lambda_time)
        return decay

    def hard_negative_sampling(self, positive_embeddings, negative_candidates, k=10):
        similarities = torch.matmul(positive_embeddings, negative_candidates.T)
        hard_negatives = torch.topk(similarities, k=k, largest=True).indices
        return hard_negatives

    def global_loss(self, embeddings):
        inv_loss = 0.0
        for i in range(len(embeddings)):
            for j in range(len(embeddings)):
                if i != j:
                    inv_loss += F.mse_loss(embeddings[i], embeddings[j])
        inv_loss *= self.inv_coeff

        var_loss = 0.0
        for emb in embeddings:
            std_emb = torch.sqrt(emb.var(dim=0) + 1e-6)
            var_loss += torch.mean(F.relu(1.0 - std_emb))
        var_loss *= self.var_coeff

        cov_loss = 0.0
        for emb in embeddings:
            cov_matrix = torch.mm(emb.T, emb) / (emb.size(0) - 1)
            cov_loss += torch.sum((cov_matrix - torch.eye(cov_matrix.size(0)).to(self.device)).pow(2))
        cov_loss *= self.cov_coeff

        return inv_loss, var_loss, cov_loss

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        batch_losses = []
        shuffle_rec_losses = []
        nonshuffle_rec_losses = []
        epoch_loss = 0
        train_batch_iter = tqdm(enumerate(dataloader), total=len(dataloader))

        for i, batch in train_batch_iter:
            batch['aug1'] = batch['aug1'].to(self.device, non_blocking=True)

            v1_hidden, v1_preds = self.model(batch,
                                             input_str='orig_sess',
                                             len_str='lens',
                                             get_last=True)
            v2_hidden, v2_preds = self.model(batch,
                                             input_str='aug1',
                                             len_str='aug_len1',
                                             get_last=True)
            
            matching_loss = self.model.compute_finegrained_matching_loss(
                batch, v1_hidden, v2_hidden, v1_preds, v2_preds, epoch
            )

            rec_loss = self.calculate_loss(v1_preds, batch)
            global_inv_loss, global_var_loss, global_cov_loss = self.global_loss([v1_preds, v2_preds])
            loss = (self.alpha * rec_loss.mean() + (1 - self.alpha) * matching_loss +
                    global_inv_loss + global_var_loss + global_cov_loss)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Gradient clipping 적용

            self.optimizer.step()

            batch_losses.append(loss.item())
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(dataloader)
        return avg_epoch_loss
    
    def calculate_loss(self, input_rep, Y_pos_rep):
        # Ensure input_rep and Y_pos_rep are 2D (Batch, Features)
        if len(input_rep.shape) == 3:
            input_rep = input_rep.view(-1, input_rep.size(-1))
            Y_pos_rep = Y_pos_rep.view(-1, Y_pos_rep.size(-1))

        input_rep = input_rep.float()
        Y_pos_rep = Y_pos_rep.float()

        # Calculate logits
        logits = torch.matmul(input_rep, Y_pos_rep.transpose(0, 1))

        # Create target indices for each row (diagonal matrix indices)
        target = torch.arange(logits.size(0), device=logits.device)

        # Calculate loss only for non-padded elements
        pad_mask = (input_rep.sum(dim=1) != 0)
        if pad_mask.any():
            loss = self.loss_func(logits[pad_mask], target[pad_mask])
        else:
            loss = torch.tensor(0.0, device=logits.device)

        return loss
    
    def calculate_neg_loss(self, input_rep, x_neg, pad_mask):
        n_items = self.model.backbone.item_embedding.weight.size(0)
        batch_size = input_rep.size(0)
        chunk_size = 32
        total_loss = 0
        total_valid = 0
        
        for i in range((batch_size + chunk_size - 1) // chunk_size):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, batch_size)
            
            chunk_input = torch.clamp(input_rep[start_idx:end_idx], 0, n_items-1)
            chunk_neg = torch.clamp(x_neg[start_idx:end_idx], 0, n_items-1)
            chunk_mask = pad_mask[start_idx:end_idx]
            
            input_emb = self.model.backbone.item_embedding(chunk_input)
            neg_emb = self.model.backbone.item_embedding(chunk_neg)
            
            logits = torch.matmul(input_emb, neg_emb.transpose(-2, -1))
            logits = logits.reshape(-1, logits.size(-1))
            
            chunk_mask = chunk_mask.reshape(-1)
            target = torch.zeros(logits.size(0), dtype=torch.long, device=self.device)
            
            valid_mask = ~chunk_mask
            if valid_mask.any():
                chunk_loss = self.loss_func(logits[valid_mask], target[valid_mask])
                total_loss += chunk_loss
                total_valid += 1
            
            del input_emb, neg_emb, logits
            torch.cuda.empty_cache()
        
        return total_loss / max(total_valid, 1)
            

    def predict(self, predictions):
        all_embs = self.model.backbone.item_embedding.weight
        logits = torch.matmul(predictions, all_embs.transpose(0, 1))
        logits = F.softmax(logits, dim=1)

        return logits
    
    def predict_next(self, session_id, input_item_id, predict_for_item_ids):
        self.model.eval()

        # 입력 배치 준비
        input_batch = self.prepare_input(session_id, input_item_id, predict_for_item_ids)

        # 모델을 통한 예측
        _, predictions = self.model(input_batch)

        # logits 계산 후 소프트맥스 확률 반환
        logits = self.predict(predictions)
        return logits.detach().cpu().numpy()
    
    def prepare_input(self, session_id, input_item_id, predict_for_item_ids):
        # 입력 데이터를 준비하는 메서드
        input_batch = {
            'orig_sess': torch.tensor([[input_item_id]]).to(self.device),  # 세션 내 아이템
            'lens': torch.tensor([1]).to(self.device),  # 시퀀스 길이
            'labels': torch.tensor(predict_for_item_ids).to(self.device)  # 예측할 아이템들
        }
        return input_batch
    
    def fit(self, train_dataloader, epochs=50):
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_idx, batch in enumerate(tqdm(train_dataloader)):
                self.optimizer.zero_grad()

                batch['orig_sess'] = batch['orig_sess'].to(self.device, non_blocking=True)
                batch['aug1'] = batch['aug1'].to(self.device, non_blocking=True)
                batch['labels'] = batch['labels'].to(self.device, non_blocking=True)

                v1_hidden, v1_preds = self.model(batch, input_str='orig_sess', len_str='lens', get_last=True)
                v2_hidden, v2_preds = self.model(batch, input_str='aug1', len_str='aug_len1', get_last=True)

                matching_loss = self.model.compute_finegrained_matching_loss(
                    batch, v1_hidden, v2_hidden, v1_preds, v2_preds, epoch
                )

                rec_loss = self.calculate_loss(v1_preds, batch)
                global_inv_loss, global_var_loss, global_cov_loss = self.global_loss([v1_preds, v2_preds])

                loss = (self.alpha * rec_loss.mean() + (1 - self.alpha) * matching_loss +
                        global_inv_loss + global_var_loss + global_cov_loss)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_dataloader)}")

class VICReg(nn.Module):
    def __init__(self, input_size, args):
        super().__init__()
        self.n_items = input_size
        self.args = args
        self.device = args.device

        self.num_features = args.hidden_size
        self.backbone = SRGNN(input_size, args)

        self.mask_default = self.mask_correlated_samples(batch_size=args.batch_size)

    def forward(self, batch, input_str='orig_sess', len_str='lens', get_last=True):
        if isinstance(batch, dict):
            hidden, preds = self.backbone(batch, input_str, len_str, get_last)
        else:
            hidden, preds = self.backbone({'orig_sess': batch, 'lens': torch.tensor([batch.size(1)]).to(self.device)}, input_str, len_str, get_last)
        return hidden, preds

    def compute_finegrained_matching_loss(self, batch,
                                          seq_hidden1, seq_hidden2,
                                          seq_pred1, seg_pred2, epoch):
        loss = 0.0
        mask1 = batch['orig_sess'].gt(0)
        mask2 = batch['aug1'].gt(0)
        mask = torch.cat([mask1.unsqueeze(0), mask2.unsqueeze(0)], dim=0)

        seq_hidden = torch.cat([seq_hidden1.unsqueeze(0),
                                seq_hidden2.unsqueeze(0)], dim=0)
        seq_pred = torch.cat([seq_pred1.unsqueeze(0), seg_pred2.unsqueeze(0)], dim=0)

        v1_position = torch.arange(self.args.maxlen).unsqueeze(0).repeat(
            batch['labels'].size(0), 1).to(self.device)
        v2_position = batch['labels'].masked_fill(
            batch['labels'] < 0, 0).to(self.device)
        locations = torch.cat([v1_position.unsqueeze(0),
                               v2_position.unsqueeze(0)], dim=0)

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
            cov_loss = cov_loss + self.off_diagonal(cov_x).pow_(2).sum().div(
                self.args.hidden_size
            )
            iter_ = iter_ + 1
        var_loss = self.args.var_coeff * var_loss / iter_
        cov_loss = self.args.cov_coeff * cov_loss / iter_

        return inv_loss, var_loss, cov_loss
    
    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

class SRGNN(nn.Module):
    def __init__(self, input_size, args):
        super(SRGNN, self).__init__()
        self.n_items = input_size
        self.args = args
        self.device = args.device

        # Embedding
        self.item_embedding = nn.Embedding(self.n_items, args.hidden_size, padding_idx=0)

        # Model Architecture
        self.gnn = GNN(args.hidden_size, step=args.n_layers)
        self.linear1 = nn.Linear(args.hidden_size, args.hidden_size, bias=True)
        self.linear2 = nn.Linear(args.hidden_size, args.hidden_size, bias=True)
        self.linear3 = nn.Linear(args.hidden_size, args.hidden_size, bias=True)
        self.linear_transform = nn.Linear(args.hidden_size * 2, args.hidden_size, bias=True)

        self._init_weights()
    
    def _init_weights(self):
        stdv = 1.0 / np.sqrt(self.args.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, batch, input_str='orig_sess', len_str='lens', get_last=True):
        seqs = batch[input_str]
        lengths_t = torch.as_tensor(batch[len_str]).to(self.device)
        alias_inputs, A, items, mask = self._get_slice(seqs)

        hidden = self.item_embedding(items)
        hidden = self.gnn(A, hidden)
        if alias_inputs.numel() > 0:
            alias_inputs = alias_inputs.view(-1, alias_inputs.size(1), 1).expand(
                -1, -1, self.args.hidden_size
            )
        else:
            alias_inputs = torch.zeros(1, 1, self.args.hidden_size).to(self.device)
        alias_inputs = alias_inputs.long()  # Ensure alias_inputs is of dtype int64
        seq_hidden = torch.gather(hidden, dim=1, index=alias_inputs)
        # fetch the last hidden state of last timestamp
        ht = self.get_last_item(seq_hidden, lengths_t)
        q1 = self.linear1(ht).view(ht.size(0), 1, ht.size(1))
        q2 = self.linear2(seq_hidden)

        alp = self.linear3(torch.sigmoid(q1 + q2))
        a = torch.sum(alp * seq_hidden * mask.view(mask.size(0), -1, 1).float(), 1)
        seq_output = self.linear_transform(torch.cat([a, ht], dim=1))
        
        return seq_hidden, seq_output

    def _get_slice(self, seqs):
        mask = seqs.gt(0)
        items, A, alias_inputs = [], [], []
        max_n_nodes = seqs.size(1)
        seqs = seqs.cpu().numpy()
        for seq in seqs:
            node = np.unique(seq)

            # 유효하지 않은 인덱스를 임베딩 범위 내로 조정
            node = np.clip(node, 0, self.n_items - 1)

            items.append(node.tolist() + (max_n_nodes - len(node)) * [0])
            u_A = np.zeros((max_n_nodes, max_n_nodes))
            for i in np.arange(len(seq) - 1):
                if seq[i+1] == 0:
                    break
                u = np.where(node == seq[i])[0][0]
                v = np.where(node == seq[i+1])[0][0]
                u_A[u][v] = 1
            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A.append(u_A)
            alias_inputs.append([np.where(node == i)[0][0] for i in seq if i in node])
        alias_inputs = torch.LongTensor(alias_inputs).to(self.device)
        A = torch.FloatTensor(np.array(A)).to(self.device)
        items = torch.LongTensor(items).to(self.device)

        return alias_inputs, A, items, mask

    def get_last_item(self, seq_hidden, lengths_t):
        idx = (lengths_t - 1).view(-1, 1).expand(len(lengths_t), seq_hidden.size(2)).unsqueeze(1)
        return seq_hidden.gather(1, idx).squeeze(1)

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

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden