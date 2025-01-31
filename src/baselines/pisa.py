import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
import time
from torch.utils.data import DataLoader, TensorDataset

class PISA:
   def __init__(self, n_sample=2252463, k=100, sample_size=1000, sampling='recent', embed_dim=256, queue_size=57600, momentum=0.995, session_key='SessionId', item_key='ItemId', time_key='Time', device='cuda'):
        self.n_sample = n_sample
        self.k = k
        self.sample_size = sample_size
        self.sampling = sampling

        self.embed_dim = embed_dim
        self.queue_size = queue_size 
        self.momentum = momentum

        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key
        self.device = torch.device(device)

        # Initialize embeddings
        self.sample_embeddings = nn.Embedding(n_sample, embed_dim).to(self.device)
        self.sample_embeddings.weight.data.uniform_(-0.01, 0.01)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.sample_embeddings.parameters(), lr=0.01)

        # Updated during recommendation
        self.session = -1
        self.session_items = []
        self.relevant_sessions = set()
        self.session_item_map = {}
        self.item_session_map = {}
        self.session_time = {}

   def run_training(self, train, tuning=False, savePath=None, epochs=10, val_size=0.1, lr_scheduler_args=None):
    # lr_scheduler 인자가 없으면 기본값 설정
    if lr_scheduler_args is None:
        lr_scheduler_args = {'T_max': epochs, 'eta_min': 0.0}

        # CosineAnnealingLR 스케줄러 초기화
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        self.optimizer,
        T_max=lr_scheduler_args['T_max'],
        eta_min=lr_scheduler_args['eta_min']
    )
    
    start_time = time.time()
    index_session = train.columns.get_loc(self.session_key)
    index_item = train.columns.get_loc(self.item_key)
    index_time = train.columns.get_loc(self.time_key)
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        progress_bar = tqdm.tqdm(train.itertuples(), total=len(train), desc=f"Training epoch {epoch+1}")
        session = -1
        session_items = []
        current_time = -1
        
        for row in progress_bar:
            if row[index_session] != session:
                if len(session_items) > 0:
                    self.session_item_map[session] = session_items
                    self.session_time[session] = current_time
                session = row[index_session]
                session_items = []
            current_time = row[index_time]
            session_items.append(row[index_item])
            
            if row[index_item] not in self.item_session_map:
                self.item_session_map[row[index_item]] = set()
            self.item_session_map[row[index_item]].add(row[index_session])
            
            progress_bar.update(1)
        
        # Validation loss 계산 및 출력
        val_loss = self.calculate_validation_loss(train, val_size=val_size)
        print(f"Epoch {epoch+1}/{epochs} - Validation Loss: {val_loss:.4f}")

        scheduler.step()

        if savePath:
            torch.save(self, savePath)

   def calculate_validation_loss(self, data, val_size=0.1):
        val_indices = np.random.choice(len(data), size=int(len(data) * val_size), replace=False)
        val_data = data.iloc[val_indices]
        
        total_loss = 0
        for _, row in val_data.iterrows():
            session_id = row[self.session_key]
            item_id = row[self.item_key]
            
            # n_sample 파라미터 사용
            scores = self.predict_next(session_id, item_id, np.arange(self.n_sample))
            loss = -np.log(scores[item_id] + 1e-10)
            total_loss += loss
                
        return total_loss / len(val_data)

   def predict_next(self, session_id, input_item_id, predict_for_item_ids, timestamp=0):
       if self.session != session_id:
           self.session = session_id
           self.session_items = []
           self.relevant_sessions = set()

       self.session_items.append(input_item_id)
       neighbors = self.find_neighbors(set(self.session_items), input_item_id, session_id)
       scores = self.score_items(neighbors, self.session_items)

       # Normalize scores
       max_score = max(scores.values()) if scores else 1
       scores = {k: v / max_score for k, v in scores.items()}

       # Convert scores to numpy array
       scores_array = np.array([scores.get(item_id, 0) for item_id in predict_for_item_ids])

       return scores_array

   def find_neighbors(self, session_items, input_item_id, session_id):
       possible_neighbors = self.possible_neighbor_sessions(session_items, input_item_id, session_id)
       neighbors = self.calc_similarity(session_items, possible_neighbors)
       neighbors = sorted(neighbors, reverse=True, key=lambda x: x[1])[:self.k]
       return neighbors

   def possible_neighbor_sessions(self, session_items, input_item_id, session_id):
       self.relevant_sessions |= self.item_session_map.get(input_item_id, set())
       if len(self.relevant_sessions) > self.sample_size:
           if self.sampling == 'recent':
               sample = self.most_recent_sessions(self.relevant_sessions, self.sample_size)
           elif self.sampling == 'random':
               sample = random.sample(self.relevant_sessions, self.sample_size)
           else:
               sample = list(self.relevant_sessions)[:self.sample_size]
           return sample
       else:
           return self.relevant_sessions

   def calc_similarity(self, session_items, sessions):
       neighbors = []
       for session in sessions:
           session_items_test = set(self.session_item_map.get(session, []))
           similarity = self.jaccard(session_items, session_items_test)
           if similarity > 0:
               neighbors.append((session, similarity))
       return neighbors

   def jaccard(self, first, second):
       intersection = len(first & second)
       union = len(first | second)
       return intersection / union if union > 0 else 0

   def score_items(self, neighbors, current_session):
       scores = {}
       for session, similarity in neighbors:
           items = self.session_item_map.get(session, [])
           for item in items:
               if item not in current_session:
                   scores[item] = scores.get(item, 0) + similarity
       return scores

   def most_recent_sessions(self, sessions, number):
       tuples = [(session, self.session_time.get(session, -1)) for session in sessions]
       tuples = sorted(tuples, key=lambda x: x[1], reverse=True)
       return [element[0] for element in tuples[:number]]