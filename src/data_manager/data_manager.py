import numpy as np
import os
import json
import operator
import pickle
import math
import scipy.sparse
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import random

from src.evaluator import Evaluator

class DataManager():
    N_SEED_SONGS = range(1,11) 

    def __init__(
        self,
        foldername = "resources/data/",
        test_size=200000,
        min_songs_test=10,
        resplit=False,
        dim=128
    ):
        self.foldername = foldername 
        self.test_size = test_size
        self.min_songs_test = min_songs_test

        self.load_playlist_track()
        self.song_embeddings_path = '%s/embeddings/song_embeddings_%d.npy' % (self.foldername, dim)
        self.album_embeddings_path = '%s/embeddings/alb_embeddings_%d.npy' % (self.foldername, dim)
        self.artist_embeddings_path = '%s/embeddings/art_embeddings_%d.npy' % (self.foldername, dim)
        self.pop_embeddings_path = '%s/embeddings/pop_embeddings_%d.npy' % (self.foldername, dim)
        self.dur_embeddings_path = '%s/embeddings/dur_embeddings_%d.npy' % (self.foldername, dim)

        self.load_track_info()
        self.load_metadata()

        # MPD 사이즈
        self.n_playlists = 10**6
        self.n_tracks = 2262292

        self.train_indices = self.get_indices("train", resplit=resplit)
        self.val_indices   = self.get_indices("val")
        self.test_indices  = self.get_indices("test")

        self.ground_truths = {}
        self.ground_truths_first = {}
        self.seed_tracks = {}

        tmp = [self.get_ground_truth("val", n_start_songs=i) for i in DataManager.N_SEED_SONGS]
        self.seed_tracks["val"] = {i: tmp[ind][0] for ind,i in enumerate(DataManager.N_SEED_SONGS)}
        self.ground_truths["val"] = {i: tmp[ind][1] for ind,i in enumerate(DataManager.N_SEED_SONGS)}
        self.ground_truths_first["val"] = {i: tmp[ind][2] for ind,i in enumerate(DataManager.N_SEED_SONGS)}

        tmp = [self.get_ground_truth("test", n_start_songs=i) for i in DataManager.N_SEED_SONGS]
        self.seed_tracks["test"] = {i: tmp[ind][0] for ind,i in enumerate(DataManager.N_SEED_SONGS)}
        self.ground_truths["test"] = {i: tmp[ind][1] for ind,i in enumerate(DataManager.N_SEED_SONGS)}
        self.ground_truths_first["test"] = {i: tmp[ind][2] for ind,i in enumerate(DataManager.N_SEED_SONGS)}

        self.binary_train_set = self.get_train_set()
        self.prepare_charts()

    def load_playlist_track(self):
        self.playlist_track = scipy.sparse.load_npz("%s/rta_input/playlist_track.npz" % self.foldername)
  
    def load_playlist_artist(self):
        self.playlist_artist = scipy.sparse.load_npz("%s/rta_input/playlist_artist.npz" % self.foldername)

    def load_playlist_album(self):
        self.playlist_album = scipy.sparse.load_npz("%s/rta_input/playlist_album.npz" % self.foldername)

    def load_metadata(self):
        self.song_album = np.load("%s/rta_input/song_album.npy" % self.foldername)
        self.song_artist = np.load("%s/rta_input/song_artist.npy" % self.foldername)
        song_infos_sorted = sorted([(info['id'], info['count'], info['duration_ms']) for info in self.tracks_info.values()])
        self.song_pop = [c[1] for c in song_infos_sorted]
        self.song_duration = [c[2] for c in song_infos_sorted]

        with open("%s/rta_input/album_ids.pkl" % self.foldername, 'rb') as f:
            self.album_ids = pickle.load(f)
        with open("%s/rta_input/artist_ids.pkl" % self.foldername, 'rb') as f:
            self.artist_ids = pickle.load(f)
        with open("%s/rta_input/artist_songs.pkl" % self.foldername, 'rb') as f:
            self.artist_songs = pickle.load(f)
        with open("%s/rta_input/album_songs.pkl" % self.foldername, 'rb') as f:
            self.album_songs = pickle.load(f)
        with open("%s/rta_input/artist_names.pkl" % self.foldername, 'rb') as f:
            self.artist_names = pickle.load(f)
        with open("%s/rta_input/album_names.pkl" % self.foldername, 'rb') as f:
            self.album_names = pickle.load(f)

    def load_track_info(self):
        with open("%s/rta_input/tracks_info.json" % self.foldername) as f:
            self.tracks_info = json.load(f)

    def get_duration_bucket(self, x):
        MAX_DURATION = 1200000
        if isinstance(x, torch.Tensor):
            buckets = torch.div(40 * x, MAX_DURATION, rounding_mode='trunc')
        else:
            buckets = (40 * np.array(x) / MAX_DURATION).astype(int)
        buckets = buckets * (buckets>0)
        buckets = buckets*(buckets<40) + 39*(buckets>=40)
        return buckets

    def get_pop_bucket(self, x):
        x[x==0] = 1
        MAX_POP = 45394
        if isinstance(x, torch.Tensor):
            buckets = 1 + torch.div(100*torch.log(x/2), np.log(MAX_POP/2), rounding_mode='trunc')
        else:
            buckets = 1 + (100*(np.log(x)-np.log(2)) / (np.log(MAX_POP)-np.log(2))).astype(int)
        buckets = buckets*(buckets>0)
        buckets = buckets*(buckets<100) + 99*(buckets>=100)
        return buckets

    def prepare_charts(self):
        track_counts = {v["id"]: v["count"] for k, v in self.tracks_info.items()}
        self.ordered_tracks = [e[0] for e in sorted(track_counts.items(), key=operator.itemgetter(1), reverse=True)]
        self.ordered_tracks.insert(0, self.n_tracks)
        self.tracks_rank = np.zeros(self.n_tracks+1, dtype=np.int32)
        for i, t in enumerate(self.ordered_tracks):
            self.tracks_rank[t] = i
        self.ordered_tracks = np.array(self.ordered_tracks)

    def split_sets(self):
        playlist_track_csc = self.playlist_track.tocsc()
        rng = np.random.default_rng()
        candidate_indices = rng.choice(
            list(set(playlist_track_csc.indices[playlist_track_csc.data>2*self.min_songs_test])), 2*self.test_size, replace=False)
        test_indices = candidate_indices[:self.test_size] 
        val_indices  = candidate_indices[self.test_size:]
        train_indices = [i for i in range(self.n_playlists) if i not in candidate_indices]

        np.save('%s/dataset_split/train_indices' % (self.foldername), train_indices)
        np.save('%s/dataset_split/val_indices'   % (self.foldername), val_indices)
        np.save('%s/dataset_split/test_indices'  % (self.foldername), test_indices)

    def get_indices(self, set_name, resplit=False):
        if resplit:
            self.split_sets()
        return np.load(f"{self.foldername}/dataset_split/{set_name}_indices.npy")

    def get_valid_playlists(self, train_indices, test_indices):
        train_tracks = set(self.playlist_track[train_indices].indices)
        test_tracks  = set(self.playlist_track[test_indices].indices)
        test_size    = len(test_indices)
        invalid_tracks = test_tracks - train_tracks
        invalid_positions = set()
        v = self.playlist_track[test_indices].tocsc()
        for i in invalid_tracks:
            invalid_positions = invalid_positions.union(set(v.indices[v.indptr[i]:v.indptr[i+1]]))
        valid_positions = np.array(sorted([p for p in range(test_size) if p not in invalid_positions]))
        return test_indices[valid_positions]
  
    def get_ground_truth(self, set_name, binary=True, resplit=False, n_start_songs=False):
        if not n_start_songs:
            n_start_songs = self.min_songs_test
        indices = self.get_indices(set_name, resplit)
        data = self.playlist_track[indices[1000*(n_start_songs-1): 1000*n_start_songs]]
        ground_truth_array = data.multiply(data>n_start_songs)
        ground_truth_first = data.multiply(data==(n_start_songs+1))
        start_data = data - ground_truth_array
        if binary:
            start_data = (start_data>0).astype(np.int32)

        ground_truth_list = []
        ground_truth_list_first = []
        for i in range(data.shape[0]):
            ground_truth_list.append(set(ground_truth_array.indices[ground_truth_array.indptr[i]:ground_truth_array.indptr[i+1]]))
            ground_truth_list_first.append(set(ground_truth_first.indices[ground_truth_first.indptr[i]:ground_truth_first.indptr[i+1]]))
        return start_data, ground_truth_list, ground_truth_list_first

    def get_train_set(self, binary=True, resplit=False):
        train_indices = self.get_indices("train", resplit)
        train_set = self.playlist_track[train_indices]
        if binary:
            train_set = (train_set>0).astype(np.int32)
        return train_set

    def get_test_data(self, mode, n_recos=500, test_batch_size=50):
        gt_test = []
        for i in DataManager.N_SEED_SONGS:
            gt_test += self.ground_truths[mode][i]
        test_evaluator = Evaluator(self, gt=np.array(gt_test), n_recos=n_recos)

        if mode=="test":
            test_dataset = EvaluationDataset(self, self.test_indices)
        else:
            test_dataset = EvaluationDataset(self, self.val_indices)

        test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=0)
        return test_evaluator, test_dataloader


class negative_sampler(object):
    """A class to speed up negative sampling. 
    Instead of sampling uniformly at every call,
    we shuffle the list of tracks once, then read chunk by chunk.
    """
    def __init__(self, n_max):
        self.n_max = n_max
        self.current_n = 0
        self.values = np.arange(n_max)
        np.random.shuffle(self.values)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self, size=1):
        if self.current_n + size >= self.n_max:
            np.random.shuffle(self.values)
            self.current_n = 0
        neg_samples = self.values[self.current_n:self.current_n+size]
        self.current_n += size
        return neg_samples


class SequentialTrainDataset(Dataset):
    def __init__(self, data_manager, indices, max_size=50, n_neg=10, sample_size=None):
        super().__init__()
        self.max_size = max_size
        self.n_neg = n_neg
        self.data_manager = data_manager

        # (1) sample_size 적용 (필요 시)
        if sample_size is not None and sample_size < len(indices):
            chosen_indices = np.random.choice(indices, size=sample_size, replace=False)
        else:
            chosen_indices = indices
        
        # (2) 길이<2 세션 제외하기
        indices_kept = []
        for idx in chosen_indices:
            items = data_manager.playlist_track[idx].indices
            if len(items) >= 2:  # 길이2 이상만
                indices_kept.append(idx)
        self.data = data_manager.playlist_track[indices_kept]

        # 제외한 후 최종 데이터
        self.data = self.data_manager.playlist_track[indices_kept]
        print(f"SequentialTrainDataset created with {len(indices_kept)} (>=2 items) out of {len(chosen_indices)} total.")

        # 네거티브 샘플 생성기
        self.neg_generator = negative_sampler(self.data_manager.n_tracks+1)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        row = self.data[index]
        items = row.indices  # 곡 인덱스(0-based)
        pos   = row.data     # 트랙 재생순서 or 위치 정보 (필요 시 사용)

        # 재생순서대로 정렬
        pairs = sorted(zip(pos, items), key=lambda x: x[0])
        seq = np.array([itm for _, itm in pairs])
        l   = len(seq)

        # (필요 시) 랜덤 crop
        if l > self.max_size:
            start = np.random.randint(0, l - self.max_size)
            seq = seq[start : start+self.max_size]
            l   = len(seq)

        # negative 샘플
        negs = self.sample_except(seq)

        # +1 shift
        seq_t = torch.LongTensor(seq + 1)
        negs_t= torch.LongTensor([x+1 for x in negs])
        return seq_t, negs_t

    def sample_except(self, seq):
        raw = self.neg_generator.next(self.n_neg)
        diff = set(raw).difference(seq)
        while len(diff)<self.n_neg:
            needed = self.n_neg - len(diff)
            diff = diff.union(set(self.neg_generator.next(needed)).difference(seq))
        return list(diff)


class EvaluationDataset(Dataset):
    """
    For each playlist, we choose 'n_seed' = floor(index/1000)+1 first items as seed,
    the rest are target
    """
    def __init__(self, data_manager, indices):
        super().__init__()
        self.data = data_manager.playlist_track[indices]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        n_seed = math.floor(idx/1000)+1
        row = self.data[idx]
        items = row.indices+1
        pos   = row.data
        sorted_pairs = sorted(zip(pos, items), key=lambda x: x[0])
        sorted_items = [itm for _,itm in sorted_pairs]
        # n_seed개 아이템만 seed로 사용
        seed_items = sorted_items[:n_seed]
        return np.array(seed_items)  # shape [n_seed]


def pad_collate(batch):
    seqs, negs = zip(*batch)
    lens = [len(s) for s in seqs]

    xx_pad = pad_sequence(seqs, batch_first=True, padding_value=0)
    yy_neg = torch.stack(negs, dim=0)  # [B, n_neg]
    x_lens = torch.LongTensor(lens)
    return xx_pad, yy_neg, x_lens
