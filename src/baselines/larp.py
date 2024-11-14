import random
import numpy as np
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta as td
from math import sqrt, log10, exp
from typing import List

class LARP:
    def __init__(self, k=100, sample_size=1000, sampling='recent', embed_dim=256, queue_size=57600, momentum=0.995, session_key = 'SessionId', item_key= 'ItemId', time_key= 'Time'):
        self.k = k
        self.sample_size = sample_size
        self.sampling = sampling

        self.embed_dim = embed_dim
        self.queue_size = queue_size
        self.momentum = momentum

        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key

        # Updated during recommendation
        self.session = -1
        self.session_items = []
        self.relevant_sessions = set()

        # Cache relations
        self.session_item_map = dict()
        self.item_session_map = dict()
        self.session_time = dict()
        self.min_time = -1

        # Initialize embedding projections
        self.audio_proj = np.random.randn(self.embed_dim)
        self.text_proj = np.random.randn(self.embed_dim)

        # Create momentum encoders
        self.audio_proj_m = self.audio_proj.copy()
        self.text_proj_m = self.text_proj.copy()

        # Create the queue
        self.audio_queue = np.random.randn(self.embed_dim, queue_size)
        self.text_queue = np.random.randn(self.embed_dim, queue_size)
        self.queue_ptr = 0

    def fit(self, train):
        """
        Trains the model with session data.

        Parameters
        --------
        train: pandas.DataFrame
            Training data containing session, item, and timestamp columns.
        """

        index_session = train.columns.get_loc( self.session_key )
        index_item = train.columns.get_loc( self.item_key )
        index_time = train.columns.get_loc( self.time_key )

        session = -1
        session_items = []
        time = -1
        for row in train.itertuples(index=False):
            # Cache items of sessions
            if row[index_session] != session:
                if len(session_items) > 0:
                    self.session_item_map.update({session: session_items})
                    self.session_time.update({session: time})
                    if time < self.min_time:
                        self.min_time = time
                session = row[index_session]
                session_items = []
            time = row[index_time]
            session_items.append(row[index_item])

            # Cache sessions involving an item
            map_is = self.item_session_map.get(row[index_item])
            if map_is is None:
                map_is = set()
                self.item_session_map.update({row[index_item]: map_is})
            map_is.add(row[index_session])

        # Add the last session
        self.session_item_map.update({session: session_items})
        self.session_time.update({session: time})

    def predict_next(self, session_id, input_item_id, predict_for_item_ids, timestamp=0):
        """
        Predicts the next items for the current session.

        Parameters
        --------
        session_id : int or string
            The session ID of the event.
        input_item_id : int or string
            The item ID of the event.
        predict_for_item_ids : list
            List of item IDs to predict scores for.

        Returns
        --------
        out : np.ndarray
            Numpy array of item scores
        """
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

# Example usage:
# model = LARPPipeline()
# model.fit(training_data)
# predictions = model.predict_next(session_id, input_item_id, predict_for_item_ids)
