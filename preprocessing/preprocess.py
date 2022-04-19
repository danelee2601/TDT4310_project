import re
import copy

import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import nltk
import numpy as np

import torch
import torch.nn as nn

from utils import root_dir


class FakeNewsDataset(Dataset):
    def __init__(self,
                 kind: str,
                 tokenizer,
                 train_test_split_rseed=1,
                 in_col='text',
                 max_len=None,
                 **kwargs
                 ):
        super().__init__()
        assert kind in ['train', 'test']
        assert in_col in ['text', 'title']
        self.kind = kind
        self.tokenizer = tokenizer
        self.max_len = tokenizer.model_max_length if max_len is None else max_len
        self.in_col = in_col

        true = pd.read_csv(root_dir.joinpath('data', 'True.csv'))
        false = pd.read_csv(root_dir.joinpath('data', 'Fake.csv'))

        # provide numeric label: {True: 1, Fake: 0}
        true['category'] = 1
        false['category'] = 0

        # Merging the 2 datasets
        self.df = pd.concat([true, false])

        # clean
        self.df['text'] = self.df['text'].apply(self.clean_text)

        # split
        X_train, X_test, y_train, y_test = train_test_split(self.df[in_col],
                                                            self.df['category'],
                                                            train_size=0.8,
                                                            test_size=0.2,
                                                            random_state=train_test_split_rseed)
        self.X = X_train if self.kind == 'train' else X_test
        self.Y = y_train if self.kind == 'train' else y_test

        self.len = self.X.shape[0]

    def clean_text(self, text):
        text = text.split('(Reuters) - ')[-1]  # remove '(Reuters)' to make the task less trivial.
        text = text.strip()
        text = text.lower()
        return text

    def __getitem__(self, idx):
        sent = self.X.iloc[idx]
        y = self.Y.iloc[idx]

        # tokenize
        x = self.tokenizer(sent, return_tensors='pt', max_length=self.max_len, truncation=True, padding='max_length')
        for k in x.keys():
            x[k] = x[k][0]

        return sent, x, y

    def __len__(self):
        return self.len


# class FakeNewsDatasetGlove(Dataset):
#     def __init__(self,
#                  kind: str,
#                  glove_vectors,
#                  train_test_split_rseed=1,
#                  in_col='text',
#                  max_len=None,
#                  device= torch.device(0),
#                  **kwargs
#                  ):
#         super().__init__()
#         assert kind in ['train', 'test']
#         assert in_col in ['text', 'title']
#         self.kind = kind
#         # self.tokenizer = tokenizer
#         self.glove_vectors = glove_vectors
#         self.max_len = max_len
#         self.in_col = in_col
#         self.device = device
#
#         # emb weight - Glove
#         emb_weights = torch.FloatTensor(glove_vectors.vectors)
#         vocab_size, emb_dim = emb_weights.shape
#         self.embedding = nn.Embedding(num_embeddings=vocab_size + 1, embedding_dim=emb_dim)  # +1 for mask-token
#         self.embedding.weight[:vocab_size, :].data = emb_weights
#         self.embedding = self.embedding.to(device)
#         for param in self.embedding.parameters():
#             param.requires_grad = False
#
#         true = pd.read_csv(root_dir.joinpath('data', 'True.csv'))
#         false = pd.read_csv(root_dir.joinpath('data', 'Fake.csv'))
#
#         # provide numeric label: {True: 1, Fake: 0}
#         true['category'] = 1
#         false['category'] = 0
#
#         # Merging the 2 datasets
#         self.df = pd.concat([true, false])
#
#         # clean
#         self.df['text'] = self.df['text'].apply(self.clean_text)
#
#         # split
#         X_train, X_test, y_train, y_test = train_test_split(self.df[in_col],
#                                                             self.df['category'],
#                                                             train_size=0.9,
#                                                             test_size=0.1,
#                                                             random_state=train_test_split_rseed)
#         self.X = X_train if self.kind == 'train' else X_test
#         self.Y = y_train if self.kind == 'train' else y_test
#
#         self.len = self.X.shape[0]
#
#     def clean_text(self, text):
#         text = text.split('(Reuters) - ')[-1]  # remove '(Reuters)' to make the task less trivial.
#         text = text.strip()
#         text = text.lower()
#         return text
#
#     def key_to_index(self, word_token: str):
#         try:
#             return self.glove_vectors.key_to_index[word_token]
#         except:
#             vocab_size = self.glove_vectors.vectors.shape[0]
#             return vocab_size  # indicates to use `mask_token`
#
#     def __getitem__(self, idx):
#         sent = self.X.iloc[idx]
#         y = self.Y.iloc[idx]
#
#         # tokenize
#         # x = self.tokenizer(sent, return_tensors='pt', max_length=self.max_len, truncation=True, padding='max_length')
#         # for k in x.keys():
#         #     x[k] = x[k][0]
#         ind = [self.key_to_index(wt) for wt in nltk.word_tokenize(sent)[:self.max_len]]
#         ind = np.array(ind)
#         ind = torch.Tensor(ind).int()
#         # if len(ind) > self.max_len:
#         #     ind = ind[:self.max_len]
#
#         # emb vector
#         z = self.embedding(ind.to(self.device))  # (L, D)
#
#         # aggregate along L
#         z = z.mean(dim=0)  # (D,)
#
#         return z, y
#
#     def __len__(self):
#         return self.len


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from transformers import RobertaTokenizer
    from transformers import OpenAIGPTTokenizer

    tokenizer = RobertaTokenizer.from_pretrained("facebook/data2vec-text-base")
    # tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")
    # tokenizer.add_special_tokens({'pad_token': '<pad>'})
    train_data_loader = DataLoader(FakeNewsDataset('train', tokenizer, in_col='text'),
                                   batch_size=4,
                                   shuffle=True)

    # fetch some data
    for batch in train_data_loader:
        sents, X, Y = batch
        break

    # print
    print(sents)
    print(X)
    print(Y)
