# -*- coding: utf-8 -*-

import copy
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

from utils import data_utils


# Set training configuration

training_config = {
    "train_data": {
        "input_path": "Clotho.v2.1",
        "dataset": "Clotho.v2.1",
        "data_splits": {
            "train": "development_captions.json",
            "val": "validation_captions.json",
            "test": "evaluation_captions.json"
        },
        "text_tokens": "tokens",
        "audio_features": "audio_logmel.hdf5",
        "word_embeddings": "word2vec_emb.pkl",
        "vocabulary": "vocab_info.pkl"
    }
}


# Set Device

cuda = torch.cuda.is_available()

device = torch.device("cuda" if cuda else "cpu")
print("Device Type:", device)


# Load data
text_datasets, vocabulary = data_utils.load_data(training_config["train_data"])

BATCH_SIZE = 64

text_loaders = {}
for split in ["train", "val", "test"]:
    _dataset = text_datasets[split]
    _loader = DataLoader(dataset=_dataset, batch_size=BATCH_SIZE,
                          shuffle=True, collate_fn=data_utils.collate_fn)
    text_loaders[split] = _loader


# Load Weights

vocabulary.weights = None
emb = vocabulary.get_weights()


# Model Definition

class CNNModule(nn.Module):

    def __init__(self):
        super(CNNModule, self).__init__()

        self.features = nn.Sequential(
            # Conv2D block
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.BatchNorm2d(32),

            # LPPool
            nn.LPPool2d(4, (2, 4)),

            # Conv2D block
            nn.Conv2d(32, 128, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.BatchNorm2d(128),

            # Conv2D block
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.BatchNorm2d(128),

            # LPPool
            nn.LPPool2d(4, (2, 4)),

            # Conv2D block
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.BatchNorm2d(128),

            # Conv2D block
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.BatchNorm2d(128),

            # LPPool
            nn.LPPool2d(4, (1, 4)),

            nn.Dropout(0.5)
        )

        self.features.apply(init_weights)

    def forward(self, x):
        """
        :param x: tensor, (batch_size, time_steps, Mel_bands).
        :return: tensor, (batch_size, time_steps / 4, 128 * Mel_bands / 64).
        """
        x = x.unsqueeze(1)
        x = self.features(x)
        x = x.transpose(1, 2).contiguous().flatten(-2)

        return x

class CRNNEncoder(nn.Module):

    def __init__(self, in_dim, out_dim, up_sampling=False):
        super(CRNNEncoder, self).__init__()
        self.up_sampling = up_sampling

        self.cnn = CNNModule()

        with torch.no_grad():
            rnn_in_dim = self.cnn(torch.randn(1, 500, in_dim)).shape
            rnn_in_dim = rnn_in_dim[-1]

        self.gru = nn.GRU(rnn_in_dim, out_dim // 2, bidirectional=True, batch_first=True)

    def forward(self, x):
        """
        :param x: tensor, (batch_size, time_steps, Mel_bands).
        :return: tensor, (batch_size, embed_dim).
        """
        batch, time, dim = x.shape

        x = self.cnn(x)
        x, _ = self.gru(x)

        if self.up_sampling:
            x = F.interpolate(x.transpose(1, 2), time, mode="linear", align_corners=False).transpose(1, 2)

        x = torch.mean(x, dim=1, keepdim=False)

        return x

def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Conv1d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class WordEmbedding(nn.Module):

    def __init__(self, num_word, embed_dim, word_embeds=None, trainable=False):
        super(WordEmbedding, self).__init__()


        self.embedding = nn.Embedding(num_word, embed_dim)

        if word_embeds is not None:
            self.load_pretrained_embedding(word_embeds)
        else:
            nn.init.kaiming_uniform_(self.embedding.weight)

        for para in self.embedding.parameters():
            para.requires_grad = trainable

    def load_pretrained_embedding(self, weight):
        assert weight.shape[0] == self.embedding.weight.size()[0], "vocabulary size mismatch!"

        weight = torch.as_tensor(weight).float()
        self.embedding.weight = nn.Parameter(weight)

    def forward(self, queries, query_lens):
        """
        :param queries: tensor, (batch_size, query_max_len).
        :param query_lens: list, [N_{1}, ..., N_{batch_size}].
        :return: (batch_size, query_max_len, embed_dim).
        """

        query_lens = torch.as_tensor(query_lens)
        batch_size, query_max = queries.size()

        query_embeds = self.embedding(queries)

        mask = torch.arange(query_max, device='cpu').repeat(batch_size).view(batch_size, query_max)
        mask = (mask < query_lens.view(-1, 1)).to(query_embeds.device)

        query_embeds = query_embeds * mask.unsqueeze(-1)

        return query_embeds


class WordEncoder(nn.Module):

    def __init__(self, num_word, embed_dim, word_embeds=None, trainable=False):
        super(WordEncoder, self).__init__()

        self.word_embedding = WordEmbedding(num_word, embed_dim, word_embeds, trainable)

    def forward(self, queries, query_lens):
        """
        :param queries: tensor, (batch_size, query_max_len).
        :param query_lens: list, [N_{1}, ..., N_{batch_size}].
        :return: (batch_size, embed_dim).
        """

        query_embeds = self.word_embedding(queries, query_lens)

        query_embeds = torch.mean(query_embeds, dim=1, keepdim=False)

        return query_embeds

class CRNNWordModel(nn.Module):

    def __init__(self, in_dim, out_dim, num_word, embed_dim, up_sampling=True, word_embeds=None, trainable=False):
        super(CRNNWordModel, self).__init__()


        self.audio_encoder = CRNNEncoder(in_dim, out_dim, up_sampling)

        self.text_encoder = WordEncoder(num_word, embed_dim, word_embeds, trainable)

    def forward(self, audio_feats, queries, query_lens):
        """
        :param audio_feats: tensor, (batch_size, time_steps, Mel_bands).
        :param queries: tensor, (batch_size, query_max_len).
        :param query_lens: list, [N_{1}, ..., N_{batch_size}].
        :return: (batch_size, embed_dim), (batch_size, embed_dim).
        """

        audio_embeds = self.audio_encoder(audio_feats)

        query_embeds = self.text_encoder(queries, query_lens)

        # audio_embeds: [N, E]    query_embeds: [N, E]
        return audio_embeds, query_embeds

model = CRNNWordModel(in_dim=64, out_dim=300, num_word=len(vocabulary), embed_dim=300, up_sampling=True, word_embeds=emb, trainable=False)
print(model)


# Loss Function

def score(audio_embed, query_embed):
    """
    Compute an audio-query score.

    :param audio_embed: tensor, (E, ).
    :param query_embed: tensor, (E, ).
    :return: similarity score: tensor, (1, ).
    """
    sim = torch.dot(audio_embed.type(torch.float32), query_embed.type(torch.float32))

    return sim
    
class TripletRankingLoss(nn.Module):

    def __init__(self, margin=1.0):
        super().__init__()

        self.margin = margin

    def forward(self, audio_embeds, query_embeds, infos):
        """
        :param audio_embeds: tensor, (N, E).
        :param query_embeds: tensor, (N, E).
        :param infos: list of audio infos.
        :return:
        """
        N = audio_embeds.size(0)
        a = 0.0
        a_imp = 0.0
        q_imp = 0.0

        # Computes the triplet margin ranking loss for each anchor audio/query pair.
        # The impostor audio/query is randomly sampled from the mini-batch.
        loss = torch.tensor(0., device=audio_embeds.device, requires_grad=True)

        for i in range(N):
            A_imp_idx = i
            while infos[A_imp_idx]["fid"] == infos[i]["fid"]:
                A_imp_idx = np.random.randint(0, N)

            Q_imp_idx = i
            while infos[Q_imp_idx]["fid"] == infos[i]["fid"]:
                Q_imp_idx = np.random.randint(0, N)

            anchor_score = score(audio_embeds[i], query_embeds[i])

            A_imp_score = score(audio_embeds[A_imp_idx], query_embeds[i])

            Q_imp_score = score(audio_embeds[i], query_embeds[Q_imp_idx])
            
            A2Q_diff = self.margin + Q_imp_score - anchor_score
            if (A2Q_diff.data > 0.).all():
                loss = loss + A2Q_diff

            Q2A_diff = self.margin + A_imp_score - anchor_score
            if (Q2A_diff.data > 0.).all():
                loss = loss + Q2A_diff

            a += float(anchor_score)
            a_imp += float(A_imp_score)
            q_imp += float(Q_imp_score)

        loss = loss / N
        a = a / N
        a_imp = a_imp / N
        q_imp = q_imp / N

        return loss, a, a_imp, q_imp


# Define Hyperparameters

criterion = TripletRankingLoss(margin=2.0)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, threshold=0.01, threshold_mode='abs')
scaler = torch.cuda.amp.GradScaler()


# Training Code

def train(model, optimizer, criterion, data_loader, split, outfile):
    criterion.to(device=device)
    model.to(device=device)

    model.train()
    total_loss = 0.0

    batch_bar = tqdm(total=len(data_loader), dynamic_ncols=True, leave=False, position=0, desc=split.capitalize()) 

    for batch_idx, data in enumerate(data_loader, 0):
        torch.cuda.empty_cache()

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Get the inputs; data is a list of tuples (audio_feats, audio_lens, queries, query_lens, infos)
        audio_feats, audio_lens, queries, query_lens, infos = data

        with torch.cuda.amp.autocast():
            audio_feats, queries = audio_feats.to(device), queries.to(device)

            # Forward + backward + optimize
            audio_embeds, query_embeds = model(audio_feats, queries, query_lens)
        
        loss, a, a_imp, q_imp = criterion(audio_embeds, query_embeds, infos)

        scaler.scale(loss).backward() # This is a replacement for loss.backward()
        scaler.step(optimizer) # This is a replacement for optimizer.step()
        scaler.update() # This is something added just for FP16

        total_loss += float(loss)
          
        batch_bar.set_postfix(loss="{:.04f}".format(float(loss)), lr="{:.06f}".format(optimizer.param_groups[0]['lr']), 
                              a="{:.04f}".format(float(a)), a_imp="{:.04f}".format(float(a_imp)), q_imp="{:.04f}".format(float(q_imp)))
        batch_bar.update()

    batch_bar.close()     
    loss = total_loss / len(data_loader)    
    print("Loss: {:.4f}\tA: {:.4f}\t\A_imp: {:.4f}\tQ_imp: {:.4f}\tLR: {:.6f}".format(loss, a, a_imp, q_imp, optimizer.param_groups[0]['lr']))
    if outfile:
        outfile.write("Loss: {:.4f}\tA: {:.4f}\t\A_imp: {:.4f}\tQ_imp: {:.4f}\tLR: {:.6f}".format(loss, a, a_imp, q_imp, optimizer.param_groups[0]['lr']))

    return loss

def eval(model, criterion, data_loader, split, outfile):
    criterion.to(device=device)
    model.to(device=device)

    model.eval()

    eval_loss, eval_steps = 0.0, 0

    batch_bar = tqdm(total=len(data_loader), dynamic_ncols=True, leave=False, position=0, desc=split.capitalize())

    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader, 0):
            torch.cuda.empty_cache()

            audio_feats, audio_lens, queries, query_lens, infos = data
            audio_feats, queries = audio_feats.to(device), queries.to(device)

            audio_embeds, query_embeds = model(audio_feats, queries, query_lens)

            loss, a, a_imp, q_imp = criterion(audio_embeds, query_embeds, infos)
            eval_loss += loss.cpu().numpy()
            eval_steps += 1

            batch_bar.set_postfix(loss="{:.04f}".format(float(loss)),
                                a="{:.04f}".format(float(a)), a_imp="{:.04f}".format(float(a_imp)), q_imp="{:.04f}".format(float(q_imp)))
            batch_bar.update()

    batch_bar.close()   

    total_loss = eval_loss / (eval_steps + 1e-20)  
    print(split.capitalize(), "Loss: {:.5f}\tA: {:.4f}\t\A_imp: {:.4f}\tQ_imp: {:.4f}".format(total_loss, a, a_imp, q_imp))
    if outfile:
      outfile.write(split.capitalize() + " Loss: {:.5f}\tA: {:.4f}\t\A_imp: {:.4f}\tQ_imp: {:.4f}".format(total_loss, a, a_imp, q_imp))

    return total_loss

model_dir = "train/"
info_dir = "train/"
outfile = open(info_dir + "logs.txt", 'w')

epochs = []
train_loss = []
val_loss = []

for epoch in range(100):
  out = "Epoch " + str(epoch+1) + " --->"
  print(out)
  outfile.write(out)

  loss = train(model, optimizer, criterion, text_loaders["train"], 'train', outfile)
  torch.save(model.state_dict(), model_dir + "model_" + str(epoch+1) + ".pth")

  epoch_results = {}
  for split in ["val"]:#, "test"]:
      epoch_results["{0}_loss".format(split)] = eval(model, criterion, text_loaders[split], split, outfile)

  epochs.append(epoch+1)
  train_loss.append(loss)
  val_loss.append(epoch_results['val_loss'])

  scheduler.step(epoch_results['val_loss'])

outfile.close()
