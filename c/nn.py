import pickle
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Union
from sklearn.model_selection import StratifiedShuffleSplit
import clang.cindex
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, classification_report, balanced_accuracy_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import torch
import numpy as np

def input_x_gradient_batch(
    model,
    dataloader,
    device,
    pad_idx: int = 0,
    normalize: bool = True,
    max_batches: int = None,      # set an int to limit for quick tests
):
    """
    Compute Input×Gradient saliency per token for each sample in `dataloader`.

    Returns:
        all_tokens_gxi: list[np.ndarray] of shape [seq_len] per sample (saliency per token)
        all_preds:      list[int] predicted label per sample (0/1)
        all_labels:     list[int] true label per sample (if labels in dataloader)
    """
    base = model.module if isinstance(model, torch.nn.DataParallel) else model
    emb_layer = base.word_embeddings

    model.eval()  # disable dropout etc.

    # storage for the embedding output (we retain grad on it)
    storage = {"emb_out": None}

    def fwd_hook(module, inputs, output):
        # output shape: [B, L, E]
        storage["emb_out"] = output
        output.retain_grad()

    hook = emb_layer.register_forward_hook(fwd_hook)

    all_tokens_gxi = []
    all_preds = []
    all_labels = []

    seen_batches = 0
    for batch in dataloader:
        # dataloader yields (input_ids, labels)
        if len(batch) == 2:
            xb, yb = batch
            yb = yb.to(device)
        else:
            xb = batch[0]
            yb = None

        xb = xb.to(device)

        # 1) forward (NO torch.no_grad here; we need grads)
        model.zero_grad(set_to_none=True)
        outputs = model(xb)                  # shape [B, 1]
        # outputs are sigmoid probs from your model
        probs = outputs.squeeze(-1)          # [B]
        preds = (probs >= 0.5).long()        # predicted class 0/1

        # 2) build a scalar that sums the selected class score per sample
        # For binary with single output prob p(y=1), selecting class 1 = p,
        # selecting class 0 = (1-p). We can use signed trick with logits,
        # but your model returns probs, so do it explicitly:
        selected = torch.where(preds == 1, probs, 1.0 - probs)  # [B]
        selected.sum().backward(retain_graph=False)  # backprop to emb_out

        # 3) fetch gradients & embeddings at the hook point
        emb_out = storage["emb_out"]                 # [B, L, E]
        grads = emb_out.grad                         # [B, L, E]

        # 4) Input×Gradient per token = sum over embedding dim of |emb * grad|
        gxi = (emb_out * grads).abs().sum(dim=-1)    # [B, L]

        # 5) mask padding tokens
        pad_mask = (xb != pad_idx).float()           # [B, L] 1 for non-pad
        gxi = gxi * pad_mask

        # 6) optional: normalize per example to [0,1]
        if normalize:
            # avoid divide-by-zero
            mins = gxi.min(dim=1, keepdim=True).values
            maxs = gxi.max(dim=1, keepdim=True).values
            denom = (maxs - mins).clamp_min(1e-8)
            gxi = (gxi - mins) / denom

        # 7) move to CPU lists
        if len(preds.shape) < 1:
            preds = torch.unsqueeze(preds, dim=0)
        gxi_cpu = gxi.detach().cpu().numpy()         # (B, L)
        all_tokens_gxi.extend([row for row in gxi_cpu])
        all_preds.extend(preds.detach().cpu().tolist())
        if yb is not None:
            all_labels.extend(yb.detach().cpu().long().squeeze(-1).tolist())

        seen_batches += 1
        if max_batches is not None and seen_batches >= max_batches:
            break

        # clear per-batch grads to lower peak memory
        storage["emb_out"] = None
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

    # remove the hook when done
    hook.remove()

    return all_tokens_gxi, all_preds, all_labels


with open('all_labels.pkl', 'rb') as handle:
    labels = pickle.load(handle)

with open('slice_lines.pkl', 'rb') as handle:
    slice_imp = pickle.load(handle)

MAX_LEN = 3058
print("MAX_LEN", MAX_LEN)

embedding_dim = 512
hidden_size = 1024
num_epochs = 1
batch_size = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Conv1DModel(nn.Module):

    def __init__(self, embedding_dim, vocab_size, channel_size=MAX_LEN, target_size=1):
        super(Conv1DModel, self).__init__()

        self.word_embeddings = nn.Embedding(vocab_size+10, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(0.5)
        self.conv1d_0 = nn.Conv1d(channel_size, 256, 7, stride=3, padding=0)
        self.conv1d_1 = nn.Conv1d(256, 128, 7, stride=3, padding=0)
        self.relu = torch.nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(128, 1024)
        self.explain = nn.Linear(1024, channel_size)
        self.output = nn.Linear(channel_size, target_size)
        self.sigmoid = nn.Sigmoid()
        # self.dropout = nn.Dropout(0.5)



    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        embeds = self.dropout(embeds)
        convd = self.conv1d_0(embeds)
        convd = self.relu(convd)
        convd = self.conv1d_1(convd)
        convd = self.relu(convd)
        pooled = torch.squeeze(self.avgpool(convd))
        pooled = self.relu(pooled)
        linear = self.relu(self.linear(pooled))
        linear = self.dropout(linear)
        explain = self.sigmoid(self.explain(linear))
        output = self.sigmoid(self.output(explain))
        return output
    
X_tokens = {i:[] for i in labels}
X_attn = {i:[] for i in labels}

ground_truth = {i:[] for i in labels}

for i in X_tokens:
    for j in labels[i]:
        # print(j)
        tokens_m = []
        attn_tokens = []
        skip = 0
        index = clang.cindex.Index.create()
        trans_unit = index.parse('./c_files/'+j)
        tokens = trans_unit.get_tokens(extent=trans_unit.cursor.extent)
        # print(slice_imp[j], labels[i][j])
        for token in tokens:
            if skip > 0:
                skip -= 1
                continue
            if str(token.kind.name) != 'COMMENT':
                if (j not in token.spelling):
                    # print(token.spelling, token.kind.name, str(token.location).split(',')[1].split(' ')[2])
                    tokens_m.append(str(token.spelling))
                    if str(token.spelling) in [',', '(', ')', ';', '{', '{']:
                        attn_tokens.append(0)
                    elif str(token.location).split(',')[1].split(' ')[2] in slice_imp[j]:
                        attn_tokens.append(1)
                    else:
                        attn_tokens.append(0)
                else:
                    skip = 2
                
        X_tokens[i].append(tokens_m)
        X_attn[i].append(attn_tokens)
        ground_truth[i].append(labels[i][j])

        if len(tokens_m) != len(attn_tokens):
            raise ValueError('A very specific bad thing happened.')


def loadDictionary(lines):
    words = []
    for line in lines:
        words += line
    words = ['<unk>', '<pad>', '<s>', '</s>', '<mask>'] + list(set(words))
    vocab = {j:i+1 for i,j in  enumerate(words)}
    return vocab


def vectorize_sequence(lines, vocab):
    maxlen = max([len(i) for i in lines])
    feature_vecs = []

    for line in lines:
        feature_vec = np.zeros(maxlen)
        for i, t in enumerate(line):
            key=t.strip()
            if key == '' or 'eResult' in key: 
                continue
            if key not in vocab:
                feature_vec[i] = 0
            else:
                feature_vec[i] = vocab[key]
        feature_vecs.append(feature_vec)
    feature_vecs = np.array(feature_vecs)
    return feature_vecs



preds = []
y_tests = []

for config in X_tokens:
    # if config == '47':
    #     print("Skip 47")
    #     continue
    raw_text = X_tokens[config][:]
    label = np.array(ground_truth[config][:])
    vocab = loadDictionary(X_tokens[config])
    vocab_size = len(vocab) + 1  # or your known vocab size

    X = vectorize_sequence(raw_text, vocab)



    for seed in [42, 1975, 5827, 7421, 9876]:
        print("Seed", seed)
        sss = StratifiedShuffleSplit(n_splits=5, test_size=1/5, random_state=seed)

        for i, (train_index, test_index) in enumerate(sss.split(raw_text, label)):
            X_train, X_test, y_train, y_test = X[train_index], X[test_index], label[train_index], label[test_index]
            print(np.bincount(y_train))


            # Convert to tensors
            X_train_tensor = torch.tensor(X_train, dtype=torch.long)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float).unsqueeze(1)
            X_test_tensor = torch.tensor(X_test, dtype=torch.long)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float).unsqueeze(1)

            # Dataloaders
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)

            lr=0.001
            criterion = nn.BCELoss()
            model = Conv1DModel(1024, len(vocab)+1, X.shape[1])
            model = nn.DataParallel(model) # Wrap the model with DataParallel
            model = model.to('cuda') # Move the model to the GPU
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            # # Handle class imbalance
            # class_counts = np.bincount(y_train)
            # class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)
            # class_weights = class_weights.to(device)
            # criterion = nn.CrossEntropyLoss(weight=class_weights)

            optimizer = optim.AdamW(model.parameters(), lr=2e-4)            # Training
            model.train()
            for epoch in range(num_epochs):
                epoch_loss = 0.0
                total_batches = 0

                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    outputs = model(xb)
                    loss = criterion(outputs, yb)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    total_batches += 1

                avg_loss = epoch_loss / total_batches
                print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")

            model.eval()
            all_preds, all_y = [], []
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb = xb.to(device)
                    outputs = model(xb)                            # shape: [B, 1]
                    # print(outputs)
                    if len(outputs.shape) > 1:
                        predicted = (outputs > 0.5).long().squeeze(1)    # shape: [B]
                    else:
                        predicted = (outputs > 0.5).long()
                    all_preds.extend(predicted.cpu().numpy())
                    all_y.extend(yb.numpy())

            tokens_gxi, preds_test, labels_test = input_x_gradient_batch(
                model=model,
                dataloader=test_loader,
                device=device,
                pad_idx=0,           # your embedding padding_idx is 0
                normalize=True,
                # max_batches=1,     # uncomment to test on just one batch
            )

            print(np.array(tokens_gxi)[0])

            np.save(config+str(seed)+str(i)+'base.npy',tokens_gxi)

            print(f"NN Iteration REPORT: \n{metrics.classification_report(all_y, all_preds, digits=4)}")
            print(f"NN Iteration CONFUSION MATRIX: \n{metrics.confusion_matrix(all_y, all_preds)}")

            preds += all_preds
            y_tests += all_y
            # break

    # Final aggregated report
    print(f"\nNN FINAL REPORT:\n{metrics.classification_report(y_tests, preds, digits=4)}")
    print(f"NN FINAL CONFUSION MATRIX:\n{metrics.confusion_matrix(y_tests, preds)}")