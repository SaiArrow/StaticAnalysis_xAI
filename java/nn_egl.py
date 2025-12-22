import pickle
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Union
from transformers import PreTrainedTokenizer
from sklearn.model_selection import StratifiedShuffleSplit
import clang.cindex
import pandas as pd
from torch.nn.functional import kl_div, log_softmax, softmax
from datasets import load_dataset, Dataset, DatasetDict
from datasets import Dataset, DatasetDict
from sklearn.metrics import f1_score, confusion_matrix, classification_report, balanced_accuracy_score, accuracy_score
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import torch
import torch.nn.functional as F
import json
import numpy as np
import numpy as np
import pandas as pd
import os.path
import yaml
from pathlib import Path
import glob
import javalang
import torch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import torch
import numpy as np



import torch
import torch.nn as nn

bce_align = nn.BCELoss()

def inputxgrad_alignment_loss(
    model,
    xb,                # [B, L] token ids (long)
    rationale_mask,    # [B, L] 0/1 float or long
    pad_idx: int = 0,
    normalize: bool = True,
    second_order: bool = False,   # set True to enable second-order (costly)
):
    device = next(model.parameters()).device
    base = model.module if isinstance(model, torch.nn.DataParallel) else model
    emb_layer = base.word_embeddings

    xb = xb.to(device)
    rationale_mask = rationale_mask.to(device).float()
    pad_mask = (xb != pad_idx).float()                      # [B, L]

    storage = {"emb": None}

    def fwd_hook(module, inputs, output):
        storage["emb"] = output
        output.retain_grad()

    h = emb_layer.register_forward_hook(fwd_hook)

    model.zero_grad(set_to_none=True)
    out = model(xb)                       # [B, 1], sigmoid probs in your model
    probs = out.squeeze(-1)               # [B]
    preds = (probs >= 0.5).long()         # [B], predicted class

    # Build a scalar per sample to differentiate: p for class-1, (1-p) for class-0
    selected = torch.where(preds == 1, probs, 1.0 - probs)  # [B]

    grads = torch.autograd.grad(
        outputs=selected.sum(),
        inputs=storage["emb"],
        retain_graph=True,                 # still need graph for main BCE backward
        create_graph=second_order,         # enable if you want second-order
        allow_unused=False
    )[0]                                   # [B, L, E]

    emb_out = storage["emb"]               # [B, L, E]
    h.remove()                             # remove hook ASAP

    sal = (emb_out * grads).abs().sum(dim=-1)    # [B, L]
    sal = sal * pad_mask                         # zero out pads

    if normalize:
        mins = sal.min(dim=1, keepdim=True).values
        maxs = sal.max(dim=1, keepdim=True).values
        sal = (sal - mins) / (maxs - mins + 1e-8)

    if not second_order:
        sal = sal.detach()

    align_loss = bce_align(sal, rationale_mask)

    return align_loss


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

        model.zero_grad(set_to_none=True)
        outputs = model(xb)                  # shape [B, 1]
        # outputs are sigmoid probs from your model
        probs = outputs.squeeze(-1)          # [B]
        preds = (probs >= 0.5).long()        # predicted class 0/1

        selected = torch.where(preds == 1, probs, 1.0 - probs)  # [B]
        selected.sum().backward(retain_graph=False)  # backprop to emb_out

        emb_out = storage["emb_out"]                 # [B, L, E]
        grads = emb_out.grad                         # [B, L, E]

        gxi = (emb_out * grads).abs().sum(dim=-1)    # [B, L]

        pad_mask = (xb != pad_idx).float()           # [B, L] 1 for non-pad
        gxi = gxi * pad_mask

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




files = ['jbmc.csv', 'jayhorn.csv']
X_tokens = {i:[] for i in files}
X_attn = {i:[] for i in files}

ground_truth = {i:[] for i in files}


embedding_dim = 128
hidden_size = 256
num_epochs = 50
batch_size = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Conv1DModel(nn.Module):

    def __init__(self, embedding_dim, vocab_size, channel_size=1024, target_size=1):
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


class GetOutOfLoop( Exception ):
    pass

with open('relevant.json', 'r') as file:
    syr = json.load(file)

relevant_dict = {}
for i in range(len(syr)):
    relevant_dict[syr[i]['name']] = syr[i]['relevant_lines']

for config in files:

    relevant_matrix = []
    df = pd.read_csv(config)
    df = df[df['label']!='UNK'].reset_index()
    raw_text = []
    Y = []
    for i in range(len(df['file_path'])):
        javafiles = []
        current_text = []
        file_name = "./java/"+df['file_path'][i]+".yml"
        if(os.path.isfile(file_name)) and (df['file_path'][i] in relevant_dict.keys()):
            conf = yaml.safe_load(Path(file_name).read_text())
            if 'input_files' not in conf.keys() or conf['options']['language'] != 'Java':
                print(conf, file_name)
            for file in conf['input_files']:
                if not len(glob.glob('/'.join(file_name.split('/')[:-1])+'/**/'+file+'/*.java', recursive=True)) and not ('common' in file):
                    print(file)
                if len(glob.glob('/'.join(file_name.split('/')[:-1])+'/**/'+file+'/**/*.java', recursive=True))  and not ('common' in file):
                    for javafile in glob.glob('/'.join(file_name.split('/')[:-1])+'/**/'+file+'/**/*.java', recursive=True):
                        javafiles.append(javafile)
            temp_matrix = []
            for javafile in set(javafiles):
                data = open(javafile).read()
                tokens = list(javalang.tokenizer.tokenize(data))
                current_text += [str(token).split(' line')[0].split(' "')[1][:-1] for token in tokens]
                
                if 'Main.java' in javafile:
                    if df['file_path'][i] in relevant_dict.keys():
                        # print('Yes')
                        line_num = list(set([i.split(' NORMAL')[0] for i in open('./slices_refined/' + df['file_path'][i] + '.txt', 'r').read().split('\n')][:-1]))
                        # line_num = []
                        # for line in relevant_dict[df['file_path'][i]]:
                        #     lines = list(line.keys())[0]
                        #     if '-' in lines:
                        #         line_num += list(str(j) for j in range(int(lines.split('-')[0]), int(lines.split('-')[1]) + 1))
                        #     else:
                        #         line_num += [str(lines)]
                        
                        for token in tokens:
                            if ('{' in str(token)) or ('}' in str(token)):
                                # print('Test', token)
                                temp_matrix.append(0)
                            elif str(token).split('line ')[1].split(',')[0] in line_num:
                                temp_matrix.append(1)
                                # print(str(token).split(' line')[0].split(' "')[1][:-1])
                                # print(str(token).split('line ')[1].split(',')[0])
                            else:
                                # print(str(token).split('line ')[1].split(',')[0], token)
                                temp_matrix.append(0)
                        

                        if not (sum(temp_matrix)):
                            print(relevant_dict[df['file_path'][i]], df['file_path'][i], temp_matrix, line_num)


                        # raise GetOutOfLoop
                else:
                    temp_matrix += len(tokens) * [0]
                    
            relevant_matrix.append(temp_matrix)
            raw_text.append(current_text)

            if len(current_text) != len(temp_matrix):
                print(len(current_text), len(temp_matrix), df['file_path'][i])
                raise ValueError('A very specific bad thing happened.')


            if df['label'][i] in ['TN', 'TP']:
                Y.append(1)
            elif df['label'][i] in ['FN', 'FP']:
                Y.append(0)

        else:
            # print(df['file_path'][i], "Check File")
            continue
    X_tokens[config] = raw_text
    X_attn[config] = relevant_matrix
    ground_truth[config] = Y

def mask_input(input_ids, attn_mask, mask_token_id=0):
    return torch.where(attn_mask.bool(), input_ids, mask_token_id)

def compute_kl_loss(original_output, masked_output):
    logp = log_softmax(original_output, dim=-1)
    p_masked = softmax(masked_output, dim=-1)
    return kl_div(logp, p_masked, reduction="batchmean")



def loadDictionary(lines):
    words = []
    for line in lines:
        words += line
    words = ['<unk>', '<pad>', '<s>', '</s>', '<mask>'] + list(set(words))
    vocab = {j:i for i,j in  enumerate(words)}
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


from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn import metrics

preds = []
y_tests = []

for config in X_tokens:
    raw_text = X_tokens[config][:]
    label = np.array(ground_truth[config][:])
    vocab = loadDictionary(X_tokens[config])
    vocab_size = len(vocab) + 1  # or your known vocab size

    X = vectorize_sequence(raw_text, vocab)
    ref_attn = []
    # print(X_attn[config])
    for i in X_attn[config]:
        ref_attn.append(i+[0]*int(X.shape[1]-len(i)))
    ref_attn = np.array(ref_attn)
    print("Config", config)
    
    for seed in [42, 1975, 5827, 7421, 9876]:
        print("Seed", seed)
        sss = StratifiedShuffleSplit(n_splits=5, test_size=1/5, random_state=seed)

        for i, (train_index, test_index) in enumerate(sss.split(raw_text, label)):
            X_train, X_test, y_train, y_test = X[train_index], X[test_index], label[train_index], label[test_index]
            X_train_attn, X_test_attn = ref_attn[train_index], ref_attn[test_index]
            # print(X_train_attn.shape, X_test_attn.shape)
            print(np.bincount(y_train))
                        # Convert to tensors
            X_train_tensor = torch.tensor(X_train, dtype=torch.long)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float).unsqueeze(1)
            X_train_attn_tensor = torch.tensor(X_train_attn, dtype=torch.long)
            X_test_tensor = torch.tensor(X_test, dtype=torch.long)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float).unsqueeze(1)
            X_test_attn_tensor = torch.tensor(X_test_attn, dtype=torch.long)




            # Dataloaders
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor, X_train_attn_tensor)
            test_dataset = TensorDataset(X_test_tensor, y_test_tensor, X_test_attn_tensor)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)

            lr=0.005
            criterion = nn.BCELoss()
            model = Conv1DModel(1024, len(vocab)+1, X.shape[1])
            model = nn.DataParallel(model) # Wrap the model with DataParallel
            model = model.to('cuda') # Move the model to the GPU
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            optimizer = optim.AdamW(model.parameters(), lr=5e-4)            # Training
            model.train()
            for epoch in range(num_epochs):
                epoch_loss = 0.0
                total_batches = 0

                for xb, yb, rationale_mask in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    rationale_mask = rationale_mask.to(device)
                    optimizer.zero_grad()
                    output_orig = model(xb)
                    loss = criterion(output_orig, yb)

                    # Compute Input×Gradient vs attention BCE alignment loss
                    align_loss = inputxgrad_alignment_loss(
                        model=model,
                        xb=xb,
                        rationale_mask=rationale_mask,   # shape [B, L] 0/1
                        pad_idx=0,
                        normalize=True,
                        second_order=False
                    )

                    lambda_align = 0.2  # tune this weight
                    total_loss = loss + lambda_align * align_loss

                    total_loss.backward()
                    optimizer.step()
                    epoch_loss += total_loss.item()
                    total_batches += 1

                avg_loss = epoch_loss / total_batches
                print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")

            model.eval()
            all_preds, all_y = [], []
            with torch.no_grad():
                for xb, yb, rationale_mask in test_loader:
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

            # print(np.array(tokens_gxi)[0])

            np.save('./nn/'+config+str(seed)+str(i)+'egl.npy',tokens_gxi)


            print(f"Iteration REPORT: \n{metrics.classification_report(all_y, all_preds, digits=4)}")
            print(f"Iteration CONFUSION MATRIX: \n{metrics.confusion_matrix(all_y, all_preds)}")

            preds += all_preds
            y_tests += all_y
            # break
    # Final aggregated report
    print(f"\nFINAL REPORT:\n{metrics.classification_report(y_tests, preds, digits=4)}")
    print(f"FINAL CONFUSION MATRIX:\n{metrics.confusion_matrix(y_tests, preds)}")
