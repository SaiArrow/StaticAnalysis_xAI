import pickle
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Union
from transformers import PreTrainedTokenizer
from sklearn.model_selection import StratifiedShuffleSplit
import clang.cindex
import pandas as pd
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
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from transformers import modeling_utils
if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none","colwise",'rowwise']


quantization_config = BitsAndBytesConfig(
    load_in_4bit = True, # enable 4-bit quantization
    bnb_4bit_quant_type = 'nf4', # information theoretically optimal dtype for normally distributed weights
    bnb_4bit_use_double_quant = True, # quantize quantized weights //insert xzibit meme
    bnb_4bit_compute_dtype = torch.bfloat16 # optimized fp format for ML
)
lora_config = LoraConfig(
    r = 16, # the dimension of the low-rank matrices
    lora_alpha = 8, # scaling factor for LoRA activations vs pre-trained weight activations
    # target_modules=['q_proj', 'v_proj'],
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj"],
    # target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    lora_dropout = 0.05, # dropout probability of the LoRA layers
    bias = 'none', # wether to train bias weights, set to 'none' for attention layers
    task_type = 'SEQ_CLS'
)



files = ['jbmc.csv', 'jayhorn.csv']
X_tokens = {i:[] for i in files}
X_attn = {i:[] for i in files}

ground_truth = {i:[] for i in files}


experiments = pd.read_csv('egl_experiments_h.csv')





class GetOutOfLoop( Exception ):
    pass

with open('relevant.json', 'r') as file:
    syr = json.load(file)

relevant_dict = {}
for i in range(len(syr)):
    relevant_dict[syr[i]['name']] = syr[i]['relevant_lines']

for config in files:
    if 'jbmc' in config:
        MAX_LEN = 1024
    else:
        MAX_LEN = 576

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
                    
            raw_text.append(current_text)

            if len(current_text) != len(temp_matrix):
                print(len(current_text), len(temp_matrix), df['file_path'][i])
                raise ValueError('A very specific bad thing happened.')

            temp_matrix += [0] * int(MAX_LEN - len(temp_matrix))
            relevant_matrix.append(temp_matrix)

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



def loadDictionary(lines):
    words = []
    for line in lines:
        words += line
    words = ['<unk>', '<pad>', '<s>', '</s>', '<mask>'] + list(set(words))
    vocab = {j:i for i,j in  enumerate(words)}
    return vocab



class FixedVocabTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab: Union[Dict[str, int], str], max_len: int = None):
        if isinstance(vocab, str):
            vocab_path = Path(vocab)
            with open(vocab_path, 'r') as f:
                self._token_ids = json.load(f)
        else:
            self._token_ids = vocab
            
        self._id_tokens: Dict[int, str] = {value: key for key, value in self._token_ids.items()}
        super().__init__(max_len=max_len)

        # Initialize special tokens for RoBERTa
        self.unk_token = '<unk>'
        self.pad_token = '<pad>'
        self.cls_token = '<s>'
        self.sep_token = '</s>'
        self.mask_token = '<mask>'
        self.unk_token_id = self._token_ids.get(self.unk_token, 0)
        self.pad_token_id = self._token_ids.get(self.pad_token, 1)
        self.cls_token_id = self._token_ids.get(self.cls_token, 2)
        self.sep_token_id = self._token_ids.get(self.sep_token, 3)
        self.mask_token_id = self._token_ids.get(self.mask_token, 4)

    def _tokenize(self, text: str, **kwargs):
        return text.split('α')

    def _convert_token_to_id(self, token: str) -> int:
        return self._token_ids[token] if token in self._token_ids else self.unk_token_id

    def _convert_id_to_token(self, index: int) -> str:
        return self._id_tokens[index] if index in self._id_tokens else self.unk_token

    def get_vocab(self) -> Dict[str, int]:
        return self._token_ids.copy()

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if filename_prefix is None:
            filename_prefix = ''
        vocab_path = Path(save_directory, filename_prefix + 'vocab.json')
        with open(vocab_path, 'w') as f:
            json.dump(self._token_ids, f)
        return (str(vocab_path),)

    @property
    def vocab_size(self) -> int:
        return len(self._token_ids)


def preprocessing_function(examples):
    # return tokenizer(examples['input'], truncation=True, max_length=MAX_LEN)
    return tokenizer(examples['text'], padding='max_length', max_length=MAX_LEN)


import torch
import torch.nn.functional as F

@torch.no_grad()
def compute_input_gradient_saliency(model, input_ids, attention_mask, labels, normalize=True):
    """
    Computes saliency using input × gradient on embeddings.
    
    Args:
        model: HuggingFace model (should output logits)
        input_ids: [batch_size, seq_len] tensor
        attention_mask: same shape as input_ids
        labels: [batch_size] class indices (0 or 1)
        normalize: if True, normalizes saliency per example to [0, 1]

    Returns:
        saliency: [batch_size, seq_len] tensor of importance scores
    """
    # Clone embeddings and enable grad tracking
    embeddings = model.model.model.embed_tokens(input_ids).detach()
    embeddings.requires_grad_()
    
    # Re-enable gradients temporarily
    with torch.enable_grad():
        outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask)
        logits = outputs.logits
        selected_logits = logits.gather(1, labels.unsqueeze(1)).squeeze()
        selected_logits.sum().backward(retain_graph=True)

    # Input × Gradient saliency
    saliency = (embeddings.grad * embeddings).sum(dim=-1)  # [batch, seq_len]

    # Normalize to [0, 1] per sample
    if normalize:
        min_vals = saliency.min(dim=1, keepdim=True)[0]
        max_vals = saliency.max(dim=1, keepdim=True)[0]
        saliency = (saliency - min_vals) / (max_vals - min_vals + 1e-8)

    return saliency.detach()


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # print(predictions)
    # print(labels)
    # print("Check Here")
    try:
        # it's a classification task, take the argmax
        predictions_processed = np.argmax(predictions, axis=1)
        # print(predictions_processed)

        
        pearson = accuracy_score(predictions_processed, labels)
        print({'accuracy': pearson, 'test_cm':confusion_matrix(predictions_processed, labels)})
        return {'accuracy': pearson}
    except Exception as e:
        print(f"Error in compute_metrics: {e}")
        return {'pearson': None}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def predict_classes(texts, batch_size=1):
    preds = []
    for i in range(0, len(texts), batch_size):
        batch = list(texts[i:i+batch_size])
        enc = tokenizer(
            batch, return_tensors="pt", padding='max_length',  return_token_type_ids=False, max_length=MAX_LEN
        )
        enc.pop("token_type_ids", None)
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        if logits.shape[-1] == 1:
            cls = (torch.sigmoid(logits.squeeze(-1)) >= 0.5).long()
        else:
            cls = logits.argmax(dim=-1)
        preds.append(cls.cpu())
    return torch.cat(preds, dim=0)  # [N]


def gradient_x_input(texts, micro_bs=1, normalize=True):
    """
    texts: list[str]
    Returns:
      tokens_list: list[list[str]]
      scores_list: list[list[float]]  (token-level GxI scores in [0,1] if normalize=True)
      pred_classes: list[int]
    """
    # 1) get predicted class once (no grad)
    pred_classes = predict_classes(texts, batch_size=1)

    tokens_list, scores_list = [], []

    # 2) process in micro-batches with gradients
    for i in range(0, len(texts), micro_bs):
        batch = list(texts[i:i+micro_bs])
        cls_mb = pred_classes[i:i+micro_bs].to(device)

        # tokenize (ids/mask on device)
        enc = tokenizer(
            batch, return_tensors="pt", padding='max_length',  return_token_type_ids=False, max_length=MAX_LEN
        )
        enc.pop("token_type_ids", None)
        input_ids = enc["input_ids"].to(device)          # [B, L]
        attention_mask = enc["attention_mask"].to(device)  # [B, L]

        # 3) get embeddings and enable grad w.r.t. embeddings
        emb_layer = model.get_input_embeddings()
        embeds = emb_layer(input_ids).detach()           # [B, L, H]
        embeds.requires_grad_(True)

        # 4) forward using inputs_embeds (NOT labels) to keep graph
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(inputs_embeds=embeds, attention_mask=attention_mask).logits  # [B, C] or [B,1]
            if logits.shape[-1] == 1:
                # single-logit binary → select signed logit (+ for class 1, - for class 0)
                logit = logits.squeeze(-1)                          # [B]
                selected = torch.where(cls_mb == 1, logit, -logit)  # [B]
            else:
                selected = logits.gather(1, cls_mb.unsqueeze(1)).squeeze(1)  # [B]

            # 5) backprop once per micro-batch
            selected.sum().backward(retain_graph=False)

        # 6) Gradient × Input on embeddings
        # grads shape: [B, L, H]
        grads = embeds.grad
        gx = (embeds * grads).sum(dim=-1).abs()  # [B, L] magnitude

        # mask out padding
        gx = gx * attention_mask

        # optional: normalize per example to [0,1]
        if normalize:
            minv = gx.min(dim=1, keepdim=True).values
            maxv = gx.max(dim=1, keepdim=True).values
            gx = (gx - minv) / (maxv - minv + 1e-8)

        # 7) collect tokens & scores
        ids_cpu = input_ids.detach().cpu()
        gx_cpu = gx.detach().cpu()
        for j in range(ids_cpu.size(0)):
            toks = tokenizer.convert_ids_to_tokens(ids_cpu[j].tolist())
            tokens_list.append(toks)
            scores_list.append(gx_cpu[j].tolist())

        # clean up to reduce peak memory
        del embeds, grads, logits, selected, gx
        torch.cuda.empty_cache()

    return tokens_list, scores_list, pred_classes.tolist()




class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.args.device)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False,  num_items_in_batch=None):
        # Extract labels and convert them to long type for cross_entropy
        labels = inputs.pop("labels").long()
        rationale_mask = inputs["rationale_mask"].float()
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.get('logits')
        if self.class_weights is not None:
            loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        else:
            loss = F.cross_entropy(logits, labels)

        saliency = compute_input_gradient_saliency(model, input_ids, attention_mask, labels)
        masked_saliency = saliency * attention_mask
        masked_rationale = rationale_mask * attention_mask
        saliency_loss = F.binary_cross_entropy(masked_saliency, masked_rationale.float())

        λ = 0.25
        total_loss = loss + λ * saliency_loss


        return (total_loss, outputs) if return_outputs else total_loss


variations = [
["deepseek-ai/deepseek-coder-1.3b-base", 5e-5],
["meta-llama/CodeLlama-7b-hf", 5e-5],
["microsoft/wavecoder-ultra-6.7b", 5e-5],
["google/codegemma-2b", 5e-5],
["stabilityai/stable-code-3b", 5e-5],
["Qwen/Qwen2.5-Coder-3B", 5e-5],
["bigcode/starcoder2-3b", 5e-5]
]


for model_name, learning_rate in variations:
    print("EGL Model Name", model_name, learning_rate)

    for config in X_tokens:
        if 'jbmc' in config:
            MAX_LEN = 1024
        else:
            MAX_LEN = 576

        print("MAX_LEN", MAX_LEN)
        raw_text = np.array(['α'.join(i) for i in X_tokens[config]])
        label = np.array(ground_truth[config])
        ref_attn = np.array(X_attn[config])
        print(ref_attn.shape)
        

        vocab = loadDictionary(X_tokens[config])
        # You can either pass the custom vocab dictionary or the path to the vocab file
        tokenizer = FixedVocabTokenizer(vocab, max_len=MAX_LEN)



        for seed in [42, 1975, 5827, 7421, 9876]:
            print("Seed", seed)
            sss = StratifiedShuffleSplit(n_splits=5, test_size=1/5, random_state=seed)
            flag = False
            if model_name in list(experiments['model']):
                temp = experiments[(experiments['model'] == model_name) & (experiments['seed'] == seed) & (experiments['config'] == config)]
                flag = True
                


            for i, (train_index, test_index) in enumerate(sss.split(raw_text, label)):
                if flag:
                    if i in list(temp['iteration']):
                        print("Skipping ", config, i)
                        continue

                X_train, X_test, y_train, y_test = raw_text[train_index], raw_text[test_index], label[train_index], label[test_index]
                X_train_attn, X_test_attn = ref_attn[train_index], ref_attn[test_index]
                print(X_train_attn.shape, X_test_attn.shape)
                print(np.bincount(y_train))
                X_train_df = pd.DataFrame({"text":X_train, "label":y_train, "rel": train_index, "rationale_mask":list(X_train_attn)})
                X_test_df = pd.DataFrame({"text":X_test, "label":y_test, "rel": test_index, "rationale_mask":list(X_test_attn)})
                datasets_train_test = DatasetDict({
                                                    "train": Dataset.from_pandas(X_train_df),
                                                    "test": Dataset.from_pandas(X_test_df)
                                                    })


                def preprocessing_function(examples):
                    # return tokenizer(examples['input'], truncation=True, max_length=MAX_LEN)
                    return tokenizer(examples['text'], padding='max_length', max_length=MAX_LEN)


                tokenized_dataset = datasets_train_test.map(preprocessing_function, batched=True)
                tokenized_dataset.set_format("torch",columns=["input_ids", "attention_mask", "label", "rel", "rationale_mask"])
                data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

                class_weights=(1/X_train_df.label.value_counts(normalize=True).sort_index()).tolist()
                class_weights=torch.tensor(class_weights)
                class_weights=class_weights/class_weights.sum()
                print(class_weights)

                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    num_labels=2,
                    # output_attentions=True,
                    # attn_implementation="sdpa",
                    cache_dir="new_cache_dir/"
                )

                model = prepare_model_for_kbit_training(model)
                model = get_peft_model(model, lora_config)
                model.config.pad_token_id = tokenizer._convert_token_to_id(tokenizer.pad_token)
                model.config.use_cache = False
                model.config.pretraining_tp = 1


                training_args = TrainingArguments(
                    output_dir = model_name+'_egl_sequence_classification_'+str(learning_rate)+str(i),
                    learning_rate = learning_rate,
                    per_device_train_batch_size = 8,
                    per_device_eval_batch_size = 8,
                    num_train_epochs = 40,
                    remove_unused_columns=False,
                    save_total_limit=1,
                    save_steps=0,
                    # weight_decay = 0.1,
                    metric_for_best_model='eval_accuracy',
                    eval_strategy = 'epoch',
                    save_strategy = 'epoch',
                    load_best_model_at_end = True
                )

                trainer = CustomTrainer(
                    model = model,
                    args = training_args,
                    train_dataset = tokenized_dataset['train'],
                    eval_dataset = tokenized_dataset['test'],
                    tokenizer = tokenizer,
                    data_collator = data_collator,
                    compute_metrics = compute_metrics,
                    # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
                    # class_weights=class_weights,
                )

                print(tokenized_dataset)
                train_result = trainer.train()

                toks, scores, preds = gradient_x_input(X_test, micro_bs=1, normalize=True)
                np.save("./saved_models/"+model_name+'_egl_sequence_classification_'+str(config)+str(seed)+str(i)+'test.npy',np.array(scores))


                metrics = train_result.metrics
                max_train_samples = len(train_index)
                metrics["train_samples"] = min(max_train_samples, len(train_index))
                trainer.log_metrics("train", metrics)
                trainer.save_metrics("train", metrics)
                trainer.save_state()
                print(trainer.evaluate(tokenized_dataset['train']), trainer.evaluate(tokenized_dataset['test']))
                data = {
                "model":model_name,
                "config":config,
                "seed": seed,
                "iteration":i,
                "eval_acc": [trainer.evaluate()['eval_accuracy']],
                # "confusion_matrix": [confusion_matrix(predictions_processed, labels)]
                }

                df = pd.DataFrame(data)
                with open('egl_experiments_h.csv', 'a') as f:
                    df.to_csv(f, header=False)

                
