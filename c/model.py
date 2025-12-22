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



with open('all_labels.pkl', 'rb') as handle:
    labels = pickle.load(handle)

with open('slice_lines.pkl', 'rb') as handle:
    slice_imp = pickle.load(handle)


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




MAX_LEN = 3072
print("MAX_LEN", MAX_LEN)


experiments = pd.read_csv('experiments.csv')


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

def preprocessing_function(examples):
    # return tokenizer(examples['input'], truncation=True, max_length=MAX_LEN)
    return tokenizer(examples['text'], padding='max_length', max_length=MAX_LEN)


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

        # Forward pass
        outputs = model(**inputs)

        # Extract logits assuming they are directly outputted by the model
        logits = outputs.get('logits')

        # Compute custom loss with class weights for imbalanced data handling
        if self.class_weights is not None:
            loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        else:
            loss = F.cross_entropy(logits, labels)

        return (loss, outputs) if return_outputs else loss


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
    print("Model Name", model_name, learning_rate)

    for config in X_tokens:
        # if config == '47':
        #     print("Skip 47")
        #     continue
        raw_text = np.array(['α'.join(i) for i in X_tokens[config][:]])
        label = np.array(ground_truth[config][:])
        

        vocab = loadDictionary(X_tokens[config])
        # You can either pass the custom vocab dictionary or the path to the vocab file
        tokenizer = FixedVocabTokenizer(vocab, max_len=MAX_LEN)



        for seed in [42, 1975, 5827, 7421, 9876]:
            print("Seed", seed)
            sss = StratifiedShuffleSplit(n_splits=5, test_size=1/5, random_state=seed)
            flag = False
            if model_name in list(experiments['model']):
                temp = experiments[(experiments['model'] == model_name) & (experiments['seed'] == seed) & (experiments['config'] == int(config))]
                flag = True
                


            for i, (train_index, test_index) in enumerate(sss.split(raw_text, label)):
                if flag:
                    if i in list(temp['iteration']):
                        print("Skipping ", config, i)
                        continue

                X_train, X_test, y_train, y_test = raw_text[train_index], raw_text[test_index], label[train_index], label[test_index]
                print(np.bincount(y_train))
                X_train_df = pd.DataFrame({"text":X_train, "label":y_train, "rel": train_index})
                X_test_df = pd.DataFrame({"text":X_test, "label":y_test, "rel": test_index})
                datasets_train_test = DatasetDict({
                                                    "train": Dataset.from_pandas(X_train_df),
                                                    "test": Dataset.from_pandas(X_test_df)
                                                    })


                tokenized_dataset = datasets_train_test.map(preprocessing_function, batched=True)
                tokenized_dataset.set_format("torch",columns=["input_ids", "attention_mask", "label", "rel"])
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
                    output_dir = model_name+'_sequence_classification_'+str(i),
                    learning_rate = learning_rate,
                    per_device_train_batch_size = 8,
                    per_device_eval_batch_size = 8,
                    num_train_epochs = 10,
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
                    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
                    # class_weights=class_weights,
                )

                print(tokenized_dataset)
                train_result = trainer.train()

                toks, scores, preds = gradient_x_input(X_test, micro_bs=1, normalize=True)
                np.save("./saved_models/"+model_name+'_base_sequence_classification_'+str(config)+str(seed)+str(i)+'test.npy',np.array(scores))

                metrics = train_result.metrics
                max_train_samples = len(train_index)
                metrics["train_samples"] = min(max_train_samples, len(train_index))
                trainer.log_metrics("train", metrics)
                trainer.save_metrics("train", metrics)
                trainer.save_state()
                # trainer.save_model(model_name+'_sequence_classification_'+str(i))

                data = {
                "model":model_name,
                "config":config,
                "seed": seed,
                "iteration":i,
                "eval_acc": [trainer.evaluate()['eval_accuracy']],
                # "confusion_matrix": [confusion_matrix(predictions_processed, labels)]
                }

                df = pd.DataFrame(data)
                with open('experiments.csv', 'a') as f:
                    df.to_csv(f, header=False)

                
