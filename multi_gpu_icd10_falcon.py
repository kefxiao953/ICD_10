# conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
# pip install datasets
# pip install peft
# pip install transformers
# pip install bitsandbytes-cuda110 bitsandbytes
# pip install accelerate
# you may need to restart your kernel for your notebook if you installed all of the above

import os
import random
import functools
import csv
import time
import numpy as np
import pandas as pd # added for ICD10 data
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from skmultilearn.model_selection import iterative_train_test_split
from datasets import Dataset, DatasetDict
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)

start = time.time()

"""
Modify the following only for quick testing
"""

# epochs
num_train_epochs = 10

# model name
model_name = 'tiiuae/falcon-rw-1b'

# name of where trained model will be housed
output_dir = 'falcon_1B_icd10_10_epochs'

# target_modules
target_modules = ["query_key_value"] # for falcon
# ['q_proj', 'k_proj', 'v_proj', 'o_proj'], # for mistral models

# batch siz; you may need to make this smaller to run
batch_size = 3


"""
End Block

Run this script in the terminal with: 
accelerate launch multi_gpu_icd10_falcon.py
"""

### ADDED FOR MULTI GPU ###
from accelerate import Accelerator
device_index = Accelerator().process_index
device_map = {"": device_index}


def tokenize_examples(examples, tokenizer):
    tokenized_inputs = tokenizer(examples['text'])
    tokenized_inputs['labels'] = examples['labels']
    return tokenized_inputs


# define custom batch preprocessor
def collate_fn(batch, tokenizer):
    dict_keys = ['input_ids', 'attention_mask', 'labels']
    d = {k: [dic[k] for dic in batch] for k in dict_keys}
    d['input_ids'] = torch.nn.utils.rnn.pad_sequence(
        d['input_ids'], batch_first=True, padding_value=tokenizer.pad_token_id
    )
    d['attention_mask'] = torch.nn.utils.rnn.pad_sequence(
        d['attention_mask'], batch_first=True, padding_value=0
    )
    d['labels'] = torch.stack(d['labels'])
    return d


# define which metrics to compute for evaluation
def compute_metrics(p):
    predictions, labels = p
    f1_micro = f1_score(labels, predictions > 0, average = 'micro')
    f1_macro = f1_score(labels, predictions > 0, average = 'macro')
    f1_weighted = f1_score(labels, predictions > 0, average = 'weighted')
    return {
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }

# create custom trainer class to be able to pass label weights and calculate mutilabel loss
class CustomTrainer(Trainer):

    def __init__(self, label_weights, **kwargs):
        super().__init__(**kwargs)
        self.label_weights = label_weights
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # compute custom loss
        loss = F.binary_cross_entropy_with_logits(logits, labels.to(torch.float32), pos_weight=self.label_weights)
        return (loss, outputs) if return_outputs else loss


x_train = pd.read_csv("xtrain_5.csv")
x_val = pd.read_csv("xval_5.csv")
y_train = pd.read_csv("ytrain_5.csv")
y_val = pd.read_csv("yval_5.csv")
labels = np.array(y_train, dtype=int)
label_weights = 1 - labels.sum(axis=0) / labels.sum()

# create hf dataset
ds = DatasetDict({
    'train': Dataset.from_dict({'text': x_train["text"], 'labels': np.array(y_train, dtype=int)}),
    'val': Dataset.from_dict({'text': x_val["text"], 'labels': np.array(y_val, dtype=int)})
})

### added for ICD10 text ###
max_length = 1024 # maximum for falcon according to error from terminal (huggingface claims 2048)

# preprocess dataset with tokenizer
def tokenize_examples(examples, tokenizer):
    tokenized_inputs = tokenizer(examples['text'], max_length=max_length, truncation=True) ### added for ICD10 text ###
    tokenized_inputs['labels'] = examples['labels']
    return tokenized_inputs

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenized_ds = ds.map(functools.partial(tokenize_examples, tokenizer=tokenizer), batched=True)
tokenized_ds = tokenized_ds.with_format('torch')

# qunatization config
quantization_config = BitsAndBytesConfig(
    load_in_4bit = True, # enable 4-bit quantization
    bnb_4bit_quant_type = 'nf4', # information theoretically optimal dtype for normally distributed weights
    bnb_4bit_use_double_quant = True, # quantize quantized weights //insert xzibit meme
    bnb_4bit_compute_dtype = torch.bfloat16 # optimized fp format for ML
)

# lora config
lora_config = LoraConfig(
    r = 16, # the dimension of the low-rank matrices
    lora_alpha = 8, # scaling factor for LoRA activations vs pre-trained weight activations
    target_modules = target_modules,
    lora_dropout = 0.05, # dropout probability of the LoRA layers
    bias = 'none', # wether to train bias weights, set to 'none' for attention layers
    task_type = 'SEQ_CLS'
)

# load model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    device_map=device_map, ### ADDED FOR MULTI GPU ###
    quantization_config=quantization_config,
    num_labels=labels.shape[1]
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.config.pad_token_id = tokenizer.pad_token_id

### ADDED FOR MULTI GPU ###
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})

# define training args
training_args = TrainingArguments(
    output_dir = output_dir,
    learning_rate = 1e-4,
    per_device_train_batch_size = batch_size, # you may need to make this smaller to run
    per_device_eval_batch_size = batch_size,
    num_train_epochs = num_train_epochs,
    weight_decay = 0.01,
    evaluation_strategy = 'epoch',
    save_strategy = 'epoch',
    load_best_model_at_end = True,
)

# train
trainer = CustomTrainer(
    model = model,
    args = training_args,
    train_dataset = tokenized_ds['train'],
    eval_dataset = tokenized_ds['val'],
    tokenizer = tokenizer,
    data_collator = functools.partial(collate_fn, tokenizer=tokenizer),
    compute_metrics = compute_metrics,
    label_weights = torch.tensor(label_weights, device=model.device),
)

timediff = time.time() - start
print(f"\n\n\n\n\n\n\n\n Took {timediff} to begin the training process\n\n\n\n\n\n\n\n")

start = time.time()
trainer.train()
timediff = time.time() - start
print(f"\n\n\n\n\n\n\n\n Took {timediff} to train this model\n\n\n\n\n\n\n\n")

# save model
peft_model_id = output_dir
trainer.model.save_pretrained(peft_model_id)
tokenizer.save_pretrained(peft_model_id)
