import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
import wandb

import math
import re
import string
import pprint
import os
import json

from copy import deepcopy
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModel, get_scheduler
from tqdm.auto import tqdm

from torchcrf import CRF
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.optim import AdamW
from transformers.modeling_outputs import  TokenClassifierOutput
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from datasets import DatasetDict, concatenate_datasets, load_dataset
from evaluate import load
from utils import choices, collate_fn_robert, collate_fn_xlm



# Create Lightning DataModule for body cleaning
class GeneralisticDataModule(pl.LightningDataModule):
    def __init__(self, model_name: str, batch_size: int, num_workers: int, verbose: bool, type: str = 'train'):
        super().__init__()
        self.verbose = verbose
        self.batch_size = batch_size
        self.model = model_name
        self.num_workers = num_workers
        self.type = type
        self.already_called = False

    # Here the tokenization will be agnostic to the entities, the splits will have a 128 token overlap
    def prepare_data(self):
        def count_weights(dataset, split):
            if split != 'train':
                return
            else:
                all_choices = {'auto_generated':{0:0, 1:0}, 'needs_action':{0:0, 1:0}, 'spam':{0:0, 1:0}, 'business':{0:0, 1:0}, 'writing_style':{0:0, 1:0, 2:0}} 
                for example in dataset:
                    example_choices = [0] * len(choices)
                    for choice in example['annotation']['choices']:
                        example_choices[choices[choice['name']][0]] = choices[choice['name']][1][choice['value']]
                    all_choices['auto_generated'][example_choices[0]] += 1
                    all_choices['needs_action'][example_choices[1]] += 1
                    all_choices['spam'][example_choices[2]] += 1
                    all_choices['business'][example_choices[3]] += 1
                    all_choices['writing_style'][example_choices[4]] += 1

                
                all_weights = {}
                for key in all_choices:
                    max_value = max(all_choices[key].values())
                    if len(all_choices[key]) > 2:
                        all_weights[key] = [max_value/all_choices[key][idx] for idx in range(len(all_choices[key]))]
                    else:
                        all_weights[key] = [all_choices[key][0]/ all_choices[key][1]]
                        
                self.weights_choices = deepcopy(all_weights)
                
                    
        def tokenize_dataset(dataset, tokenizer):
            def tokenize_and_align_labels(examples):
                    tokenized_inputs = tokenizer(examples['token_classification']["tokens"], is_split_into_words=True)
                    example_choices = [0] * len(choices)
                    for choice in examples['annotation']['choices']:
                        example_choices[choices[choice['name']][0]] = choices[choice['name']][1][choice['value']]
                    tokenized_inputs['labels_choices'] = example_choices
                    return tokenized_inputs

            def split_and_append(dataset, example):
                    if len(example['input_ids']) <= 512:
                        working_example = {'dataset_id': example['dataset_id'], 'personal_id': 0, 'input_ids':example['input_ids'], 'attention_mask':example['attention_mask'], 'labels_choices':example['labels_choices']}
                        dataset.append(working_example)
                    else:
                        # Split into 512 tokens splits that have an 128 token overlap
                        i = 0
                        repeat = True
                        while repeat:
                            working_example = {'dataset_id': example['dataset_id'], 'personal_id': i, 'input_ids':[], 'attention_mask':[], 'labels_choices':example['labels_choices']} # The tokens here for input_ids and attention are chosen because of observed roberta tokens
                            if i == 0:
                                working_example['input_ids'] = example['input_ids'][:511] + [4]
                                working_example['attention_mask'] = example['attention_mask'][:511] + [1]
                            else:
                                if i * 384 + 511 >= len(example['input_ids']):
                                    working_example['input_ids'] = [3] + example['input_ids'][i * 384: i * 384 + 511]
                                    working_example['attention_mask'] = [1] + example['attention_mask'][i * 384: i * 384 + 511]
                                    repeat = False
                                else:
                                    working_example['input_ids'] = [3] + example['input_ids'][i * 384: i * 384 + 510] + [4]
                                    working_example['attention_mask'] = [1] + example['attention_mask'][i * 384: i * 384 + 510] + [1]
                            
                            dataset.append(working_example)
                            i += 1
                    return dataset
            
            new_dataset = []
            for example in dataset:
                tokenized_example = tokenize_and_align_labels(example)
                tokenized_example['dataset_id'] = example['id']
                split_and_append(new_dataset, tokenized_example)
            return new_dataset
        
        if self.already_called:
            return
        
        else: 
            self.already_called = True
            if self.verbose:
                print(f"Preparing Data...")
            tokenizer = AutoTokenizer.from_pretrained(self.model)
            for file in ['train', 'val', 'test']:
                with open(f'./data/{file}.json') as f:
                    data = json.load(f)

                tokenized_dataset = tokenize_dataset(data, tokenizer)
                count_weights(data, file)
                with open(f'./data/{file}_tokenized_RoBERT.json', 'w') as f:
                    for example in tokenized_dataset:
                        json.dump(example, f)
                        f.write('\n')
            
            del tokenizer, data, tokenized_dataset

    def setup(self, stage=None):
            if self.verbose:
                print(f"Loading dataset...")
            dataset = load_dataset('json', data_files={'train': f'./data/train_tokenized_RoBERT.json', 'validation': f'./data/val_tokenized_RoBERT.json', 'test': f'./data/test_tokenized_RoBERT.json'})
            
            self.train_dataset = GeneralisticDataset(dataset['train'], verbose=self.verbose)

            self.validation_dataset = GeneralisticDataset(dataset=dataset['validation'], verbose=self.verbose)
            
            self.test_dataset = GeneralisticDataset(dataset=dataset['test'], verbose=self.verbose)
            
    def cleanup(self, stage=None):
        if self.verbose:
            print(f"Cleaning up dataset...")
        for file in ['train', 'val', 'test']:
            os.remove(f'./data/{file}_tokenized_RoBERT.json')
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
            sampler=RandomSampler(self.train_dataset), num_workers=self.num_workers, collate_fn=collate_fn_robert)
    
    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size,
            sampler=SequentialSampler(self.validation_dataset), num_workers=self.num_workers, collate_fn=collate_fn_robert)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
            sampler=SequentialSampler(self.test_dataset), num_workers=self.num_workers, collate_fn=collate_fn_robert)

# Create Dataset for body cleaning
class GeneralisticDataset(Dataset):
    def __init__(self, dataset: Dataset, verbose: bool = False):
        self.dataset = dataset
        self.verbose = verbose

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        return_dict = {
            'dataset_id': example['dataset_id'],
            'personal_id': example['personal_id'],
            'input_ids': torch.LongTensor(example['input_ids']),
            'attention_mask': torch.LongTensor(example['attention_mask']),
            'labels_choices': torch.LongTensor(example['labels_choices'])
        }
        return return_dict 

# Create Model
class Heads(torch.nn.Module):
    def __init__(self, in_features: int):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()

        self.auto_generated_head = torch.nn.Linear(in_features, 1, bias=True)
        self.auto_generated_dropout = torch.nn.Dropout(0.1)

        self.needs_action_head = torch.nn.Linear(in_features, 1, bias=True)
        self.needs_action_dropout = torch.nn.Dropout(0.1)

        self.spam_head = torch.nn.Linear(in_features, 1, bias=True)
        self.spam_dropout = torch.nn.Dropout(0.1)

        self.business_head = torch.nn.Linear(in_features, 1, bias=True)
        self.business_dropout = torch.nn.Dropout(0.1)

        self.writing_style_head = torch.nn.Linear(in_features, 3, bias=True)
        self.writing_style_dropout = torch.nn.Dropout(0.1)



    def forward(self, x, attention_mask):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        class_x =  x.sum(axis=1) / attention_mask.sum(axis=-1).unsqueeze(-1)
       
        auto_generated = self.auto_generated_head(self.auto_generated_dropout(class_x))
        needs_action = self.needs_action_head(self.needs_action_dropout(class_x))
        spam = self.spam_head(self.spam_dropout(class_x))
        business = self.business_head(self.business_dropout(class_x))
        writing_style = self.writing_style_head(self.writing_style_dropout(class_x))

        return [auto_generated, needs_action, spam, business, writing_style]
        

class Model(pl.LightningModule):
    def __init__(self, model_name: str, n_training_steps: int, n_warmup_steps: int,
            optimizer: str, scheduler: str, learning_rate: float, extra_hyperparams: str):
        super().__init__()
        #self.trainer.datamodule to access atributes from DataModule
        self.model = AutoModel.from_pretrained(model_name)
        self.heads = Heads(self.model.config.hidden_size)
        # self.model.resize_token_embeddings(len(self.tokenizer))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.criterion = torch.nn.CrossEntropyLoss()
        self.binary_criterion = torch.nn.BCEWithLogitsLoss()

        self.optimizer_name = optimizer
        self.scheduler_name = scheduler
        self.learning_rate = learning_rate
        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask, labels):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        [auto_generated, needs_action, spam, business, writing_style] = self.heads(output.last_hidden_state, attention_mask)
        if labels:
            auto_generated_loss = self.binary_criterion(auto_generated, labels['labels_choices'][:,0].unsqueeze(-1).float())
            needs_action_loss = self.binary_criterion(needs_action, labels['labels_choices'][:,1].unsqueeze(-1).float())
            spam_loss = self.binary_criterion(spam, labels['labels_choices'][:,2].unsqueeze(-1).float())
            business_loss = self.binary_criterion(business, labels['labels_choices'][:,3].unsqueeze(-1).float())
            writing_style_loss = self.criterion(writing_style, labels['labels_choices'][:,4])
            
            loss = auto_generated_loss + needs_action_loss + spam_loss + business_loss + writing_style_loss
            loss_dict = {'loss': loss, 'auto_generated_loss': auto_generated_loss, 'needs_action_loss': needs_action_loss, 'spam_loss': spam_loss, 'business_loss': business_loss, 'writing_style_loss': writing_style_loss}
            return TokenClassifierOutput(loss=loss_dict, logits=[auto_generated, needs_action, spam, business, writing_style], )
        else:
            return TokenClassifierOutput(logits=[auto_generated, needs_action, spam, business, writing_style])

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
        
        outputs = self(input_ids, attention_mask, labels)
        loss = outputs.loss['loss']
        logs = {'train_loss': loss, 'lr': self.trainer.optimizers[0].param_groups[0]['lr'],
                'train_auto_generated_loss': outputs.loss['auto_generated_loss'],
                'train_needs_action_loss': outputs.loss['needs_action_loss'], 'train_spam_loss': outputs.loss['spam_loss'],
                'train_business_loss': outputs.loss['business_loss'], 'train_writing_style_loss': outputs.loss['writing_style_loss']}

        self.log_dict(logs, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']

        outputs = self(input_ids, attention_mask, labels)
        loss = outputs.loss['loss']
        logs = {'val_loss': loss,
                'val_auto_generated_loss': outputs.loss['auto_generated_loss'],
                'val_needs_action_loss': outputs.loss['needs_action_loss'], 'val_spam_loss': outputs.loss['spam_loss'],
                'val_business_loss': outputs.loss['business_loss'], 'val_writing_style_loss': outputs.loss['writing_style_loss']}

        self.log_dict(logs, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return {'loss': loss}
    
    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']

        outputs = self(input_ids, attention_mask, labels)
        loss = outputs.loss['loss']
        logs = {'val_loss': loss, 'lr': self.trainer.optimizers[0].param_groups[0]['lr'],
                'val_auto_generated_loss': outputs.loss['auto_generated_loss'],
                'val_needs_action_loss': outputs.loss['needs_action_loss'], 'val_spam_loss': outputs.loss['spam_loss'],
                'val_business_loss': outputs.loss['business_loss'], 'val_writing_style_loss': outputs.loss['writing_style_loss']}

        self.log_dict(logs, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return {'loss': loss}
    
    def configure_optimizers(self):
        if self.optimizer_name == 'AdamW':
            optimizer = AdamW(self.parameters(), lr=self.learning_rate)

        if self.scheduler_name == 'constant':
            scheduler = get_scheduler(name=self.scheduler_name, optimizer=optimizer)
        elif self.scheduler_name in ('linear', 'cosine', 'cosine_with_restarts'):
            scheduler = get_scheduler(name=self.scheduler_name, optimizer=optimizer, num_warmup_steps=self.n_warmup_steps,
                    num_training_steps=self.n_training_steps)
        
        return {'optimizer':optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval':'step'}}



class WeightsModel(pl.LightningModule):
    def __init__(self, model_name: str, n_training_steps: int, n_warmup_steps: int,
            optimizer: str, scheduler: str, learning_rate: float, extra_hyperparams: str,
            datamodule: pl.LightningDataModule):
        super().__init__()
        #self.trainer.datamodule to access atributes from DataModule
        self.model = AutoModel.from_pretrained(model_name)
        self.heads = Heads(self.model.config.hidden_size)
        # self.model.resize_token_embeddings(len(self.tokenizer))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps

        self.binary_criterion = torch.nn.BCEWithLogitsLoss()
        self.writing_style_criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(datamodule.weights_choices['writing_style']))

        self.optimizer_name = optimizer
        self.scheduler_name = scheduler
        self.learning_rate = learning_rate
        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask, labels):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        [auto_generated, needs_action, spam, business, writing_style] = self.heads(output.last_hidden_state, attention_mask)
        if labels:
            auto_generated_loss = self.binary_criterion(auto_generated, labels['labels_choices'][:,0].unsqueeze(-1).float())
            needs_action_loss = self.binary_criterion(needs_action, labels['labels_choices'][:,1].unsqueeze(-1).float())
            spam_loss = self.binary_criterion(spam, labels['labels_choices'][:,2].unsqueeze(-1).float())
            business_loss = self.binary_criterion(business, labels['labels_choices'][:,3].unsqueeze(-1).float())
            writing_style_loss = self.writing_style_criterion(writing_style, labels['labels_choices'][:,4])
            
            loss = auto_generated_loss + needs_action_loss + spam_loss + business_loss + writing_style_loss
            loss_dict = {'loss': loss, 'auto_generated_loss': auto_generated_loss, 'needs_action_loss': needs_action_loss, 'spam_loss': spam_loss, 'business_loss': business_loss, 'writing_style_loss': writing_style_loss}
            return TokenClassifierOutput(loss=loss_dict, logits=[auto_generated, needs_action, spam, business, writing_style], )
        else:
            return TokenClassifierOutput(logits=[auto_generated, needs_action, spam, business, writing_style])

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
        
        outputs = self(input_ids, attention_mask, labels)
        loss = outputs.loss['loss']
        logs = {'train_loss': loss, 'lr': self.trainer.optimizers[0].param_groups[0]['lr'],
                'train_auto_generated_loss': outputs.loss['auto_generated_loss'],
                'train_needs_action_loss': outputs.loss['needs_action_loss'], 'train_spam_loss': outputs.loss['spam_loss'],
                'train_business_loss': outputs.loss['business_loss'], 'train_writing_style_loss': outputs.loss['writing_style_loss']}

        self.log_dict(logs, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']

        outputs = self(input_ids, attention_mask, labels)
        loss = outputs.loss['loss']
        logs = {'val_loss': loss,
                'val_auto_generated_loss': outputs.loss['auto_generated_loss'],
                'val_needs_action_loss': outputs.loss['needs_action_loss'], 'val_spam_loss': outputs.loss['spam_loss'],
                'val_business_loss': outputs.loss['business_loss'], 'val_writing_style_loss': outputs.loss['writing_style_loss']}

        self.log_dict(logs, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return {'loss': loss}
    
    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']

        outputs = self(input_ids, attention_mask, labels)
        loss = outputs.loss['loss']
        logs = {'val_loss': loss, 'lr': self.trainer.optimizers[0].param_groups[0]['lr'],
                'val_auto_generated_loss': outputs.loss['auto_generated_loss'],
                'val_needs_action_loss': outputs.loss['needs_action_loss'], 'val_spam_loss': outputs.loss['spam_loss'],
                'val_business_loss': outputs.loss['business_loss'], 'val_writing_style_loss': outputs.loss['writing_style_loss']}

        self.log_dict(logs, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return {'loss': loss}
    
    def configure_optimizers(self):
        if self.optimizer_name == 'AdamW':
            optimizer = AdamW(self.parameters(), lr=self.learning_rate)

        if self.scheduler_name == 'constant':
            scheduler = get_scheduler(name=self.scheduler_name, optimizer=optimizer)
        elif self.scheduler_name in ('linear', 'cosine', 'cosine_with_restarts'):
            scheduler = get_scheduler(name=self.scheduler_name, optimizer=optimizer, num_warmup_steps=self.n_warmup_steps,
                    num_training_steps=self.n_training_steps)
        
        return {'optimizer':optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval':'step'}}



def train(MODEL_NAME: str, TRAIN_EPOCHS: int, BATCH_SIZE: int, GRADIENT_ACCUMULATION_STEPS: int,
        NUM_WORKERS: int, NUM_GPUs: int, WARMUP_PROCENTAGE: int, OPTIMIZER: str, SCHEDULER: str, 
        LEARNING_RATE: float, EARLY_STOPPING: int, STRATEGY: str, ACCELERATOR: str,
        RUN_NAME: str, CHECKPOINT_PATH: str, RESUME_TRAINING: bool, CRF_REDUCTION: str, verbose: bool):
    
    def _hparams_logger(**kwargs):
        """This parameters wil appear in the logs in the config.yaml file"""
        return kwargs

    try:
        if verbose:
            print(f'Huggingface Model used: {MODEL_NAME}')
            print(f'Training for {TRAIN_EPOCHS} epochs on {NUM_GPUs} GPUs')
            print(f'Batch size: {BATCH_SIZE}')
            print(f'Gradient accumulation steps: {GRADIENT_ACCUMULATION_STEPS}')

        
        data = GeneralisticDataModule(model_name=MODEL_NAME, batch_size= BATCH_SIZE, num_workers=NUM_WORKERS, verbose=verbose, type='train')
        data.prepare_data()
        data.setup()

        TRUE_NUM_GPUs = NUM_GPUs if type(NUM_GPUs) is int else len(NUM_GPUs)
        total_training_steps = math.ceil(len(data.train_dataset) // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * TRUE_NUM_GPUs)) * TRAIN_EPOCHS
        total_warmup_steps = math.ceil(total_training_steps // 100) * WARMUP_PROCENTAGE
        
        if verbose: 
            print(f'Total training steps: {total_training_steps}, out of which {WARMUP_PROCENTAGE}% are warmup steps')
            print("Creating LightningModule")
            print(f'Optimizer used: {OPTIMIZER}')
            print(f'Scheduler used: {SCHEDULER}')
            print(f'Learning Rate used: {LEARNING_RATE}')

        logger_hyperparams = _hparams_logger(MODEL_NAME=MODEL_NAME, TRAIN_EPOCHS=TRAIN_EPOCHS, BATCH_SIZE=BATCH_SIZE,
                    GRADIENT_ACCUMULATION_STEPS=GRADIENT_ACCUMULATION_STEPS, NUM_GPUs=NUM_GPUs,
                    WARMUP_PROCENTAGE=WARMUP_PROCENTAGE, OPTIMIZER=OPTIMIZER, SCHEDULER=SCHEDULER, 
                    LEARNING_RATE=LEARNING_RATE, total_training_steps=total_training_steps, 
                    total_warmup_steps=total_warmup_steps)

        if RESUME_TRAINING:
            model = Model.load_from_checkpoint(checkpoint_path=CHECKPOINT_PATH)
        else:
            # model = Model(model_name=MODEL_NAME, n_training_steps=total_training_steps, n_warmup_steps=total_warmup_steps,
            #     optimizer=OPTIMIZER, scheduler=SCHEDULER, learning_rate=LEARNING_RATE, extra_hyperparams=logger_hyperparams)
            model = WeightsModel(model_name=MODEL_NAME, n_training_steps=total_training_steps, n_warmup_steps=total_warmup_steps,
                optimizer=OPTIMIZER, scheduler=SCHEDULER, learning_rate=LEARNING_RATE, extra_hyperparams=logger_hyperparams,
                datamodule=data)

        callbacks = []
        
        logger = WandbLogger(project='Email Classification', name=RUN_NAME)
        if EARLY_STOPPING:
            early_stopping_callback = EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING, mode='min')
            callbacks.append(early_stopping_callback)

        # Saving only best checkpoint
        checkpoint_callback = ModelCheckpoint(
            dirpath=f'checkpoints/{logger.experiment.name}',
            filename="best-{epoch:02d}-{val_loss:.3f}",
            save_top_k=1,
            verbose=True,
            monitor="val_loss",
            mode="min",
        )
        callbacks.append(checkpoint_callback)

        strategy = STRATEGY if TRUE_NUM_GPUs >= 1 else None

        trainer = pl.Trainer(strategy=strategy, accelerator=ACCELERATOR, devices=NUM_GPUs, max_epochs=TRAIN_EPOCHS,
                accumulate_grad_batches=GRADIENT_ACCUMULATION_STEPS, callbacks=callbacks, logger=logger)

        if RESUME_TRAINING:
            trainer.fit(model, data, ckpt_path=CHECKPOINT_PATH)
        else:
            trainer.fit(model, data)
    finally:
        # data.cleanup()
        wandb.finish()

def validation(MODEL_NAME: str, BATCH_SIZE: int, NUM_WORKERS: int, NUM_GPUs: int, CHECKPOINT_PATH: str,
            CRF_REDUCTION: str, verbose: bool):
    
    def _validate(model, data, dataloader, GPU):
        auto_generated_score = load('f1')
        needs_action_score = load('f1')
        spam_score = load('f1')
        business_score = load('f1')
        writing_style_score = load('f1')
        softmax = torch.nn.Softmax(dim = 0)
        sigmoid = torch.nn.Sigmoid()

        
        # Same Dictionary

        with torch.no_grad():
            overlap_index = -1
            working_example = {}
            for batch in dataloader():
                input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
                
                outputs = model(input_ids=input_ids.to(GPU), attention_mask=attention_mask.to(GPU), labels=None)
                [auto_generated, needs_action, spam, business, writing_style] = outputs.logits
                
                overlap_index += 1
                if working_example == {}:
                    working_example['dataset_id'] = batch['dataset_id'][0]
                    working_example['personal_id'] = batch['personal_id'][0]
                    
                    
                    # Pentru choices
                    working_example['auto_generated'] = {'tokens': auto_generated[0]}
                    working_example['needs_action'] = {'tokens': needs_action[0]}
                    working_example['spam'] = {'tokens': spam[0]}
                    working_example['business'] = {'tokens': business[0]}
                    working_example['writing_style'] = {'tokens': writing_style}

                    working_labels['auto_generated'] = {'tokens': labels['labels_choices'][0][0]}
                    working_labels['needs_action'] = {'tokens': labels['labels_choices'][0][1]}
                    working_labels['spam'] = {'tokens': labels['labels_choices'][0][2]}
                    working_labels['business'] = {'tokens': labels['labels_choices'][0][3]}
                    working_labels['writing_style'] = {'tokens': labels['labels_choices'][0][4]}


                else:
                    if working_example['dataset_id'] == batch['dataset_id'][0]:
                        assert working_example['personal_id'] == batch['personal_id'][0] - 1, "You need to modify the RandomSampler to a SequentialSampler"
                        working_example['personal_id'] = batch['personal_id'][0]

                        # For choices
                        working_example['auto_generated']['tokens'] = torch.cat([working_example['auto_generated']['tokens'], auto_generated[0]])
                        working_example['needs_action']['tokens'] = torch.cat([working_example['needs_action']['tokens'], needs_action[0]])
                        working_example['spam']['tokens'] = torch.cat([working_example['spam']['tokens'], spam[0]])
                        working_example['business']['tokens'] = torch.cat([working_example['business']['tokens'], business[0]])
                        working_example['writing_style']['tokens'] = torch.cat([working_example['writing_style']['tokens'], writing_style], dim=0)


                    else:
                        overlap_index = -1

                        # For choices
                        working_example['auto_generated']['tokens'] = torch.mean(working_example['auto_generated']['tokens'], dim=0)
                        working_example['needs_action']['tokens'] = torch.mean(working_example['needs_action']['tokens'], dim=0)
                        working_example['spam']['tokens'] = torch.mean(working_example['spam']['tokens'], dim=0)
                        working_example['business']['tokens'] = torch.mean(working_example['business']['tokens'], dim=0)
                        working_example['writing_style']['tokens'] = torch.mean(working_example['writing_style']['tokens'], dim=0)


                        # Apply activations
                        working_example['auto_generated']['tokens'] = sigmoid(working_example['auto_generated']['tokens'])
                        working_example['needs_action']['tokens'] = sigmoid(working_example['needs_action']['tokens'])
                        working_example['spam']['tokens'] = sigmoid(working_example['spam']['tokens'])
                        working_example['business']['tokens'] = sigmoid(working_example['business']['tokens'])
                        working_example['writing_style']['tokens'] = torch.argmax(softmax(working_example['writing_style']['tokens']))

                        # # Obtain results for email

                        auto_generated_score.add(prediction=(working_example['auto_generated']['tokens'] > 0.5).long().item(), reference=working_labels['auto_generated']['tokens'])
                        needs_action_score.add(prediction=(working_example['needs_action']['tokens'] > 0.5).long().item(), reference=working_labels['needs_action']['tokens'].item())
                        spam_score.add(prediction=(working_example['spam']['tokens'] > 0.5).long().item(), reference=working_labels['spam']['tokens'].item())
                        business_score.add(prediction=(working_example['business']['tokens'] > 0.5).long().item(), reference=working_labels['business']['tokens'].item())
                        writing_style_score.add(prediction=working_example['writing_style']['tokens'].item(), reference=working_labels['writing_style']['tokens'].item())

                        working_example = {}
                        working_labels = {}

        # Final Results
        auto_generated_result = auto_generated_score.compute()
        needs_action_result = needs_action_score.compute()
        spam_result = spam_score.compute()
        business_result = business_score.compute()
        writing_style_result = writing_style_score.compute(average='macro')
        
        print(f"Auto Generated: {auto_generated_result['f1']:4f}")
        print(f"Needs Action: {needs_action_result['f1']:4f}")
        print(f"Spam: {spam_result['f1']:4f}")
        print(f"Business: {business_result['f1']:4f}")
        print(f"Writing Style: {writing_style_result['f1']:4f}")
        print(f"Overall F1: {np.mean([auto_generated_result['f1'], needs_action_result['f1'], spam_result['f1'], business_result['f1'], writing_style_result['f1']]):4f}")


    data = GeneralisticDataModule(model_name=MODEL_NAME, batch_size=1, num_workers=NUM_WORKERS, verbose=verbose, type='train')
    data.prepare_data()
    data.setup()

    model = WeightsModel.load_from_checkpoint(checkpoint_path=CHECKPOINT_PATH, model_name=MODEL_NAME, n_training_steps=None, n_warmup_steps=None,
                optimizer=OPTIMIZER, scheduler=SCHEDULER, learning_rate=LEARNING_RATE, extra_hyperparams=None,
                datamodule=data)
    GPU = torch.device('cuda')
    model.to(GPU)
    model.eval()
    # for train
    print('Train statistics:')
    _validate(model, data, data.train_dataloader, GPU)
    print()
    # for val
    print('Validation statistics:')
    _validate(model, data, data.val_dataloader, GPU)
    print()
    # for test
    print('Test statistics:')
    _validate(model, data, data.test_dataloader, GPU)
    print()



if __name__ == '__main__':
    """
    Modify this to train or validate; Also modify hyperparameters here
    
    ### Common params
        MODEL_NAME: Name of huggingface model to use
        BATCH_SIZE: Batch size
        NUM_WORKERS: Number of workers for data loading
        SEED: Random seed
        VERBOSE: Print training information (boolean)
        CHECKPOINT_PATH: Path to checkpoint for evaluation or resuming training (e.g. 'checkpoints/RUN_NAME/best-epoch=X-val_loss=Y.ckpt')

    ### Training params
        TRAIN_EPOCHS: Number of epochs to train
        GRADIENT_ACCUMULATION_STEPS: Number of steps to accumulate gradients for
        NUM_GPUs: Number of GPUs to use
            - Normally you give it as an int (ex: 1, 2, 4)
            - -1 to use all GPUs
            - Can also give a list of gpu ids (ex: for fep could be [0,1], [0] or [1])
        WARMUP_PROCENTAGE: Percentage of training steps to use for warmup (in ints => 10% = 10)
        OPTIMIZER: Optimizer name to use between [AdamW]
        SCHEDULER: Scheduler name to use between [constant, linear, cosine, cosine_with_restarts]
        LEARNING_RATE: Learning rate to use
        EARLY_STOPPING: Use early stopping (Either None or int representing patience)
        ACCELERATOR: Accelerator to use (Either 'gpu' or 'cpu')
        STRATEGY: Strategy to use; One of [None, 'ddp', 'deepspeed_stage_3', 'deepspeed_stage_2', 'deepspeed_stage_3_offload', 'deepspeed_stage_2_offload']
                When NUM_GPUs == 1, None will be used instead of selected strategy
        RUN_NAME: Name of run to use for logging (by default is left blank and will be automatically generated)
        RESUME_TRAINING: Resume training from checkpoint (boolean)
    """

    # Common params
    MODEL_NAME = 'readerbench/RoBERT-large'
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    VERBOSE=True
    SEED = 42
    CHECKPOINT_PATH = 'checkpoint/...'
    # Training params
    TRAIN_EPOCHS = 100
    GRADIENT_ACCUMULATION_STEPS = 1
    NUM_GPUs = list(range(torch.cuda.device_count()))
    WARMUP_PROCENTAGE = 5
    OPTIMIZER = 'AdamW'
    SCHEDULER = 'constant'
    LEARNING_RATE = 5e-5
    EARLY_STOPPING = 10
    ACCELERATOR = 'gpu'
    STRATEGY = 'ddp_find_unused_parameters_true'
    RUN_NAME = None
    RESUME_TRAINING = False
    CRF_REDUCTION = 'token_mean'


    pl.seed_everything(SEED)

    # for LEARNING_RATE in [2.5e-4, 1e-4, 7.5e-5, 5e-5, 2.5e-5, 1e-5, 7.5e-6]:
    # # Uncomment train or evaluation depending on needs
    #     RUN_NAME = f'RoBERT_{LEARNING_RATE:.2e}'
    #     train(MODEL_NAME=MODEL_NAME, BATCH_SIZE=BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS=GRADIENT_ACCUMULATION_STEPS,
    #         TRAIN_EPOCHS=TRAIN_EPOCHS, NUM_WORKERS=NUM_WORKERS, NUM_GPUs=NUM_GPUs, WARMUP_PROCENTAGE=WARMUP_PROCENTAGE,
    #         OPTIMIZER=OPTIMIZER, SCHEDULER=SCHEDULER, LEARNING_RATE=LEARNING_RATE, EARLY_STOPPING=EARLY_STOPPING,
    #         STRATEGY=STRATEGY, ACCELERATOR=ACCELERATOR, RUN_NAME=RUN_NAME, CHECKPOINT_PATH=CHECKPOINT_PATH,
    #         RESUME_TRAINING=RESUME_TRAINING, CRF_REDUCTION=CRF_REDUCTION, verbose=VERBOSE)

    # validation(MODEL_NAME=MODEL_NAME, BATCH_SIZE=BATCH_SIZE, NUM_WORKERS=NUM_WORKERS, NUM_GPUs=NUM_GPUs,
    #     CHECKPOINT_PATH=CHECKPOINT_PATH, CRF_REDUCTION=CRF_REDUCTION, verbose=VERBOSE)
    