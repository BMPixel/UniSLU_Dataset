import os
import pandas as pd
import math
from random import shuffle
from copy import deepcopy
import torch
import typing
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import transformers

ATIS_DATA_PATH = 'processed_datasets/ATIS'
BANKING_DATA_PATH = 'processed_datasets/Banking-8k'
CAIS_DATA_PATH = 'processed_datasets/CAIS'
ECDT_DATA_PATH = 'processed_datasets/ECDT'
HWU_DATA_PATH = 'processed_datasets/HWU64'
MATIS_DATA_PATH = 'processed_datasets/MixATIS'
MSNIPS_DATA_PATH = 'processed_datasets/MixSNIPS'
TOP_DATA_PATH = 'processed_datasets/TOP'
ALL_DATA_PATH = [
    ATIS_DATA_PATH, BANKING_DATA_PATH, CAIS_DATA_PATH, ECDT_DATA_PATH, HWU_DATA_PATH, MATIS_DATA_PATH, MSNIPS_DATA_PATH, TOP_DATA_PATH
]


class ProcessedDataset(Dataset):
    """Dataset of one single task, e.g. train set of ATIS"""

    def __init__(self, data_path: str, section='train', args=None) -> None:
        super().__init__()
        self.data_path = data_path
        train_path = os.path.join(data_path, section + '.tsv')
        intents_path = os.path.join(data_path, 'intent.txt')
        with open(intents_path) as f:
            self.intents = f.read().strip()
        self.raw_dataset = pd.read_csv(train_path, sep='\t')

    def __getitem__(self, index):
        return {
            'task': self.data_path,
            'intents': self.intents,
            'context': self.raw_dataset['utterance'][index],
            'output': self.raw_dataset['output'][index],
        }

    def __len__(self):
        return len(self.raw_dataset)


class ConcatDataset(Dataset):
    def __init__(self, dataset_dict: typing.Dict[str, ProcessedDataset], upsample_temp=1, section='train'):
        self.upsample_temp = upsample_temp
        # Name -> list(dataset)
        self.name2data = {}
        for name, dataset in dataset_dict.items():
            self.name2data[name] = [dataset[i]
                                    for i in range(len(dataset))]

        # Up sampling
        # TODO: Upsample
        self.dataset = []
        for name in sorted(self.name2data.keys()):
            self.dataset.extend(self.name2data[name])

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def upsample(data, weight):
        n_data = len(data)
        assert weight >= 1

        integral = list(range(n_data)) * int(math.floor(weight))
        residual = list(range(n_data))
        shuffle(residual)
        residual = residual[:int(n_data * (weight - int(math.floor(weight))))]
        return [deepcopy(data[idx]) for idx in integral + residual]


class TokenizedDataset(Dataset):
    def __init__(self, tokenizer, seq2seq_dataset, source_len=512, target_len=512) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.seq2seq_dataset = seq2seq_dataset
        self.source_len = source_len
        self.target_len = target_len

        # Input format:
        #  {request(indentify (mixed) intent (and slot))}; intents:{all intents in uppercase}; context: {context}
        self.prompt_intent = 'Indentify intent'
        self.prompt_intent_slot = 'Indentify intent and slots'
        self.prompt_mixintent_slot = 'Indentify mixed intent and slots'
        self.prompt_top = 'Indentify nested intent and slots'
        self.template = '{prompt}; Intents: {intents}; Context: {context}'

    def __len__(self):
        return len(self.seq2seq_dataset)

    def __getitem__(self, index):
        raw_item = self.seq2seq_dataset[index]
        task = raw_item['task']
        intents = raw_item['intents']
        context = raw_item['context']
        output = raw_item['output']

        if task in [BANKING_DATA_PATH]:
            prompt = self.prompt_intent
        elif task in [TOP_DATA_PATH]:
            prompt = self.prompt_top
        elif task in [MATIS_DATA_PATH, MSNIPS_DATA_PATH]:
            prompt = self.prompt_mixintent_slot
        else:
            prompt = self.prompt_intent_slot

        source_text = self.template.format(
            prompt=prompt,
            intents=intents,
            context=context
        )

        target_text = output

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.target_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
        }
