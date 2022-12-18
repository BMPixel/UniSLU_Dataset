import os
import re
import pandas as pd
import numpy as np
import json

ATIS_DATASET = 'ATIS'
BANKING_DATASET = 'Banking-8k'
CAIS_DATASET = 'CAIS'
ECDT_DATASET = 'ECDT'
HWU_DATASET = 'HWU64'
MATIS_DATASET = 'MixATIS'
MSNIPS_DATASET = 'MixSNIPS'
TOP_DATASET = 'TOP'


def assemble_output_sequence(utterance: str, slots: dict, intent: str):
    '''Assemble utterance, slots and intent into a unified output sequences'''
    output = utterance
    for name, value in slots.items():
        output = output.replace(value, f'[SL:{name.upper()} {value}]')
    return f'[IN:{intent.upper()} {output}]'


def _read_one_atis_sample(f):
    '''Read one sample of ATIS datasets

    :param f: file handle
    :return: utterance, slots and intent
    '''
    sentence = []
    taggings = []
    line = f.readline().strip()
    if line == '':
        return None
    while line != '':
        pair = line.split(' ')
        sentence.append(pair[0])
        if len(pair) > 1:
            taggings.append(pair[1])
        line = f.readline().strip()
    sentence, intent = sentence[:-1], sentence[-1]

    slots = {}
    for i, tag in enumerate(taggings):
        if tag.startswith('B-'):
            slots[tag[2:]] = sentence[i]
        if tag.startswith('I-'):
            slots[tag[2:]] += ' ' + sentence[i]

    utterance = ' '.join(sentence).strip()
    return utterance, slots, intent


def load_atis_dataset(path='raw_datasets/ATIS'):
    '''Load ATIS dataset from given dictionary

    :param path: data path, defaults to 'raw_datasets/ATIS'
    :type path: str, optional
    :return: dataset df
    :rtype: pd.DataFrame
    '''
    intents = set()
    datasets = {}
    for dataset_name in ['dev', 'test', 'train']:
        datasets[dataset_name] = pd.DataFrame(
            columns=['utterance', 'output'])
        with open(os.path.join(path, dataset_name+'.txt'), 'r') as f:
            while True:
                sample = _read_one_atis_sample(f)
                if sample == None:
                    break
                utterance, slots, intent = sample
                output = assemble_output_sequence(utterance, slots, intent)
                intents.add(intent)
                datasets[dataset_name] = datasets[dataset_name].append(
                    {'utterance': utterance, 'output': output}, ignore_index=True)

    return datasets['train'], datasets['dev'], datasets['test'], list(intents)


def load_banking_dataset(path='raw_datasets/Banking-8k', train_dev_ratio=[0.8, 0.1]):
    """Load Banking-8k dataset"""
    intents = json.load(open(os.path.join(path, 'categories.json')))
    # Load test set
    df = pd.read_csv(os.path.join(path, 'test.csv'))
    test = pd.DataFrame(columns=['utterance', 'output'])
    for row in df.iloc:
        output = assemble_output_sequence(row['text'], {}, row['category'])
        test = test.append({
            'utterance': row['text'],
            'output': output
        }, ignore_index=True)

    # Load train set
    df = pd.read_csv(os.path.join(path, 'train.csv'))
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_size = int(train_dev_ratio[0] * len(df))
    train = pd.DataFrame(columns=['utterance', 'output'])
    dev = pd.DataFrame(columns=['utterance', 'output'])
    for i, row in enumerate(df.iloc):
        output = assemble_output_sequence(row['text'], {}, row['category'])
        if i < train_size:
            train = train.append({
                'utterance': row['text'],
                'output': output
            }, ignore_index=True)
        else:
            dev = dev.append({
                'utterance': row['text'],
                'output': output
            }, ignore_index=True)

    return train, dev, test, intents


def load_ecdt_dataset(path='raw_datasets/ECDT', train_dev_test_ratio=[0.8, 0.1, 0.1]):
    obj = json.load(open(os.path.join(path, 'train.json'), 'r'))
    intents = set()
    df = pd.DataFrame(columns=['utterance', 'output'])
    for sample in obj:
        utterance = sample['text']
        intent = sample['intent']
        slots = sample['slots']
        output = assemble_output_sequence(utterance, slots, intent)
        df = df.append({'utterance': utterance, 'output': output},
                       ignore_index=True)
        intents.add(intent)
    train_ratio, dev_ratio, _ = train_dev_test_ratio
    train, dev, test = np.split(df.sample(frac=1, random_state=42),
                                [int(train_ratio*len(df)), int((train_ratio+dev_ratio)*len(df))])
    return train, dev, test, list(intents)


def load_mix_dataset(path='raw_datasets/MixIntent/MixATIS_clean'):
    """Load ATIS dataset from given dictionary"""
    intents = set()
    datasets = {}
    for dataset_name in ['dev', 'test', 'train']:
        datasets[dataset_name] = pd.DataFrame(
            columns=['utterance', 'output'])
        with open(os.path.join(path, dataset_name+'.txt'), 'r') as f:
            while True:
                sample = _read_one_atis_sample(f)
                if sample == None:
                    break
                utterance, slots, intent = sample
                output = assemble_output_sequence(utterance, slots, intent)
                intents.update(intent.split('#'))
                datasets[dataset_name] = datasets[dataset_name].append(
                    {'utterance': utterance, 'output': output}, ignore_index=True)

    return datasets['train'], datasets['dev'], datasets['test'], list(intents)


def load_top_dataset(path='raw_datasets/TOPv2'):
    """Load TopV2 dataset"""
    datasets = {}
    domains = ['alarm', 'event', 'messaging', 'music',
               'navigation', 'reminder', 'timer', 'weather']
    intents = set()
    reg = re.compile('IN:\S+ ')
    for dataset_name in ['train', 'dev', 'test']:
        datasets[dataset_name] = []
        if dataset_name == 'train':
            dataset_name_suffix = '_train.tsv'
        elif dataset_name == 'dev':
            dataset_name_suffix = '_eval.tsv'
        else:
            dataset_name_suffix = '_test.tsv'

        for domain in domains:
            df = pd.read_csv(os.path.join(
                path, domain+dataset_name_suffix), sep='\t')
            for row in df.iloc:
                datasets[dataset_name].append({
                    "utterance": row['utterance'],
                    "output": row['semantic_parse']
                })
                intents.update([s[3:-1]
                                for s in reg.findall(row['semantic_parse'])])

    train = pd.DataFrame(datasets['train'])
    dev = pd.DataFrame(datasets['dev'])
    test = pd.DataFrame(datasets['test'])
    return train, dev, test, list(intents)


def load_hwu64_dataset(path='raw_datasets/HWU64',
                       train_dev_test_ratio=[.8, .1, .1]):
    """Load HWU64 dataset and return DataFrames"""
    intents = set()
    df = pd.read_csv(os.path.join(path, 'all.txt'), sep=';').reset_index()

    datasets = []
    reg = re.compile(r'\[[^\[\]]*\]')
    for row in df.iloc:
        if type(row['answer_annotation']) != str:
            continue

        intent = row['intent']
        slot_pattern = reg.findall(row['answer_annotation'])
        utterance = row['answer_annotation']
        output = utterance
        for pattern in slot_pattern:
            name, value = pattern[1:-1].split(':')
            name = name.strip()
            value = value.strip()
            utterance = utterance.replace(pattern, value)
            output = output.replace(name+' ', 'SL:' + name.upper())
        output = f"[IN:{intent.upper()} {output}]"
        datasets.append({
            'utterance': utterance,
            'output': output
        })
        intents.add(intent)

    df = pd.DataFrame(datasets)
    train_ratio, _, test_ratio = train_dev_test_ratio
    train, dev, test = np.split(
        df.sample(frac=1, random_state=42), [int(train_ratio * len(df)), int((1-test_ratio) * len(df))])

    return train, dev, test, list(intents)


def _read_one_cais_sample(f_tag, f_in):
    """Read one CAIS sample"""
    sentence = []
    taggings = []
    line = f_tag.readline().strip()
    intent = f_in.readline().strip()
    if line == '':
        return None

    while line != '':
        pair = line.split(' ')
        if len(pair) == 2:
            word, tag = pair
        else:
            word = ' '
            tag = pair[0]
        sentence.append(word)
        taggings.append(tag)
        line = f_tag.readline().strip()

    slots = {}
    for i, tag in enumerate(taggings):
        if tag.startswith('B-'):
            slots[tag[2:]] = sentence[i]
        if tag.startswith('I-') or tag.startswith('E-'):
            slots[tag[2:]] += '' + sentence[i]

    utterance = ''.join(sentence).strip()
    return utterance, slots, intent


def load_cais_dataset(path='raw_datasets/CAIS'):
    """Load CAIS dataset and return DataFrames"""
    intents = set()
    datasets = {}
    for dataset_name in ['valid', 'test', 'train']:
        datasets[dataset_name] = []
        f_tag = open(os.path.join(path, dataset_name, 'ch.' + dataset_name))
        f_in = open(os.path.join(path, dataset_name,
                                 'ch.' + dataset_name + '.intent'))

        while True:
            sample = _read_one_cais_sample(f_tag, f_in)
            if sample == None:
                break
            utterance, slots, intent = sample
            output = assemble_output_sequence(utterance, slots, intent)
            intents.add(intent)
            datasets[dataset_name].append({
                'utterance': utterance,
                'output': output
            })

        f_tag.close()
        f_in.close()

    train = pd.DataFrame(datasets['train'])
    valid = pd.DataFrame(datasets['valid'])
    test = pd.DataFrame(datasets['test'])

    return train, valid, test, list(intents)


def preprocess_dataset(name, output_dir='processed_datasets'):
    """Read abitery dataset using above methods
    And write into given folder"""
    if name == ATIS_DATASET:
        train, dev, test, intents = load_atis_dataset()
    elif name == BANKING_DATASET:
        train, dev, test, intents = load_banking_dataset()
    elif name == ECDT_DATASET:
        train, dev, test, intents = load_ecdt_dataset()
    elif name == MATIS_DATASET:
        train, dev, test, intents = load_mix_dataset()
    elif name == MSNIPS_DATASET:
        train, dev, test, intents = load_mix_dataset(
            path='raw_datasets/MixIntent/MixSNIPS_clean')
    elif name == TOP_DATASET:
        train, dev, test, intents = load_top_dataset()
    elif name == HWU_DATASET:
        train, dev, test, intents = load_hwu64_dataset()
    elif name == CAIS_DATASET:
        train, dev, test, intents = load_cais_dataset()
    else:
        raise ValueError(f'{name} not a valid dataset')

    os.makedirs(os.path.join(output_dir, name), exist_ok=True)
    train.to_csv(os.path.join(output_dir, name, 'train.tsv'),
                 sep='\t', index=False)
    dev.to_csv(os.path.join(output_dir, name, 'dev.tsv'),
               sep='\t', index=False)
    test.to_csv(os.path.join(output_dir, name, 'test.tsv'),
                sep='\t', index=False)
    with open(os.path.join(output_dir, name, 'intents.txt'), 'w') as f_tag:
        f_tag.write(' '.join(intents))
    print(f'Successfully processed {name} dataset!')
