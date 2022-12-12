# %%
import os
import re
import pandas as pd
import numpy as np

# %%
""" 
Dataset samples
{
    "train": {
        "dialogues": [
            {
                "dialogue_id": 0,
                "turns": [{
                    "turn_id": 0,
                    "speaker": "user",
                    "utterance": "Hello. Please play the music named Whisper.",
                    "annotations": [
                        {
                            "intent": "MUSIC",
                            "slots": {
                                "SONG": "Whisper"
                            }
                        }
                    ]
                }]
            }, 
            {
                "dialogue_id": 1, 
                ...
            }
        ],
        "meta": { ... }
    },
    "test": { ... },
    "dev": { ... }
}
"""


def _read_one_atis_sample(obj, f, dataset_name, intent2slots):
    """Read one sample of ATIS datasets

    :param obj: dataset object
    :param f: file handle
    :param dataset_name: dataset name, dev, test or train
    :return: whether successfully read one sample
    """
    sentence = []
    taggings = []
    line = f.readline().strip()
    if line == '':
        return False
    while line != '':
        pair = line.split(' ')
        sentence.append(pair[0])
        if len(pair) > 1:
            taggings.append(pair[1])
        line = f.readline().strip()
    sentence, intent = sentence[:-1], sentence[-1]
    # update intent-slot mapping
    if intent not in intent2slots:
        intent2slots[intent] = set()

    slots = {}
    for i, tag in enumerate(taggings):
        if tag.startswith('B-'):
            slots[tag[2:]] = sentence[i]
            intent2slots[intent].add(tag[2:])
        if tag.startswith('I-'):
            slots[tag[2:]] += " " + sentence[i]

    obj[dataset_name]['dialogues'].append({
        "dialogue_id": len(obj[dataset_name]['dialogues']),
        "turns": [{
            "turn_id": 0,
            "speaker": "user",
            "utterance": " ".join(sentence).strip(),
            "annotations": [
                {
                    "intent": intent,
                    "slots": slots
                }
            ]
        }]
    })
    return True


def load_atis_dataset(path='raw_datasets/ATIS'):
    """Load ATIS dataset from given dictionary

    :param path: data path, defaults to 'raw_datasets/ATIS'
    :type path: str, optional
    :return: dataset obj
    :rtype: dict
    """
    obj = {}
    intent2slots = {}
    for dataset_name in ['dev', 'test', 'train']:
        obj[dataset_name] = {
            'dialogues': [],
            'meta': {}
        }
        with open(os.path.join(path, dataset_name+'.txt'), 'r') as f:
            while True:
                if not _read_one_atis_sample(obj, f, dataset_name, intent2slots):
                    break

    # turn intent2slots into list dict
    for intent in intent2slots:
        intent2slots[intent] = list(intent2slots[intent])
    for dataset_name in ['dev', 'test', 'train']:
        obj[dataset_name]['meta']['intent2slots'] = intent2slots

    return obj


# %%
def convert_utterance_to_input(utterance, intent2slots):
    """transform utterance into model input

    :param utterance: user utterance
    :param intent2slots: intent2slot 
    :return: model input
    """
    ontology = ' '.join([k.upper() for k in intent2slots])
    return f"[INTENT] {ontology} [UTTERANCE] {utterance}"


def convert_annotation_to_output(annotations, intent2slots):
    """transform a annotation object into output string
    e.g. annotation object:
        [{'intent': 'atis_flight',
        'slots': {'fromloc.city_name': 'baltimore',
        'toloc.city_name': 'dallas',
        'round_trip': 'round trip'}}]

    corresponding output_string: 
        [IN:ATIS_FLIGHT [SL:FROMLOC.CITY_NAME baltimore] [SL:TOLOC.CITY_NAME dallas] [SL:ROUND_TRIP round trip]]

    usage:
    obj = load_atis_dataset()
    format_annotation(obj['train']['dialogues'][0]['turns'][0]['annotations'])

    :param annotations: annotation object
    :param intent2slots: intent2slots
    :return: output sequences
    """
    output_string = ''
    for annotation in annotations:
        intent, slots = annotation['intent'], annotation['slots']
        slots_string = " ".join(
            [f"[SL:{k.upper()} {slots[k] if k in slots else 'No'}]" for k in intent2slots[intent]])
        output_string += f"[IN:{intent.upper()} {slots_string}]"
    return output_string


# %%
def load_hwu64_dataset(path='raw_datasets/HWU64', partition=[0.8, 0.1, 0.1]):
    """Load a HWU64 dataset

    :param path: dataset path, defaults to 'raw_datasets/HWU64'
    :type path: str, optional
    :param partition: partition ratio of train/dev/test set, defaults to [0.8, 0.1, 0.1]
    :type partition: list, optional
    """
    train_size, validate_size, test_size = partition
    intent2slots = {}
    df = pd.read_csv(os.path.join(path, 'all.txt'), sep=';').reset_index()

    samples = []
    for row in df.iloc:
        sample = transform_one_hwu64_sample(row, intent2slots)
        if sample != None:
            samples.append(sample)

    np.random.seed(42)
    np.random.shuffle(samples)
    train, dev, test = np.split(
        samples, [int(train_size * len(samples)), int(1 - test_size) * len(samples)])

    # turn intent2slots into list dict
    for intent in intent2slots:
        intent2slots[intent] = list(intent2slots[intent])

    return {
        'train': {
            'dialogues': train,
            'meta': {'intent2slots': intent2slots}
        },
        'dev': {
            'dialogues': dev,
            'meta': {'intent2slots': intent2slots}
        },
        'test': {
            'dialogues': test,
            'meta': {'intent2slots': intent2slots}
        }
    }


def transform_one_hwu64_sample(row: pd.DataFrame, intent2slots):
    """turn a dataframe form sample into a object one and update intent2slots

    :param row: one sample
    :type row: pd.DataFrame
    :param intent2slots: intent2slots reference
    :type intent2slots: Any
    :return: object sample
    """
    if type(row['answer_annotation']) != str:
        return None

    intent = row['scenario'] + '_' + row['intent']
    if intent not in intent2slots:
        intent2slots[intent] = set()
    # extract entity
    reg = re.compile(r'\[[^\[\]]*\]')
    slots = {}
    slot_pattern = reg.findall(row['answer_annotation'])
    utterance = row['answer_annotation']
    for pattern in slot_pattern:
        # print(pattern)
        name, value = pattern[1:-1].split(':')
        slots[name.strip()] = value.strip()
        intent2slots[intent].add(name.strip())
        utterance = utterance.replace(pattern, value.strip())

    return {
        "dialogue_id": int(row['index']),
        "turns": [{
            "turn_id": 0,
            "speaker": "user",
            "utterance": utterance.strip(),
            "annotations": [
                {
                    "intent": intent,
                    "slots": slots
                }
            ]
        }]
    }


# %%
def load_top_datset(path='raw_datasets/TOPv2'):
    """Load a TopV2 dataset

    :param path: dataset path, defaults to 'raw_datasets/TOPv2'
    :type path: str, optional
    :return: dataset object
    """
    obj = {}
    domains = ['alarm', 'event', 'messaging', 'music',
               'navigation', 'reminder', 'timer', 'weather']
    intents = set()
    for dataset_name in ['train', 'dev', 'test']:
        if dataset_name == 'train':
            dataset_name_suffix = '_train.tsv'
        elif dataset_name == 'dev':
            dataset_name_suffix = '_eval.tsv'
        else:
            dataset_name_suffix = '_test.tsv'
        obj[dataset_name] = {
            'dialogues': [],
            'meta': {}
        }

        for domain in domains:
            domain_dialogues, domain_intents = transform_one_top_dataset_file(
                os.path.join(path, domain+dataset_name_suffix))
            obj[dataset_name]['dialogues'] += domain_dialogues
            intents.update(domain_intents)

    for dataset_name in ['train', 'dev', 'test']:
        obj[dataset_name]['meta']['intent2slots'] = intents

    return obj


def transform_one_top_dataset_file(path):
    """Turn one top dataset file in data array and an intent set

    :param path: file path
    :return: data list and sets of all intents
    """
    print('Loading dataset', path)
    dialogues = []
    intents = set()
    df = pd.read_csv(path, sep='\t').reset_index()
    for row in df.iloc:
        reg = re.compile('IN:\S+ ')
        intents.update([s[3:-1] for s in reg.findall(row['semantic_parse'])])
        dialogues.append({
            'dialogue_id': row['index'],
            'turns': [{
                'turn_id': 0,
                'speaker': 'user',
                'utterance': row['utterance'],
                'annotations': row['semantic_parse']
            }]
        })
    return dialogues, intents


# %%
# load datasets
obj = load_atis_dataset()
obj['train']['meta']['intent2slots']
# %%

inputs = convert_utterance_to_input(
    obj['train']['dialogues'][5]['turns'][0]['utterance'], obj['train']['meta']['intent2slots'])
outputs = convert_annotation_to_output(
    obj['train']['dialogues'][5]['turns'][0]['annotations'], obj['train']['meta']['intent2slots'])
print(inputs, '\n')
print(outputs)

# %%
obj = load_top_datset()

# %%
inputs = convert_utterance_to_input(
    obj['train']['dialogues'][5]['turns'][0]['utterance'], obj['train']['meta']['intents2slots'])
obj['train']['dialogues'][5]['turns'][0]['annotations']
