import os
import json
from typing import Dict
import json
import argparse


def convert_default(example: Dict) -> Dict:
    offset_mapping = []
    text = ''
    for sent in example['sentences']:
        for token in sent:
            if text == '':
                offset_mapping.append((0, len(token)))
                text += token
            else:
                text += ' ' + token
                offset_mapping.append((len(text) - len(token), len(text)))
    entity_types, entity_start_chars, entity_end_chars = [], [], []
    for ann in example['ner']:
        for start, end, label in ann:
            start, end = offset_mapping[start][0], offset_mapping[end][1]
            entity_types.append(label)
            entity_start_chars.append(start)
            entity_end_chars.append(end)

    start_words, end_words= zip(*offset_mapping)
    return {
        'text': text,
        'entity_types': entity_types,
        'entity_start_chars': entity_start_chars,
        'entity_end_chars': entity_end_chars,
        'id': example['doc_key'],
        'word_start_chars': start_words,
        'word_end_chars': end_words
    }


def convert_conll2003(example: Dict) -> Dict:
    # From token offset to char offset.
    offset_mapping = []
    text = ''
    sentence_offset = [0]
    for sent in example['sentences']:
        sentence_offset.append(sentence_offset[-1] + len(sent))
        for token in sent:
            if text == '':
                offset_mapping.append((0, len(token)))
                text += token
            else:
                text += ' ' + token
                offset_mapping.append((len(text) - len(token), len(text)))
    # Group NER annotations by entity type and use char-level offset.
    assert len(example['sentences']) == len(example['ners']), breakpoint()
    entity_types, entity_start_chars, entity_end_chars = [], [], []
    for sent_i, ann in enumerate(example['ners']):
        offset = sentence_offset[sent_i]
        for start, end, label in ann:
            start += offset
            end += offset
            start, end = offset_mapping[start][0], offset_mapping[end][1]
            entity_types.append(label)
            entity_start_chars.append(start)
            entity_end_chars.append(end)

    start_words, end_words= zip(*offset_mapping)
    return {
        'text': text,
        'entity_types': entity_types,
        'entity_start_chars': entity_start_chars,
        'entity_end_chars': entity_end_chars,
        'id': example['doc_key'],
        'word_start_chars': start_words,
        'word_end_chars': end_words
    }


def main(args):
    if args.task == "conll2003":
        convert = convert_conll2003
    else:
        convert = convert_default
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    entities, docs = 0, 0
    with open(args.output, 'w', encoding='utf-8') as fw, open(args.input, encoding='utf-8') as fr:
        ids = set()
        for ln in fr:
            if ln == '\n':
                continue
            example = json.loads(ln)
            example = convert(example) 
            entities += len(example["entity_types"])
            docs += 1
            assert example['id'] not in ids
            ids.add(example['id'])
            fw.write(json.dumps(example) + '\n')
    print(f"Entities: {entities}")
    print(f"Docs: {docs}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input file")
    parser.add_argument("output", help="Output file")
    parser.add_argument("--task", default=None, help="Task name")
    args = parser.parse_args()
    main(args)