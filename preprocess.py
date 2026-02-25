import random
import re
import torch

from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
from torch.utils.data import Dataset

MAX_LEN = 128
MLM_PROB = 0.15


def split_sentences(text):
    if text is None:
        return []

    text = text.strip()
    return [s for s in re.split(r'(?<=[.!?])\s+', text) if len(s) > 5]


def load_wikipedia_corpus():
    wiki_en = load_dataset("wikipedia", "20220301.en", split="train")
    wiki_mt = load_dataset("wikipedia", "20220301.mt", split="train")

    wiki = concatenate_datasets([wiki_en, wiki_mt])

    wiki = wiki.remove_columns(
        [c for c in wiki.column_names if c != "text"]
    )

    wiki = wiki.filter(lambda x: x["text"] is not None and len(x["text"]) > 50)

    return wiki


def mask_tokens(input_ids, tokenizer):
    labels = input_ids.clone()

    prob = torch.full(labels.shape, MLM_PROB)

    special_tokens_mask = tokenizer.get_special_tokens_mask(
        input_ids.tolist(),
        already_has_special_tokens=True
    )

    prob.masked_fill_(
        torch.tensor(special_tokens_mask, dtype=torch.bool),
        0.0
    )

    masked_indices = torch.bernoulli(prob).bool()

    labels[~masked_indices] = -100
    input_ids[masked_indices] = tokenizer.mask_token_id

    return input_ids, labels


def create_sentence_pairs(wiki_dataset):
    documents = wiki_dataset["text"]
    doc_count = len(documents)

    pairs = []

    for doc in documents:
        if doc is None:
            continue

        sentences = split_sentences(doc)

        if len(sentences) < 2:
            continue

        if random.random() < 0.5:
            idx = random.randint(0, len(sentences) - 2)
            s1 = sentences[idx]
            s2 = sentences[idx + 1]
            label = 0
        else:
            s1 = random.choice(sentences)

            rand_doc = documents[random.randint(0, doc_count - 1)]
            rand_sents = split_sentences(rand_doc)

            if len(rand_sents) == 0:
                continue

            s2 = random.choice(rand_sents)
            label = 1

        pairs.append((s1, s2, label))

    return pairs


class BertPretrainingDataset(Dataset):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-multilingual-cased"
        )

        wiki = load_wikipedia_corpus()
        self.data = create_sentence_pairs(wiki)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        s1, s2, nsp_label = self.data[idx]

        enc = self.tokenizer(
            s1,
            s2,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = enc["input_ids"].squeeze(0)
        token_type_ids = enc["token_type_ids"].squeeze(0)

        input_ids, mlm_labels = mask_tokens(
            input_ids.clone(),
            self.tokenizer
        )

        return {
            "tokens": input_ids,
            "segment_ids": token_type_ids,
            "is_random_next": torch.tensor(nsp_label, dtype=torch.long)
        }


def build_dataset():
    return BertPretrainingDataset()