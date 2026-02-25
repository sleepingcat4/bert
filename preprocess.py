import random
from typing import List
from datasets import load_dataset
from transformers import BertTokenizerFast
from tensorflow.core.example import example_pb2


class Config:
    model_name = "bert-base-uncased"
    max_seq_length = 128
    max_predictions_per_seq = 20
    masked_lm_prob = 0.15
    dupe_factor = 5
    random_seed = 1234
    output_file = "wiki_bert_pretrain.pb"


class TrainingInstance:
    def __init__(self, tokens, segment_ids, masked_lm_positions,
                 masked_lm_labels, is_random_next):
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels
        self.is_random_next = is_random_next


def create_int_feature(values):
    return example_pb2.Feature(
        int64_list=example_pb2.Int64List(value=list(values))
    )


def create_float_feature(values):
    return example_pb2.Feature(
        float_list=example_pb2.FloatList(value=list(values))
    )


def create_masked_lm_predictions(tokens, tokenizer, cfg, rng):
    cand_indices = []
    for i, token in enumerate(tokens):
        if token in ["[CLS]", "[SEP]"]:
            continue
        cand_indices.append(i)

    rng.shuffle(cand_indices)
    output_tokens = list(tokens)

    num_to_predict = min(
        cfg.max_predictions_per_seq,
        max(1, int(round(len(tokens) * cfg.masked_lm_prob)))
    )

    masked_positions = []
    masked_labels = []

    for idx in cand_indices:
        if len(masked_positions) >= num_to_predict:
            break

        masked_positions.append(idx)
        masked_labels.append(tokens[idx])

        if rng.random() < 0.8:
            output_tokens[idx] = "[MASK]"
        elif rng.random() < 0.5:
            output_tokens[idx] = tokens[idx]
        else:
            rand_id = rng.randint(0, tokenizer.vocab_size - 1)
            output_tokens[idx] = tokenizer.convert_ids_to_tokens(rand_id)

    return output_tokens, masked_positions, masked_labels


def truncate_seq_pair(tokens_a, tokens_b, max_len, rng):
    while len(tokens_a) + len(tokens_b) > max_len:
        if len(tokens_a) > len(tokens_b):
            if rng.random() < 0.5:
                del tokens_a[0]
            else:
                tokens_a.pop()
        else:
            if rng.random() < 0.5:
                del tokens_b[0]
            else:
                tokens_b.pop()


def create_instances(docs, tokenizer, cfg, rng):
    instances = []
    max_tokens = cfg.max_seq_length - 3

    for _ in range(cfg.dupe_factor):
        for doc_index in range(len(docs)):
            document = docs[doc_index]
            if len(document) < 2:
                continue

            for i in range(len(document) - 1):
                tokens_a = tokenizer.tokenize(document[i])
                is_random_next = rng.random() < 0.5

                if is_random_next:
                    rand_doc = docs[rng.randint(0, len(docs) - 1)]
                    tokens_b = tokenizer.tokenize(
                        rand_doc[rng.randint(0, len(rand_doc) - 1)]
                    )
                else:
                    tokens_b = tokenizer.tokenize(document[i + 1])

                truncate_seq_pair(tokens_a, tokens_b, max_tokens, rng)

                tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
                segment_ids = (
                    [0] * (len(tokens_a) + 2) +
                    [1] * (len(tokens_b) + 1)
                )

                tokens, masked_pos, masked_labels = create_masked_lm_predictions(
                    tokens, tokenizer, cfg, rng
                )

                instances.append(
                    TrainingInstance(
                        tokens,
                        segment_ids,
                        masked_pos,
                        masked_labels,
                        is_random_next
                    )
                )
    return instances


def save_instances(instances, tokenizer, cfg):
    with open(cfg.output_file, "wb") as f:
        for inst in instances:
            input_ids = tokenizer.convert_tokens_to_ids(inst.tokens)
            input_mask = [1] * len(input_ids)
            segment_ids = list(inst.segment_ids)

            while len(input_ids) < cfg.max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            masked_ids = tokenizer.convert_tokens_to_ids(inst.masked_lm_labels)
            masked_weights = [1.0] * len(masked_ids)

            masked_positions = list(inst.masked_lm_positions)

            while len(masked_positions) < cfg.max_predictions_per_seq:
                masked_positions.append(0)
                masked_ids.append(0)
                masked_weights.append(0.0)

            example = example_pb2.Example()
            features = example.features.feature

            features["input_ids"].CopyFrom(create_int_feature(input_ids))
            features["input_mask"].CopyFrom(create_int_feature(input_mask))
            features["segment_ids"].CopyFrom(create_int_feature(segment_ids))
            features["masked_lm_positions"].CopyFrom(
                create_int_feature(masked_positions)
            )
            features["masked_lm_ids"].CopyFrom(
                create_int_feature(masked_ids)
            )
            features["masked_lm_weights"].CopyFrom(
                create_float_feature(masked_weights)
            )
            features["next_sentence_labels"].CopyFrom(
                create_int_feature([1 if inst.is_random_next else 0])
            )

            f.write(example.SerializeToString())


def main():
    cfg = Config()
    rng = random.Random(cfg.random_seed)

    dataset = load_dataset("wikipedia", "20220301.en", split="train")
    tokenizer = BertTokenizerFast.from_pretrained(cfg.model_name)

    documents = []
    for article in dataset:
        paragraphs = article["text"].split("\n")
        sentences = [p.strip() for p in paragraphs if len(p.strip()) > 0]
        if len(sentences) > 1:
            documents.append(sentences)

    instances = create_instances(documents, tokenizer, cfg, rng)
    save_instances(instances, tokenizer, cfg)


if __name__ == "__main__":
    main()