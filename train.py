import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tensorflow.core.example import example_pb2

from model import BertForPreTraining
from optimization import create_optimizer


class ProtoDataset(Dataset):
    def __init__(self, file_path):
        self.records = []
        with open(file_path, "rb") as f:
            data = f.read()

        offset = 0
        while offset < len(data):
            example = example_pb2.Example()
            size = example.ParseFromString(data[offset:])
            self.records.append(example)
            offset += size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        ex = self.records[idx].features.feature

        def get_int(name):
            return torch.tensor(ex[name].int64_list.value)

        def get_float(name):
            return torch.tensor(ex[name].float_list.value)

        return {
            "input_ids": get_int("input_ids").long(),
            "input_mask": get_int("input_mask").long(),
            "segment_ids": get_int("segment_ids").long(),
            "masked_lm_positions": get_int("masked_lm_positions").long(),
            "masked_lm_ids": get_int("masked_lm_ids").long(),
            "masked_lm_weights": get_float("masked_lm_weights").float(),
            "next_sentence_labels": get_int("next_sentence_labels").long().view(-1),
        }


def gather_indexes(sequence_output, positions):
    batch_size = sequence_output.size(0)
    seq_length = sequence_output.size(1)
    width = sequence_output.size(2)

    flat_offsets = torch.arange(batch_size, device=sequence_output.device) * seq_length
    flat_positions = (positions + flat_offsets.unsqueeze(1)).view(-1)

    flat_sequence = sequence_output.view(batch_size * seq_length, width)
    return flat_sequence[flat_positions]


def compute_mlm_loss(sequence_output, embedding_weight,
                     masked_positions, masked_ids, masked_weights):

    masked_output = gather_indexes(sequence_output, masked_positions)

    logits = masked_output @ embedding_weight.t()
    log_probs = torch.log_softmax(logits, dim=-1)

    loss_fct = nn.NLLLoss(reduction="none")
    per_example_loss = loss_fct(log_probs, masked_ids.view(-1))

    numerator = torch.sum(per_example_loss * masked_weights.view(-1))
    denominator = torch.sum(masked_weights) + 1e-5

    return numerator / denominator


def compute_nsp_loss(pooled_output, labels):
    classifier = nn.Linear(pooled_output.size(-1), 2).to(pooled_output.device)
    logits = classifier(pooled_output)
    return nn.CrossEntropyLoss()(logits, labels.view(-1))


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ProtoDataset(args.input_file)
    dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)

    model = BertForPreTraining.from_config(args.bert_config_file)
    model.to(device)

    optimizer, scheduler = create_optimizer(
        model,
        args.learning_rate,
        args.num_train_steps,
        args.num_warmup_steps,
    )

    global_step = 0
    model.train()

    while global_step < args.num_train_steps:
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()

            sequence_output, pooled_output = model.bert(
                input_ids=batch["input_ids"],
                token_type_ids=batch["segment_ids"],
                attention_mask=batch["input_mask"],
            )

            mlm_loss = compute_mlm_loss(
                sequence_output,
                model.bert.embeddings.word_embeddings.weight,
                batch["masked_lm_positions"],
                batch["masked_lm_ids"],
                batch["masked_lm_weights"],
            )

            nsp_loss = compute_nsp_loss(
                pooled_output,
                batch["next_sentence_labels"],
            )

            loss = mlm_loss + nsp_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            global_step += 1

            if global_step % 100 == 0:
                print(f"step {global_step} | loss {loss.item():.4f}")

            if global_step % args.save_checkpoints_steps == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(args.output_dir, f"ckpt_{global_step}.pt"),
                )

            if global_step >= args.num_train_steps:
                break


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bert_config_file", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_train_steps", type=int, default=100000)
    parser.add_argument("--num_warmup_steps", type=int, default=10000)
    parser.add_argument("--save_checkpoints_steps", type=int, default=1000)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    train(args)


if __name__ == "__main__":
    main()