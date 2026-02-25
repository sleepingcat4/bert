import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm

from model import BertPretrainingModel
from config import BertConfig
from preprocess import build_dataset


learning_rate = 1e-4
epochs = 10
batch_size = 32
cut_frac = 0.25
ratio = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = BertConfig()
model = BertPretrainingModel(config).to(device)

optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)

loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

dataset = build_dataset()

train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4
)


def slanted_triangular_lr(step, total_steps):
    cut = int(total_steps * cut_frac)

    if step < cut:
        p = step / max(1, cut)
    else:
        p = 1 - (step - cut) / max(1, total_steps - cut)

    return learning_rate * (1 + p * (ratio - 1))


@torch.no_grad()
def evaluate_model(model, dataloader):
    model.eval()

    total_loss = 0
    total_tokens = 0

    loss_fn_eval = nn.CrossEntropyLoss(
        ignore_index=-100,
        reduction="sum"
    )

    for batch in dataloader:
        input_ids = batch["tokens"].to(device)
        token_type_ids = batch["segment_ids"].to(device)

        mlm_logits, _ = model(input_ids, token_type_ids)

        vocab_size = mlm_logits.size(-1)

        loss = loss_fn_eval(
            mlm_logits.view(-1, vocab_size),
            input_ids.view(-1)
        )

        total_loss += loss.item()
        total_tokens += input_ids.numel()

    avg_loss = total_loss / max(1, total_tokens)

    perplexity = torch.exp(torch.tensor(avg_loss))

    print(f"\nValidation Loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}\n")

    model.train()


total_steps = epochs * len(train_loader)
global_step = 0

valid_loader = train_loader

for epoch in range(epochs):
    pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

    for batch in pbar:
        input_ids = batch["tokens"].to(device)
        token_type_ids = batch["segment_ids"].to(device)
        is_random_next = batch["is_random_next"].to(device)

        for param_group in optimizer.param_groups:
            param_group["lr"] = slanted_triangular_lr(
                global_step,
                total_steps
            )

        mlm_logits, nsp_logits = model(input_ids, token_type_ids)

        vocab_size = mlm_logits.size(-1)

        mlm_loss = loss_fn(
            mlm_logits.view(-1, vocab_size),
            input_ids.view(-1)
        )

        nsp_loss = loss_fn(
            nsp_logits,
            is_random_next
        )

        total_loss = mlm_loss + 0.5 * nsp_loss

        optimizer.zero_grad()
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        pbar.set_postfix(
            MLM=mlm_loss.item(),
            NSP=nsp_loss.item(),
            LR=optimizer.param_groups[0]["lr"]
        )

        global_step += 1

    evaluate_model(model, valid_loader)

torch.save(model.state_dict(), "bert_pretrained.pth")
print("MODEL SAVED!")