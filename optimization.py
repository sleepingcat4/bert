import math
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class AdamWeightDecay(Optimizer):
    def __init__(
        self,
        params,
        lr,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-6,
        exclude_from_weight_decay=("bias", "LayerNorm.weight")
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )
        super().__init__(params, defaults)
        self.exclude_from_weight_decay = exclude_from_weight_decay

    def step(self):
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                m, v = state["m"], state["v"]
                state["step"] += 1

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                update = m / (v.sqrt() + eps)

                if wd > 0 and not any(x in p.shape.__repr__() for x in self.exclude_from_weight_decay):
                    update = update + wd * p.data

                p.data.add_(update, alpha=-lr)


def create_scheduler(optimizer, num_train_steps, num_warmup_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_train_steps - current_step)
            / float(max(1, num_train_steps - num_warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda)


def create_optimizer(model, init_lr, num_train_steps, num_warmup_steps):
    optimizer = AdamWeightDecay(
        model.parameters(),
        lr=init_lr,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-6,
    )

    scheduler = create_scheduler(
        optimizer,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
    )

    return optimizer, scheduler


def training_step(model, batch, optimizer, scheduler):
    model.train()
    optimizer.zero_grad()

    loss = model(**batch)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()
    scheduler.step()

    return loss.item()