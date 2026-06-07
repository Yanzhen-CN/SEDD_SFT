import torch
from torch.cuda.amp import autocast

from model import utils as mutils


def get_answer_loss_fn(noise, graph, train, sampling_eps=1e-3):
    def loss_fn(model, input_ids, answer_mask, t=None):
        if t is None:
            t = (1 - sampling_eps) * torch.rand(input_ids.shape[0], device=input_ids.device) + sampling_eps

        sigma, dsigma = noise(t)
        sampled = graph.sample_transition(input_ids, sigma[:, None])
        perturbed = torch.where(answer_mask, sampled, input_ids)

        log_score_fn = mutils.get_score_fn(model, train=train, sampling=False)
        with autocast(enabled=input_ids.is_cuda, dtype=torch.bfloat16):
            log_score = log_score_fn(perturbed, sigma)

        token_loss = graph.score_entropy(log_score, sigma[:, None], perturbed, input_ids)
        token_loss = dsigma[:, None] * token_loss
        token_loss = token_loss * answer_mask.to(token_loss.dtype)
        denom = answer_mask.sum(dim=-1).clamp_min(1).to(token_loss.dtype)
        return token_loss.sum(dim=-1) / denom

    return loss_fn


def evaluate_answer_loss(model, ema, noise, graph, loader, device, eval_batches=0):
    loss_fn = get_answer_loss_fn(noise, graph, train=False)
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if eval_batches > 0 and batch_idx >= eval_batches:
                break
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            answer_mask = batch["answer_mask"].to(device, non_blocking=True)
            ema.store(model.parameters())
            ema.copy_to(model.parameters())
            loss = loss_fn(model, input_ids, answer_mask).mean()
            ema.restore(model.parameters())
            total += float(loss.item())
            count += 1
    if count == 0:
        raise ValueError("No evaluation batches were produced.")
    return total / count, count
