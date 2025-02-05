import torch
import torch.nn.functional as F


def evaluate(cfg, model, dl, mode="eval"):

    # Disable model training
    model.eval()

    correct = torch.zeros(len(dl[0].dataset.concept2idx), device=cfg.device)
    total = torch.zeros(len(dl[0].dataset.concept2idx), device=cfg.device)
    correct_per_task = torch.zeros(len(dl), device=cfg.device)
    total_per_task = torch.zeros(len(dl), device=cfg.device)
    correct_per_instr = torch.zeros(len(cfg.all_tasks["indices"]), device=cfg.device)
    total_per_instr = torch.zeros_like(correct_per_instr)

    for t in range(len(dl)):
        for _, batch in enumerate(dl[t]):
            batch = [b.to(cfg.device) for b in batch]

            # Compute output and loss
            with torch.no_grad():
                base, pos, neg = model(
                    x1=batch[0],
                    x2=batch[1],
                    x3=batch[2],
                    mission=batch[3],
                    task=t,
                )
                preds = F.cosine_similarity(base, pos, dim=-1) > F.cosine_similarity(
                    base, neg, dim=-1
                )

            # Count correct samples per concept
            concept_counts = torch.bincount(batch[4], minlength=len(total))
            correct += torch.bincount(
                batch[4], weights=preds.float(), minlength=len(total)
            )
            total += concept_counts

            # Count correct samples per task
            correct_per_task[t] += torch.sum(preds)
            total_per_task[t] += len(preds)

            # Count correct samples per instruction
            instr_counts = torch.bincount(batch[5], minlength=len(total_per_instr))
            correct_per_instr += torch.bincount(
                batch[5], weights=preds.float(), minlength=len(total_per_instr)
            )
            total_per_instr += instr_counts

    # After the loop finishes, keep separate variables for concept-level stats:
    correct_concept = correct  # concept-level counts
    total_concept   = total

    # Then define separate variables for the instruction-level stats:
    correct_instr = correct_per_instr[cfg.all_tasks["indices"]]
    total_instr   = total_per_instr[cfg.all_tasks["indices"]]

    # Now compute overall accuracy based on instructions:
    log_stats = {
        f"{mode}/acc": torch.mean(correct_instr / total_instr)
    }

    # Per-instruction accuracy
    for instr_name, c, t in zip(cfg.all_tasks["tasks"], correct_instr, total_instr):
        log_stats[f"{mode}_per_instr/acc_{'_'.join(instr_name)}"] = (c / t).item()

    # Per-concept accuracy
    for concept_name, (c, t) in zip(dl[0].dataset.concept2idx.keys(), zip(correct_concept, total_concept)):
        log_stats[f"{mode}_per_concept/acc_{concept_name}"] = (c / t).item()

    # Per-task accuracy
    for i, (c, t) in enumerate(zip(correct_per_task, total_per_task)):
        log_stats[f"{mode}_per_task/acc_task_{i}"] = (c / t).item()

    # Reenable model training
    model.train()

    return log_stats
