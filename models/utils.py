import math
import random
from collections import defaultdict

import humanize
import numpy as np
import psutil
import GPUtil
import torch
import wandb
from omegaconf import open_dict

def seed(cfg):
    # Set seed for all randomness sources (don't forget to set PYTHONHASHSEED as env variable)
    if cfg.seed == 0:
        with open_dict(cfg):
            cfg.seed = np.random.randint(10000)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_number(num):
    return "{:,}".format(num)


# Log count of trainable model params
def log_network_summary(cfg, model, log):
    log.info("===== Overview model parameters =====")
    param_count_total = count_parameters(model)
    log.info(f"Total number of model parameters: {format_number(param_count_total)}")
    param_count_base = count_parameters(model.encoder)
    log.info(
        f"    VLM encoder parameters (frozen): {format_number(param_count_base)}"
    )
    param_count_shared = count_parameters(model.shared)
    log.info(
        f"    Shared parameters (shared across tasks): {format_number(param_count_shared)}"
    )
    param_count_split = count_parameters(model.specialized)
    log.info(
        f"    Task-specific parameters (total): {format_number(param_count_split)}"
    )
    log.info(
        f"    Task-specific parameters (per 1 task): {format_number(int(param_count_split / cfg.T))}"
    )
    with open_dict(cfg):
        cfg.param_count_total = param_count_total
        cfg.param_count_base = param_count_base
        cfg.param_count_shared = param_count_shared
        cfg.param_count_split = param_count_split
        cfg.param_count_per_task = param_count_split / cfg.T


# Report CPU and GPU memory usage
def mem_report(log):
    log.info("============= Memory report =============")
    log.info(
        f"CPU   ... RAM Free: {humanize.naturalsize(psutil.virtual_memory().available)}"
    )
    GPUs = GPUtil.getGPUs()
    for i, gpu in enumerate(GPUs):
        log.info(
            f"GPU {i:d} ... Mem Free: {gpu.memoryFree:.0f}MB / {gpu.memoryTotal:.0f}MB | Utilization {gpu.memoryUtil * 100:3.0f}%"
        )


def measure_importances(cfg, model, dl, loss_fn, t, log_stats={}):
    """
    Measures both:
      1) Activation-based importance (sum of absolute activations)
      2) Gradient-based importance (sum of |param| * 0.5 * grad)
    for each submodule of the fusion network, then logs the result to wandb.
    """

    # 1. Identify relevant submodules for architecture type
    submodules_to_hook = {}
    
    if "FiLM" in cfg.arch:
        for i, layer in enumerate([
            model.FiLM_Layer_0, model.FiLM_Layer_1, 
            model.FiLM_Layer_2, model.FiLM_Layer_3
        ]):
            for sub_name in ["conv1", "conv2", "gamma", "beta", "bn1", "bn2"]:
                submodules_to_hook[f"FiLM_Layer_{i}_{sub_name}"] = getattr(layer, sub_name)
    elif "Transformer" in cfg.arch: 
        for i, layer in enumerate([
            model.Transformer_Layer_0, model.Transformer_Layer_1, 
            model.Transformer_Layer_2, model.Transformer_Layer_3
        ]):
            for sub_name in ["head1", "head2", "ffn1", "ffn2", "ln1", "ln2"]:
                submodules_to_hook[f"Transformer_Layer_{i}_{sub_name}"] = getattr(layer, sub_name)
    else:
        return  # No measurement for other architectures

    # 2. Prepare dictionaries to store sums
    activation_sums = defaultdict(float)
    grad_importance_sums = defaultdict(float)
    grad_sums = defaultdict(float)
    param_sums = defaultdict(float)
    param_counts = {}

    # 3. Forward hook definition for activation importance
    def make_hook(module_name):
        def forward_hook(module, input_, output):
            # Sum of absolute output as a measure of activation
            if isinstance(output, tuple):
                activation_sums[module_name] += output[0].abs().sum().item()
            else:
                activation_sums[module_name] += output.abs().sum().item()
        return forward_hook

    # Register forward hooks and also store param counts
    hooks = []
    for name, submodule in submodules_to_hook.items():
        # Count parameters once
        pcount = sum(p.numel() for p in submodule.parameters() if p.requires_grad)
        param_counts[name] = pcount if pcount > 0 else 1  # to avoid log(0)

        hook_fn = make_hook(name)
        h = submodule.register_forward_hook(hook_fn)
        hooks.append(h)

    # 4. One pass over the dataloader *with* gradient tracking so we can compute grad-based importance
    model.eval()
    # Make sure we allow gradients (no 'torch.no_grad()' here)
    for batch in dl:
        batch = [b.to(cfg.device) for b in batch]

        # Forward pass -> forward hooks will accumulate activation sums
        p1, z2, z3 = model(
            x1=batch[0], x2=batch[1], x3=batch[2], 
            mission=batch[3], task=t
        )

        # Calculate loss and compute gradients
        loss = loss_fn(p1, z2, z3)
        loss.backward()

        # Accumulate gradient-based importance
        with torch.no_grad():
            for submodule_name, submodule in submodules_to_hook.items():
                for param in submodule.parameters():
                    grad = param.grad
                    if grad is not None:
                        # Compute absolute sums once and store as scalars
                        param_abs_sum = param.abs().sum().item()
                        grad_abs_sum = grad.abs().sum().item()

                        # Accumulate the sums
                        param_sums[submodule_name] += param_abs_sum
                        grad_sums[submodule_name] += grad_abs_sum

                        # Compute grad importance using the precomputed sums
                        grad_importance = param_abs_sum + 0.5 * grad_abs_sum
                        grad_importance_sums[submodule_name] += grad_importance

        # Zero gradients before next iteration
        model.zero_grad()

    # 5. Unregister forward hooks to avoid side effects in the rest of the pipeline
    for h in hooks:
        h.remove()

    # 6. Compute final metrics and log them to wandb
    for module_name in activation_sums.keys():
        log_stats[f"importances/{module_name}_num_params"] = param_counts[module_name]
        log_stats[f"importances/{module_name}_is_act"] = activation_sums[module_name]
        log_stats[f"importances/{module_name}_is_grad"] = grad_importance_sums[module_name]
        log_stats[f"importances/{module_name}_grad_sum"] = grad_sums[module_name]
        log_stats[f"importances/{module_name}_param_sum"] = param_sums[module_name]

    # Revert model to training mode
    model.train()

    return log_stats
