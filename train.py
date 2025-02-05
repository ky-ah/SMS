import logging
import os

import hydra
import torch
from torch.utils.data import DataLoader
from info_nce import InfoNCE
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import wandb
from datasets.utils import download_dataset, load_data
from datasets import ReplayBuffer
from utils import evaluate
from models import AttentionModel, FiLMedModel, AllAttentionModel
from models.utils import log_network_summary, mem_report, seed, measure_importances
from utils.specialization import ATTENTION_SPLIT, FILM_SPLIT

logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S", level=logging.INFO
)
log = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    log.info(f"Run Configuration: \n{OmegaConf.to_yaml(cfg)}")
    assert not (cfg.ac and cfg.strategy in ["all-shared", "all-split"])
    assert not (cfg.ewc and cfg.replay)

    # Fix all possible sources of randomness
    seed(cfg)

    # Initialize online logging with wandb
    if cfg.wandb:
        wandb_args = dict(
            project=cfg.project_name,
            entity=cfg.entity,
            config=OmegaConf.to_container(cfg),
        )
        if "tags" in cfg:
            wandb_args["tags"] = cfg.tags
        wandb.init(**wandb_args)

    # Check if GPUs available when cuda enabled
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        log.info("CUDA not available. Switching to CPU computing instead.")
        cfg.device = "cpu"

    # Download dataset if necessary
    if not os.path.exists(os.path.join(cfg.data_dir, cfg.dataset)):
        log.info("Download dataset from URL...")
        download_dataset(cfg)
        log.info("Successfully downloaded and extracted dataset from URL.")

    # Initialize model
    if cfg.arch == "FiLM_ResNet" and cfg.strategy in FILM_SPLIT.keys():
        model = FiLMedModel(cfg, FILM_SPLIT[cfg.strategy])
    elif cfg.arch == "Transformer_ResNet" and cfg.strategy in ATTENTION_SPLIT.keys():
        model = AttentionModel(cfg, ATTENTION_SPLIT[cfg.strategy])
    elif cfg.arch == "Transformer_ViT"and cfg.strategy in ATTENTION_SPLIT.keys():
        model = AllAttentionModel(cfg, ATTENTION_SPLIT[cfg.strategy])
    else:
        raise Exception(
            f"The requested combination of {cfg.arch} model and {cfg.strategy} is not configured. "
            f"Check typos or implementation."  # noqa: E501
        )
    
    model = model.float().to(cfg.device)
    log.info(f"Initialized {cfg.arch} model instance.")

    # Load train and test sets
    log.info(f"Loading datasets for training and evaluation...")
    train_dl, eval_dl, test_dl = load_data(cfg)
    log.info(f"Loaded training and evaluation sets.")

    # Check GPU usage after dataset and model loading
    mem_report(log)

    # Log trainable network parameters
    log_network_summary(cfg, model, log)
    if wandb:
        wandb.config.update(OmegaConf.to_container(cfg, resolve=True), allow_val_change=True)
        if cfg.log_weights:
            os.makedirs(os.path.join(wandb.run.dir, "checkpoints"), exist_ok=True)
            torch.save(
                model.state_dict(),
                os.path.join(wandb.run.dir, "checkpoints/model_after_init.pth"),
            )

    # Set up optimizer
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    # Initialize episodic memory / experience replay buffer
    if cfg.replay:
        replay_buffer = ReplayBuffer(capacity=cfg.buffer_size)

    # Configure loss function
    # See https://github.com/RElbers/info-nce-pytorch
    loss_fn = InfoNCE()

    # Freeze VLM params after init
    model.encoder.requires_grad_(False)

    # Initial evaluation to get base performance
    if cfg.eval_after_each_task:
        log_stats = evaluate(cfg, model, eval_dl)
        if cfg.wandb:
            wandb.log(log_stats)
        else:
            log.info(
                f"  Average accuracy after initialization: {log_stats['eval/acc']:.4f}"
            )

    # Continual training
    log.info("================ CL phase ===============")
    for t in range(cfg.T):

        for e in range(cfg.epochs):
            if cfg.replay and t > 0:
                replay_dl = iter(
                    DataLoader(
                        replay_buffer, batch_size=cfg.batch_size, shuffle=True
                    )
                )
            
            for i, batch in tqdm(
                    enumerate(train_dl[t]),
                    desc=f"Epoch {e + 1}",
                    total=len(train_dl[t]),
                    position=0,
                    leave=True,
            ):
                if cfg.ac:
                    if (i + 1) % cfg.ac_freq == 0:
                        model.shared.requires_grad_(True)
                        model.specialized.requires_grad_(False)
                    else:
                        model.shared.requires_grad_(False)
                        model.specialized.requires_grad_(True)

                batch = [b.to(cfg.device) for b in batch]

                # Compute output and loss
                p1, z2, z3 = model(
                    x1=batch[0], x2=batch[1], x3=batch[2], mission=batch[3], task=t
                )
                loss = loss_fn(p1, z2, z3)

                if cfg.wandb and not cfg.continual:
                    wandb.log({"train/loss": loss.item()})

                if (
                        cfg.ewc
                        and t > 0
                        and not (cfg.ac and (i + 1) % cfg.ac_freq != 0)
                ):
                    params = list(model.shared.parameters())

                    ewc_loss = sum(
                        [
                            torch.sum(importances[i] * (param - saved_params[i]) ** 2)
                            for i, param in enumerate(params)
                        ]
                    )
                    loss += cfg.ewc_lambda * ewc_loss

                # Update parameters
                optim.zero_grad()
                loss.backward()
                optim.step()

                # If accommodation + replay, take replay batch
                if cfg.replay and t > 0 and i == 0:
                    try:
                        replay_batch = next(replay_dl)
                        replay_batch = [b.to(cfg.device) for b in replay_batch]

                        # Replay Step
                        p1, z2, z3 = model(
                            x1=replay_batch[0], x2=replay_batch[1], x3=replay_batch[2], mission=replay_batch[3],
                            task=replay_batch[5]
                        )
                        loss = loss_fn(p1, z2, z3)
                        optim.zero_grad()
                        loss.backward()
                        optim.step()

                    except StopIteration:
                        log.info(
                            f"Task {t}, epoch {e}: Replay buffer empty. All replay samples have been used."
                        )
                        pass

                # Store replay samples
                if cfg.replay and e == 0:
                    batch[5] = torch.full_like(batch[5], t)
                    replay_buffer.reservoir_sampling(batch)

            if cfg.eval_after_each_task:
                log_stats["train/epoch"] = e

        if cfg.ewc:
            # Save parameters after learning the current task
            saved_params = [
                param.detach().clone() for param in model.shared.parameters()
            ]

            # Compute importances for the shared parameters
            params = list(model.shared.parameters())

            tmp_importances = [torch.zeros_like(param) for param in params]

            for i, batch in enumerate(train_dl[t]):
                batch = [b.to(cfg.device) for b in batch]

                # Compute output and loss
                p1, z2, z3 = model(
                    x1=batch[0],
                    x2=batch[1],
                    x3=batch[2],
                    mission=batch[3],
                    task=t,
                )
                loss = loss_fn(p1, z2, z3)

                loss.backward()

                for param, imp in zip(params, tmp_importances):
                    if param.grad is not None:
                        imp += param.grad.detach().clone() ** 2

            # Normalize by the number of backprop steps (batches)
            tmp_importances = [
                imp / float(len(train_dl[t])) for imp in tmp_importances
            ]

            # Add to discounted importances
            if cfg.ewc and t > 0:
                importances = [
                    imp * cfg.ewc_discount + tmp_importances[i]
                    for i, imp in enumerate(importances)
                ]
            else:
                importances = tmp_importances

            # Clear gradients
            optim.zero_grad()

        # Evaluate accuracy
        if cfg.eval_after_each_task:
            log_stats = evaluate(cfg, model, eval_dl)
            log_stats["train/task"] = t

            # Measure activations for all-shared strategy if W&B logging enabled
            if cfg.strategy == "all-shared":
                log.info(f"Measuring submodule activations for task {t}...")
                log_stats = measure_importances(cfg, model, train_dl[t], loss_fn, t, log_stats)

            if cfg.wandb:
                wandb.log(log_stats)
            else:
                log.info(
                    f"  Average accuracy after training on task {t}: {log_stats['eval/acc']:.4f}"
                )
        
        if cfg.wandb and cfg.log_weights:
            log.info(f"Task {t}: Logging model weights.")
            torch.save(
                model.state_dict(),
                os.path.join(wandb.run.dir, f"checkpoints/model_after_task_{t}.pth"),
            )

    log.info("=========== CL phase completed ==========")

    # Test model
    log.info("=========== Final evaluation ============")

    if not cfg.eval_after_each_task:
        log_stats = evaluate(cfg, model, eval_dl)
        if cfg.wandb:
            wandb.log(log_stats)
        else:
            log.info(
                f"  Average validation set accuracy after training: {log_stats['eval/acc']:.4f}"
            )

    log_stats = evaluate(cfg, model, test_dl, "test")
    if cfg.wandb:
        wandb.log(log_stats)
    log.info(
        f"Final test accuracy of {cfg.arch} model on {cfg.dataset}: {log_stats['test/acc']:.4f}"
    )
    log.info(f"  Average test set accuracy after training: {log_stats['test/acc']:.4f}")

    log.info("========== Evaluation completed =========")

    if cfg.wandb:
        if cfg.log_weights:
            torch.save(
                model.state_dict(),
                os.path.join(wandb.run.dir, f"checkpoints/model.pth"),
            )
        OmegaConf.save(cfg, os.path.join(wandb.run.dir, "run_config.yaml"))
        wandb.finish()


if __name__ == "__main__":
    main()
