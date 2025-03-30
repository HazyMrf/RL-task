import math
import torch

def rescale_adam(optimizer: torch.optim.Optimizer, C: float):
    for param_group in optimizer.param_groups:
        if "lr" in param_group:
            old_lr = param_group["lr"]
            new_lr = old_lr / math.sqrt(C)
            param_group["lr"] = new_lr

        if "betas" in param_group:
            old_betas = param_group["betas"]  # (beta1, beta2)
            beta1_rescaled = old_betas[0] ** (1.0 / C)
            beta2_rescaled = old_betas[1] ** (1.0 / C)

            # TODO: Maybe add clamp for small betas?
            param_group["betas"] = (beta1_rescaled, beta2_rescaled)