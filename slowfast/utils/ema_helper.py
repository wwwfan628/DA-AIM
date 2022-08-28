from slowfast.models import build_model
import torch


@torch.no_grad()
def create_ema_model(cfg):
    ema_model = build_model(cfg)
    # detach parameter values
    if cfg.NUM_GPUS > 1:
        for param in ema_model.module.parameters():
            param.requires_grad = False
    else:
        for param in ema_model.parameters():
            param.requires_grad = False
    return ema_model


@torch.no_grad()
def update_ema_variables(ema_model, model, alpha, global_step, cfg):
    # Use the "true" average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    if cfg.NUM_GPUS > 1:
        for ema_param, param in zip(ema_model.module.parameters(), model.module.parameters()):
            ema_param.mul_(alpha).add_(param.clone().detach(), alpha=1 - alpha)
            assert not ema_param.requires_grad
            # ema_param.data[:] = alpha * ema_param[:].data[:] + (1 - alpha) * param[:].data[:]
    else:
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.mul_(alpha).add_(param.clone().detach(), alpha=1 - alpha)
            assert not ema_param.requires_grad
            # ema_param.data[:] = alpha * ema_param[:].data[:] + (1 - alpha) * param[:].data[:]
    return ema_model
