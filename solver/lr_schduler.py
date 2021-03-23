from numpy import cos, pi
def adjust_lr(opt, it, lr_init, lr_end, total_it, warmup):
    """
    it: iteration
    total_it: total iteration
    warmup = your epoch * len(trainloader)

    """
    if it < warmup:
        lr = lr_init * ((1 / warmup) * it)
    else:
        lr = lr_end + (lr_init - lr_end) * 0.5 * (1 + cos((it - warmup) / (total_it - warmup) * pi))

    if lr < lr_end:  # boundary
        lr = lr_end
    for param_group in opt.param_groups:
        param_group['lr'] = lr
    return lr