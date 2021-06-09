import torch
from torch.optim import SGD, Adam


def create_dp_optimizer(cls):
    class DPOptimizer(cls):
        def __init__(self, l2_norm_clip, noise_multiplier, minibatch_size, microbatch_size=1, *args, **kwargs):
            super(DPOptimizer, self).__init__(*args, **kwargs)

            self.l2_norm_clip = l2_norm_clip
            self.noise_multiplier = noise_multiplier
            self.minibatch_size = minibatch_size
            self.mircobatch_size = microbatch_size

            for group in self.param_groups:
                group['accum_grads'] = [torch.zeros_like(param.data) if param.requires_grad else None for param in group['params']]

        def zero_mircobatch_grad(self):
            super(DPOptimizer, self).zero_grad()

        def mircobatch_step(self):
            total_norm = 0.
            for group in self.param_groups:
                for param in group['params']:
                    if param.requires_grad:
                        total_norm += param.grad.data.norm(2).item() ** 2.
            total_norm = total_norm ** .5

            clip_coeff = min(self.l2_norm_clip / (total_norm + 1e-6), 1.)

            for group in self.param_groups:
                for param, accum_grad in zip(group['params'], group['accum_grads']):
                    if param.requires_grad:
                        accum_grad.add_(param.grad.data.mul_(clip_coeff))

        def zero_grad(self):
            for group in self.param_groups:
                for accum_grad in group['accum_grads']:
                    if accum_grad is not None:
                        accum_grad.zero_()

        def step(self, *args, **kwargs):
            for group in self.param_groups:
                for param, accum_grad in zip(group['params'], group['accum_grads']):
                    if param.requires_grad:
                        param.grad.data = accum_grad.detach().clone()
                        param.grad.data.add_(self.l2_norm_clip * self.noise_multiplier * torch.randn_like(param.grad.data))
                        param.grad.data.mul_(self.mircobatch_size / self.minibatch_size)
            super(DPOptimizer, self).step(*args, **kwargs)

    return DPOptimizer


if __name__ == '__main__':
    DPSGD = create_dp_optimizer(SGD)
    DPAdam = create_dp_optimizer(Adam)
