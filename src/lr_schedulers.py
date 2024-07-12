import math
import torch
import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import _LRScheduler



# https://github.com/developer0hye/Learning-Rate-WarmUp
class LearningRateWarmUP(_LRScheduler):
    def __init__(self, optimizer, warmup_iteration, target_lr, after_scheduler=None):
        self.optimizer = optimizer
        self.warmup_iteration = warmup_iteration
        self.target_lr = target_lr
        self.after_scheduler = after_scheduler
        self.step(1)

    def warmup_learning_rate(self, cur_iteration):
        warmup_lr = self.target_lr*float(cur_iteration)/float(self.warmup_iteration)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = warmup_lr

    def step(self, cur_iteration):
        if cur_iteration <= self.warmup_iteration:
            self.warmup_learning_rate(cur_iteration)
        else:
            self.after_scheduler.step(cur_iteration-self.warmup_iteration)
    
    def load_state_dict(self, state_dict):
        self.after_scheduler.load_state_dict(state_dict)
        
class WarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Warmup phase
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_steps 
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing phase
            return [
                self.min_lr + (base_lr - self.min_lr) * 0.5 * 
                (1 + math.cos(math.pi * (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)))
                for base_lr in self.base_lrs
            ]