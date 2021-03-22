import torch
import numpy as np
from torch.optim.optimizer import Optimizer, required
from DNN_models import cifar10_CNN, cifar10_DenseNet, cifar10_ResNet18



class Adamoptimizer(Optimizer):
    def __init__(self, params, lr=required, momentum=0.9, beta=0.99, weight_decay=0,epsilon=0.0000001):
        defaults = dict(lr=lr, momentum=momentum, beta=beta, weight_decay=weight_decay, epsilon=epsilon)

        super(Adamoptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adamoptimizer, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            epsilon = group['epsilon']
            beta = group['beta']
            grad_length = torch.tensor(0).cuda().float()
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if 'v_buffer' not in param_state:
                    param_state['v_buffer'] = (1-beta) * (d_p.clone() ** 2)
                    param_state['time_buffer'] = 1
                else:
                    param_state['v_buffer'] = param_state['v_buffer'] * beta + (1 - beta) * (d_p.clone() ** 2)
                    param_state['time_buffer'] += 1
                v_buf = param_state['v_buffer'] / (1 - beta ** param_state['time_buffer'])
                if momentum != 0:
                    if 'I_buffer' not in param_state:
                        I_buf = param_state['I_buffer'] = torch.zeros_like(p.data)
                        I_buf = d_p * (1 - momentum)
                    else:
                        I_buf = param_state['I_buffer']
                        I_buf.mul_(momentum).add_(1 - momentum, d_p)
                else:
                    raise ValueError('Please using RMSprop instead')

                grad = (I_buf / (1 - momentum ** param_state['time_buffer'])) / (v_buf ** 0.5 + epsilon)
                p.data.add_(-group['lr'], grad)
                grad_length += torch.sum(grad.detach() ** 2)
            #print(torch.sqrt(grad_length))
        return loss


class SGD_momentumoptimizer(Optimizer):  #SGD with momentum
    def __init__(self, params, lr=required, momentum=0.9, weight_decay=0):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)

        super(SGD_momentumoptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD_momentumoptimizer, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            grad_length = torch.tensor(0).cuda().float()
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if 'time_buffer' not in param_state:
                    param_state['time_buffer'] = 1
                else:
                    param_state['time_buffer'] += 1
                if momentum != 0:
                    if 'I_buffer' not in param_state:
                        param_state['I_buffer'] = d_p.clone() * (1 - momentum)
                    else:
                        param_state['I_buffer'] = param_state['I_buffer'] * momentum + (1 - momentum) * d_p.clone()
                else:
                    raise ValueError('Please using RMSprop instead')
                grad = param_state['I_buffer'] / (1 - momentum ** param_state['time_buffer'])
                p.data.add_(-group['lr'], grad)
                grad_length += torch.sum(grad ** 2)
            #print(torch.sqrt(grad_length))
        return loss

class B3SGDoptimizer(Optimizer): #normalization
    def __init__(self, params, lr=required, momentum=0.9, weight_decay=0):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)

        super(B3SGDoptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(B3SGDoptimizer, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            if 'time_buffer' not in group:
                group['time_buffer'] = 1
            else:
                group['time_buffer'] += 1
            grad_length = torch.tensor(0).cuda().float()
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    if 'I_buffer' not in param_state:
                        param_state['I_buffer'] = d_p.clone() * (1 - momentum)
                    else:
                        param_state['I_buffer'] = param_state['I_buffer'] * momentum + (1 - momentum) * d_p.clone()
                else:
                    raise ValueError('Please using RMSprop instead')
                param_state['grad1'] = param_state['I_buffer'] / (1 - momentum ** group['time_buffer'])
                grad_length += torch.sum(param_state['grad1'] ** 2)
            bound_l2norm = torch.clamp(torch.sqrt(grad_length), 1)
            grad_decade_rate = group['lr'] / bound_l2norm
            for p in group['params']:
                param_state = self.state[p]
                p.data.add_(-grad_decade_rate, param_state['grad1'])

        return loss

class D3SGDoptimizer(Optimizer): #normalizetion接SGD
    def __init__(self, params, lr=required, momentum=0.9, weight_decay=0):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)

        super(D3SGDoptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(D3SGDoptimizer, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            if 'time_buffer' not in group:
                group['time_buffer'] = 1
            else:
                group['time_buffer'] += 1
            grad_length = torch.tensor(0).cuda().float()
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    if 'I_buffer' not in param_state:
                        param_state['I_buffer'] = d_p.clone() * (1 - momentum)
                    else:
                        param_state['I_buffer'] = param_state['I_buffer'] * momentum + (1 - momentum) * d_p.clone()
                else:
                    raise ValueError('Please using RMSprop instead')
                param_state['grad1'] = param_state['I_buffer'] / (1 - momentum ** group['time_buffer'])
                grad_length += torch.sum(param_state['grad1'] ** 2)
            bound_l2norm = torch.sqrt(grad_length)
            grad_decade_rate = group['lr'] / (bound_l2norm + 0.001)
            for p in group['params']:
                param_state = self.state[p]
                p.data.add_(-grad_decade_rate, param_state['grad1'])

        return loss

class adadiroptimizer(Optimizer): #单自适应方向
    def __init__(self, params, lr=required, momentum=0.9, beta=0.99, weight_decay=0,epsilon=0.0000001):
        defaults = dict(lr=lr, momentum=momentum, beta=beta, weight_decay=weight_decay, epsilon=epsilon)

        super(adadiroptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(adadiroptimizer, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            epsilon = group['epsilon']
            beta = group['beta']
            if 'time_buffer' not in group:
                group['time_buffer'] = 1
            else:
                group['time_buffer'] += 1
            grad_length1 = torch.tensor(0).cuda().float()
            grad_length2 = torch.tensor(0).cuda().float()
            pointmultiply = torch.tensor(0).cuda().float()
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if 'v_buffer' not in param_state:
                    param_state['v_buffer'] = (1-beta) * (d_p.clone() ** 2)
                else:
                    param_state['v_buffer'] = param_state['v_buffer'] * beta + (1 - beta) * (d_p.clone() ** 2)
                v_buf = param_state['v_buffer'] / (1 - beta ** group['time_buffer'])
                if momentum != 0:
                    if 'I_buffer' not in param_state:
                        I_buf = param_state['I_buffer'] = torch.zeros_like(p.data)
                        I_buf = d_p * (1 - momentum)
                    else:
                        I_buf = param_state['I_buffer']
                        I_buf.mul_(momentum).add_(1 - momentum, d_p)
                else:
                    raise ValueError('Please using RMSprop instead')
                param_state['grad1'] = (I_buf / (1 - momentum ** group['time_buffer'])) / (v_buf ** 0.5 + epsilon)
                param_state['grad2'] = I_buf / (1 - momentum ** group['time_buffer'])
                grad_length1 += torch.sum(param_state['grad1'].detach() ** 2)
                grad_length2 += torch.sum(param_state['grad2'].detach() ** 2)
                pointmultiply += torch.sum(param_state['grad1'].detach() * param_state['grad2'].detach())
            bound_l2norm1 = torch.sqrt(grad_length1) + epsilon
            bound_l2norm2 = torch.sqrt(grad_length2) + epsilon
            cosin = pointmultiply / (bound_l2norm2 * bound_l2norm1)
            print(cosin)
            grad_decade_rate = group['lr'] * bound_l2norm2 / bound_l2norm1
            for p in group['params']:
                param_state = self.state[p]
                p.data.add_(-grad_decade_rate, param_state['grad1'])
        return loss


class adabssoptimizer(Optimizer): #带方向自适应的normalization接adadir，SGD学习率逐步下降（有下限）
    def __init__(self, params, lr=required, momentum=0.9, beta=0.99, weight_decay=0,epsilon=0.0000001):
        defaults = dict(lr=lr, momentum=momentum, beta=beta, weight_decay=weight_decay, epsilon=epsilon)

        super(adabssoptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(adabssoptimizer, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            epsilon = group['epsilon']
            beta = group['beta']
            if 'time_buffer' not in group:
                group['time_buffer'] = 1
            else:
                group['time_buffer'] += 1
            grad_length1 = torch.tensor(0).cuda().float()
            grad_length2 = torch.tensor(0).cuda().float()
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if 'v_buffer' not in param_state:
                    param_state['v_buffer'] = (1-beta) * (d_p.clone() ** 2)
                else:
                    param_state['v_buffer'] = param_state['v_buffer'] * beta + (1 - beta) * (d_p.clone() ** 2)
                v_buf = param_state['v_buffer'] / (1 - beta ** group['time_buffer'])
                if momentum != 0:
                    if 'I_buffer' not in param_state:
                        I_buf = param_state['I_buffer'] = torch.zeros_like(p.data)
                        I_buf = d_p * (1 - momentum)
                    else:
                        I_buf = param_state['I_buffer']
                        I_buf.mul_(momentum).add_(1 - momentum, d_p)
                else:
                    raise ValueError('Please using RMSprop instead')
                param_state['grad1'] = (I_buf / (1 - momentum ** group['time_buffer'])) / (v_buf ** 0.5 + epsilon)
                param_state['grad2'] = I_buf / (1 - momentum ** group['time_buffer'])
                grad_length1 += torch.sum(param_state['grad1'].detach() ** 2)
                grad_length2 += torch.sum(param_state['grad2'].detach() ** 2)
            bound_l2norm1 = torch.sqrt(grad_length1) + epsilon
            bound_l2norm2 = torch.sqrt(grad_length2) + epsilon
            low_bound = min(1, 0.1 + group['time_buffer'] * 0.001)
            grad_decade_rate = group['lr'] * min(low_bound, bound_l2norm2) / (bound_l2norm1 * low_bound)
            for p in group['params']:
                param_state = self.state[p]
                p.data.add_(-grad_decade_rate, param_state['grad1'])
        return loss


class adabssoptimizer1(Optimizer): #双alpha只调alpha2的adabss
    def __init__(self, params, lr=required, sgd_lr=required, momentum=0.9, beta=0.99, alpha=2, weight_decay=0,epsilon=0.0000001):
        defaults = dict(lr=lr, sgd_lr=sgd_lr, momentum=momentum, beta=beta, weight_decay=weight_decay, epsilon=epsilon,
                        alpha=alpha)

        super(adabssoptimizer1, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(adabssoptimizer1, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            epsilon = group['epsilon']
            beta = group['beta']
            alpha = group['alpha']
            if 'time_buffer' not in group:
                group['time_buffer'] = 1
            else:
                group['time_buffer'] += 1
            grad_length1 = torch.tensor(0).cuda().float()
            grad_length2 = torch.tensor(0).cuda().float()
            pointmultiply = torch.tensor(0).cuda().float()
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if 'v_buffer' not in param_state:
                    param_state['v_buffer'] = (1-beta) * (d_p.clone() ** 2)
                else:
                    param_state['v_buffer'] = param_state['v_buffer'] * beta + (1 - beta) * (d_p.clone() ** 2)
                v_buf = param_state['v_buffer'] / (1 - beta ** group['time_buffer'])
                if momentum != 0:
                    if 'I_buffer' not in param_state:
                        I_buf = param_state['I_buffer'] = torch.zeros_like(p.data)
                        I_buf = d_p * (1 - momentum)
                    else:
                        I_buf = param_state['I_buffer']
                        I_buf.mul_(momentum).add_(1 - momentum, d_p)
                else:
                    raise ValueError('Please using RMSprop instead')
                param_state['grad1'] = (I_buf / (1 - momentum ** group['time_buffer'])) / (v_buf ** alpha + epsilon)
                param_state['grad2'] = I_buf / (1 - momentum ** group['time_buffer'])
                grad_length1 += torch.sum(param_state['grad1'].detach() ** 2)
                grad_length2 += torch.sum(param_state['grad2'].detach() ** 2)
                pointmultiply += torch.sum(param_state['grad1'].detach() * param_state['grad2'].detach())
            bound_l2norm1 = torch.sqrt(grad_length1) + epsilon
            bound_l2norm2 = torch.sqrt(grad_length2) + epsilon
            cosin = pointmultiply / (bound_l2norm2 * bound_l2norm1)
            if group['time_buffer'] % 10 == 0:
                print(cosin)
            grad_decade_rate = min(group['sgd_lr'] * bound_l2norm2 / bound_l2norm1, group['lr'] / bound_l2norm1)
            for p in group['params']:
                param_state = self.state[p]
                p.data.add_(-grad_decade_rate, param_state['grad1'])
        return loss

class adasgdoptimizer(Optimizer): #adam单自适应步长
    def __init__(self, params, lr=required, momentum=0.9, beta=0.99, weight_decay=0,epsilon=0.0000001):
        defaults = dict(lr=lr, momentum=momentum, beta=beta, weight_decay=weight_decay, epsilon=epsilon)

        super(adasgdoptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(adasgdoptimizer, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            epsilon = group['epsilon']
            beta = group['beta']
            if 'time_buffer' not in group:
                group['time_buffer'] = 1
            else:
                group['time_buffer'] += 1
            grad_length1 = torch.tensor(0).cuda().float()
            grad_length2 = torch.tensor(0).cuda().float()
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if 'v_buffer' not in param_state:
                    param_state['v_buffer'] = (1-beta) * (d_p.clone() ** 2)
                else:
                    param_state['v_buffer'] = param_state['v_buffer'] * beta + (1 - beta) * (d_p.clone() ** 2)
                v_buf = param_state['v_buffer'] / (1 - beta ** group['time_buffer'])
                if momentum != 0:
                    if 'I_buffer' not in param_state:
                        I_buf = param_state['I_buffer'] = torch.zeros_like(p.data)
                        I_buf = d_p * (1 - momentum)
                    else:
                        I_buf = param_state['I_buffer']
                        I_buf.mul_(momentum).add_(1 - momentum, d_p)
                else:
                    raise ValueError('Please using RMSprop instead')
                param_state['grad1'] = (I_buf / (1 - momentum ** group['time_buffer'])) / (v_buf ** 0.5 + epsilon)
                param_state['grad2'] = I_buf / (1 - momentum ** group['time_buffer'])
                grad_length1 += torch.sum(param_state['grad1'].detach() ** 2)
                grad_length2 += torch.sum(param_state['grad2'].detach() ** 2)
            bound_l2norm1 = torch.sqrt(grad_length1) + epsilon
            bound_l2norm2 = torch.sqrt(grad_length2) + epsilon
            grad_decade_rate = group['lr'] * bound_l2norm1 / bound_l2norm2
            for p in group['params']:
                param_state = self.state[p]
                p.data.add_(-grad_decade_rate, param_state['grad2'])
        return loss



class datsoptimizer(Optimizer): #adam接adadir
    def __init__(self, params, lr=required, sgd_lr=required, momentum=0.9, beta=0.99, weight_decay=0,epsilon=0.0000001):
        defaults = dict(lr=lr, sgd_lr=sgd_lr, momentum=momentum, beta=beta, weight_decay=weight_decay, epsilon=epsilon)

        super(datsoptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(datsoptimizer, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            epsilon = group['epsilon']
            beta = group['beta']
            if 'time_buffer' not in group:
                group['time_buffer'] = 1
            else:
                group['time_buffer'] += 1
            grad_length1 = torch.tensor(0).cuda().float()
            grad_length2 = torch.tensor(0).cuda().float()
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if 'v_buffer' not in param_state:
                    param_state['v_buffer'] = (1-beta) * (d_p.clone() ** 2)
                else:
                    param_state['v_buffer'] = param_state['v_buffer'] * beta + (1 - beta) * (d_p.clone() ** 2)
                v_buf = param_state['v_buffer'] / (1 - beta ** group['time_buffer'])
                if momentum != 0:
                    if 'I_buffer' not in param_state:
                        I_buf = param_state['I_buffer'] = torch.zeros_like(p.data)
                        I_buf = d_p * (1 - momentum)
                    else:
                        I_buf = param_state['I_buffer']
                        I_buf.mul_(momentum).add_(1 - momentum, d_p)
                else:
                    raise ValueError('Please using RMSprop instead')
                param_state['grad1'] = (I_buf / (1 - momentum ** group['time_buffer'])) / (v_buf ** 0.5 + epsilon)
                param_state['grad2'] = I_buf / (1 - momentum ** group['time_buffer'])
                grad_length1 += torch.sum(param_state['grad1'].detach() ** 2)
                grad_length2 += torch.sum(param_state['grad2'].detach() ** 2)
            bound_l2norm1 = torch.sqrt(grad_length1)
            bound_l2norm2 = torch.sqrt(grad_length2)
            grad_decade_rate = min(group['lr'], group['sgd_lr']*bound_l2norm2/bound_l2norm1)
            for p in group['params']:
                param_state = self.state[p]
                p.data.add_(-grad_decade_rate, param_state['grad1'])
        return loss

class adatsoptimizer(Optimizer): #adam自适应步长接SGD
    def __init__(self, params, lr=required, sgd_lr=required, momentum=0.9, beta=0.99, weight_decay=0,epsilon=0.0000001):
        defaults = dict(lr=lr, sgd_lr=sgd_lr, momentum=momentum, beta=beta, weight_decay=weight_decay, epsilon=epsilon)

        super(adatsoptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(adatsoptimizer, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            epsilon = group['epsilon']
            beta = group['beta']
            if 'time_buffer' not in group:
                group['time_buffer'] = 1
            else:
                group['time_buffer'] += 1
            grad_length1 = torch.tensor(0).cuda().float()
            grad_length2 = torch.tensor(0).cuda().float()
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if 'v_buffer' not in param_state:
                    param_state['v_buffer'] = (1-beta) * (d_p.clone() ** 2)
                else:
                    param_state['v_buffer'] = param_state['v_buffer'] * beta + (1 - beta) * (d_p.clone() ** 2)
                v_buf = param_state['v_buffer'] / (1 - beta ** group['time_buffer'])
                if momentum != 0:
                    if 'I_buffer' not in param_state:
                        I_buf = param_state['I_buffer'] = torch.zeros_like(p.data)
                        I_buf = d_p * (1 - momentum)
                    else:
                        I_buf = param_state['I_buffer']
                        I_buf.mul_(momentum).add_(1 - momentum, d_p)
                else:
                    raise ValueError('Please using RMSprop instead')
                param_state['grad1'] = (I_buf / (1 - momentum ** group['time_buffer'])) / (v_buf ** 0.5 + epsilon)
                param_state['grad2'] = I_buf / (1 - momentum ** group['time_buffer'])
                grad_length1 += torch.sum(param_state['grad1'].detach() ** 2)
                grad_length2 += torch.sum(param_state['grad2'].detach() ** 2)
            bound_l2norm1 = torch.sqrt(grad_length1)
            bound_l2norm2 = torch.sqrt(grad_length2)
            grad_decade_rate = min(group['lr']*bound_l2norm1/bound_l2norm2, group['sgd_lr'])
            for p in group['params']:
                param_state = self.state[p]
                p.data.add_(-grad_decade_rate, param_state['grad2'])
        return loss

class adabssoptimizer2(Optimizer):  #单alpha的带方向自适应normalization接SGD
    def __init__(self, params, lr=required, sgd_lr=required, momentum=0.9, beta=0.99, alpha=2, weight_decay=0,epsilon=0.0000001):
        defaults = dict(lr=lr, sgd_lr=sgd_lr, momentum=momentum, beta=beta, weight_decay=weight_decay, epsilon=epsilon,
                        alpha=alpha)

        super(adabssoptimizer2, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(adabssoptimizer2, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            epsilon = group['epsilon']
            beta = group['beta']
            alpha = group['alpha']
            if 'time_buffer' not in group:
                group['time_buffer'] = 1
            else:
                group['time_buffer'] += 1
            grad_length1 = torch.tensor(0).cuda().float()
            grad_length2 = torch.tensor(0).cuda().float()
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if 'v_buffer' not in param_state:
                    param_state['v_buffer'] = (1-beta) * (torch.abs(d_p.clone()) ** alpha)
                else:
                    param_state['v_buffer'] = param_state['v_buffer'] * beta + (1 - beta) * (torch.abs(d_p.clone()) ** alpha)
                v_buf = param_state['v_buffer'] / (1 - beta ** group['time_buffer'])
                if momentum != 0:
                    if 'I_buffer' not in param_state:
                        I_buf = param_state['I_buffer'] = torch.zeros_like(p.data)
                        I_buf = d_p * (1 - momentum)
                    else:
                        I_buf = param_state['I_buffer']
                        I_buf.mul_(momentum).add_(1 - momentum, d_p)
                else:
                    raise ValueError('Please using RMSprop instead')
                param_state['grad1'] = (I_buf / (1 - momentum ** group['time_buffer'])) / (v_buf ** (1/alpha) + epsilon)
                param_state['grad2'] = I_buf / (1 - momentum ** group['time_buffer'])
                grad_length1 += torch.sum(param_state['grad1'].detach() ** 2)
                grad_length2 += torch.sum(param_state['grad2'].detach() ** 2)
            bound_l2norm1 = torch.sqrt(grad_length1) + epsilon
            bound_l2norm2 = torch.sqrt(grad_length2) + epsilon
            grad_decade_rate = min(group['sgd_lr'] * bound_l2norm2 / bound_l2norm1, group['lr'] / bound_l2norm1)
            for p in group['params']:
                param_state = self.state[p]
                p.data.add_(-grad_decade_rate, param_state['grad1'])
        return loss

class extractionoptimizer(Optimizer): #方向开方优化器
    def __init__(self, params, lr=required, sgd_lr=required, alpha=required, momentum=0.9, weight_decay=0,epsilon=0.0000001):
        defaults = dict(lr=lr, sgd_lr=sgd_lr, momentum=momentum, alpha=alpha, weight_decay=weight_decay, epsilon=epsilon)

        super(extractionoptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(extractionoptimizer, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            epsilon = group['epsilon']
            alpha = group['alpha']
            if 'time_buffer' not in group:
                group['time_buffer'] = 1
            else:
                group['time_buffer'] += 1
            grad_length1 = torch.tensor(0).cuda().float()
            grad_length2 = torch.tensor(0).cuda().float()
            pointmultiply = torch.tensor(0).cuda().float()
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    if 'I_buffer' not in param_state:
                        I_buf = param_state['I_buffer'] = torch.zeros_like(p.data)
                        I_buf = d_p * (1 - momentum)
                    else:
                        I_buf = param_state['I_buffer']
                        I_buf.mul_(momentum).add_(1 - momentum, d_p)
                else:
                    raise ValueError('Please using RMSprop instead')
                mparam = I_buf / (1 - momentum ** group['time_buffer'])
                param_state['grad'] = mparam
                param_state['signgrad'] = (torch.abs(mparam) ** alpha) * torch.sign(mparam)
                grad_length1 += torch.sum(param_state['signgrad'].detach() ** 2)
                grad_length2 += torch.sum(param_state['grad'].detach() ** 2)
                pointmultiply += torch.sum(param_state['grad'] * param_state['signgrad'])
            bound_l2norm1 = torch.sqrt(grad_length1) + epsilon
            bound_l2norm2 = torch.sqrt(grad_length2) + epsilon
            cosin = pointmultiply / (bound_l2norm2 * bound_l2norm1)
            if group['time_buffer'] % 10 == 0:
                print(cosin)
            grad_decade_rate = min(group['sgd_lr'] * bound_l2norm2 / bound_l2norm1, group['lr'] / bound_l2norm1)
            for p in group['params']:
                param_state = self.state[p]
                p.data.add_(-grad_decade_rate, param_state['signgrad'])
        return loss

class signAdamoptimizer(Optimizer):
    def __init__(self, params, lr=required, momentum=0.9, beta=0.99, weight_decay=0,epsilon=0.0000001):
        defaults = dict(lr=lr, momentum=momentum, beta=beta, weight_decay=weight_decay, epsilon=epsilon)

        super(signAdamoptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(signAdamoptimizer, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            epsilon = group['epsilon']
            beta = group['beta']
            grad_length = torch.tensor(0).cuda().float()
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if 'v_buffer' not in param_state:
                    param_state['v_buffer'] = (1-beta) * (d_p.clone() ** 2) * torch.sign(d_p.clone())
                    param_state['time_buffer'] = 1
                else:
                    param_state['v_buffer'] = param_state['v_buffer'] * beta + (1 - beta) * (d_p.clone() ** 2) * torch.sign(d_p.clone())
                    param_state['time_buffer'] += 1
                v_buf = param_state['v_buffer'] / (1 - beta ** param_state['time_buffer'])
                if momentum != 0:
                    if 'I_buffer' not in param_state:
                        I_buf = param_state['I_buffer'] = torch.zeros_like(p.data)
                        I_buf = d_p * (1 - momentum)
                    else:
                        I_buf = param_state['I_buffer']
                        I_buf.mul_(momentum).add_(1 - momentum, d_p)
                else:
                    raise ValueError('Please using RMSprop instead')

                grad = (I_buf / (1 - momentum ** param_state['time_buffer'])) / (torch.abs(v_buf) ** 0.5 + epsilon)
                p.data.add_(-group['lr'], grad)
                grad_length += torch.sum(grad.detach() ** 2)
            #print(torch.sqrt(grad_length))
        return loss