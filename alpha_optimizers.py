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
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
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


class alpha_Adamoptimizer(Optimizer):
    def __init__(self, params, lr=required, momentum=0.9, beta=0.99, weight_decay=0,epsilon=0.0000001, alpha=2):
        defaults = dict(lr=lr, momentum=momentum, beta=beta, weight_decay=weight_decay, epsilon=epsilon, alpha=alpha)

        super(alpha_Adamoptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(alpha_Adamoptimizer, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            epsilon = group['epsilon']
            beta = group['beta']
            alpha=group['alpha']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if 'v_buffer' not in param_state:
                    param_state['v_buffer'] = (1-beta) * (torch.abs(d_p.clone()) ** alpha)
                    param_state['time_buffer'] = 1
                else:
                    param_state['v_buffer'] = param_state['v_buffer'] * beta + (1-beta) * (torch.abs(d_p.clone()) ** alpha)
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

                p.data.add_(-group['lr'], (I_buf / (1 - momentum ** param_state['time_buffer'])) / (v_buf ** (1 / alpha) + epsilon))

        return loss

class alpha_SGDoptimizer(Optimizer):
    def __init__(self, params, lr=required, momentum=0.9, weight_decay=0, alpha=1):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, alpha=alpha)

        super(alpha_SGDoptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(alpha_SGDoptimizer, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            alpha=group['alpha']
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
                        param_state['I_buffer'] = (torch.sign(d_p.clone()) * (torch.abs(d_p.clone()) ** alpha)) * (1 - momentum)
                    else:
                        param_state['I_buffer'] = param_state['I_buffer'] * momentum + (1 - momentum) * (torch.sign(d_p) * (torch.abs(d_p) ** alpha))
                else:
                    raise ValueError('Please using RMSprop instead')

                p.data.add_(-group['lr'], ((torch.sign(param_state['I_buffer']) *
                                            (torch.abs(param_state['I_buffer']) ** (1 / alpha))) / (1 - momentum ** param_state['time_buffer'])))

        return loss

class alpha_ascent_Adamoptimizer(Optimizer):
    def __init__(self, params, lr=required, momentum=0.9, beta=0.99, weight_decay=0,epsilon=0.0000001):
        defaults = dict(lr=lr, momentum=momentum, beta=beta, weight_decay=weight_decay, epsilon=epsilon)

        super(alpha_ascent_Adamoptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(alpha_ascent_Adamoptimizer, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            epsilon = group['epsilon']
            beta = group['beta']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if 'time_buffer' not in param_state:
                    param_state['v_buffer0'] = (1 - beta) * (torch.abs(d_p.clone()) ** 1.5)
                    param_state['v_buffer1'] = (1 - beta) * (torch.abs(d_p.clone()) ** 3)
                    param_state['time_buffer'] = 1
                    v_buf = param_state['v_buffer0'] / (1 - beta ** param_state['time_buffer'])
                elif param_state['time_buffer'] < 1000:
                    param_state['v_buffer0'] = param_state['v_buffer0'] * beta + (1 - beta) * (torch.abs(d_p.clone()) ** 1.5)
                    param_state['v_buffer1'] = param_state['v_buffer1'] * beta + (1 - beta) * (torch.abs(d_p.clone()) ** 3)
                    param_state['time_buffer'] += 1
                    v_buf = param_state['v_buffer0'] / (1 - beta ** param_state['time_buffer'])
                else:
                    param_state['v_buffer1'] = param_state['v_buffer1'] * beta + (1 - beta) * (torch.abs(d_p.clone()) ** 3)
                    param_state['time_buffer'] += 1
                    v_buf = param_state['v_buffer0'] / (1 - beta ** param_state['time_buffer'])
                if momentum != 0:
                    if 'I_buffer' not in param_state:
                        I_buf = param_state['I_buffer'] = torch.zeros_like(p.data)
                        I_buf = d_p * (1 - momentum)
                    else:
                        I_buf = param_state['I_buffer']
                        I_buf.mul_(momentum).add_(1 - momentum, d_p)
                else:
                    raise ValueError('Please using RMSprop instead')

                if param_state['time_buffer'] <= 1000:
                    p.data.add_(-group['lr'], (I_buf / (1 - momentum ** param_state['time_buffer'])) / (v_buf ** (2/3) + epsilon))
                else:
                    p.data.add_(-group['lr'], (I_buf / (1 - momentum ** param_state['time_buffer'])) / (v_buf ** (1/3) + epsilon))

        return loss

class double_alpha_Adamoptimizer(Optimizer):
    def __init__(self, params, lr=required, momentum=0.9, beta=0.99, weight_decay=0,epsilon=0.0000001, alpha=[2,2]):
        defaults = dict(lr=lr, momentum=momentum, beta=beta, weight_decay=weight_decay, epsilon=epsilon, alpha=alpha)

        super(double_alpha_Adamoptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(double_alpha_Adamoptimizer, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            epsilon = group['epsilon']
            beta = group['beta']
            alpha = group['alpha']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if 'v_buffer' not in param_state:
                    param_state['v_buffer'] = (1-beta) * (torch.abs(d_p.clone()) ** alpha[0])
                    param_state['time_buffer'] = 1
                else:
                    param_state['v_buffer'] = param_state['v_buffer'] * beta + (1-beta) * (torch.abs(d_p.clone()) ** alpha[0])
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

                p.data.add_(-group['lr'], (I_buf / (1 - momentum ** param_state['time_buffer'])) / (v_buf ** (1 / alpha[1]) + epsilon))

        return loss


class alpha2ascent_Adamoptimizer(Optimizer):
    def __init__(self, params, lr=required, sgd_lr=0.01, momentum=0.9, beta=0.99, alpha=[2,2], weight_decay=0, epsilon=0.0000001):
        lr_amplify = 1/sgd_lr
        basic_lr = (lr * lr_amplify) ** alpha[0]
        defaults = dict(basic_lr = basic_lr, momentum=momentum, lr_amplify=lr_amplify, beta=beta, alpha=alpha,  weight_decay=weight_decay, epsilon=epsilon)

        super(alpha2ascent_Adamoptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(alpha2ascent_Adamoptimizer, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            epsilon = group['epsilon']
            beta = group['beta']
            alpha = group['alpha']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if 'v_buffer' not in param_state:
                    param_state['v_buffer'] = (1-beta) * (torch.abs(d_p.clone()) ** alpha[0])
                    param_state['time_buffer'] = 1
                else:
                    param_state['v_buffer'] = param_state['v_buffer'] * beta + (1-beta) * (torch.abs(d_p.clone()) ** alpha[0])
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
                alpha2 = alpha[1] + param_state['time_buffer'] / 1000
                lr = (group['basic_lr'] ** (1 / alpha2)) / group['lr_amplify']
                p.data.add_(-lr, (I_buf / (1 - momentum ** param_state['time_buffer'])) / (v_buf ** (1 / alpha2) + epsilon))

        return loss

class alpha2_SGDoptimizer(Optimizer):
    def __init__(self, params, lr=required, momentum=0.9, weight_decay=0, alpha=1):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, alpha=alpha)

        super(alpha2_SGDoptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(alpha2_SGDoptimizer, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            alpha=group['alpha']
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

                p.data.add_(-group['lr'], param_state['I_buffer'] / (1 - momentum ** param_state['time_buffer']))#((torch.sign(param_state['I_buffer']) * (torch.abs(param_state['I_buffer']) ** (1/alpha))) / (1 - momentum ** param_state['time_buffer'])))

        return loss

class SGD_momentumoptimizer(Optimizer):
    def __init__(self, params, lr=required, momentum=0.9, weight_decay=0):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)

        super(SGD_momentumoptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD_momentumoptimizer, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
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

class Adam_to_SGDoptimizer(Optimizer):
    def __init__(self, params, lr=required, sgd_lr=0.01, momentum=0.9, beta=0.99, weight_decay=0,epsilon=0.0000001):
        defaults = dict(lr=lr, momentum=momentum, sgd_lr=sgd_lr, beta=beta, weight_decay=weight_decay, epsilon=epsilon)

        super(Adam_to_SGDoptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam_to_SGDoptimizer, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            epsilon = group['epsilon']
            beta = group['beta']
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
                if param_state['time_buffer'] < 2000:
                    p.data += (-group['lr'] * (8000 - param_state['time_buffer']) *
                               (I_buf / (1 - momentum ** param_state['time_buffer'])) / (v_buf ** 0.5 + epsilon)
                               -group['sgd_lr'] * param_state['time_buffer'] * I_buf / (1 - momentum ** param_state['time_buffer'])) / 30000
                else:
                    p.data.add_(-group['sgd_lr'], I_buf / (1 - momentum ** param_state['time_buffer']))
        return loss


class Adaboundoptimizer(Optimizer):
    def __init__(self, params, lr=required, sgd_lr=0.01, momentum=0.9, beta=0.99, alpha=0.01, weight_decay=0,epsilon=0.0000001):
        defaults = dict(lr=lr, momentum=momentum, sgd_lr=sgd_lr, beta=beta, alpha=alpha, weight_decay=weight_decay, epsilon=epsilon)

        super(Adaboundoptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adaboundoptimizer, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            epsilon = group['epsilon']
            beta = group['beta']
            lr = group['lr']
            sgd_lr = group['sgd_lr']
            alpha = group['alpha']
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
                bound_lr = torch.clamp(lr / (v_buf ** 0.5 + epsilon), sgd_lr * (1 - 1 / (1 + param_state['time_buffer'] * alpha)),
                                       sgd_lr * (1 + 1 / (param_state['time_buffer'] * alpha)))
                p.data -= bound_lr * I_buf / (1 - momentum ** param_state['time_buffer'])

        return loss

class bound_ASGDoptimizer(Optimizer):
    def __init__(self, params, lr=required, sgd_lr=0.01, momentum=0.9, beta=0.99, alpha=0.01, weight_decay=0,epsilon=0.0000001):
        defaults = dict(lr=lr, momentum=momentum, sgd_lr=sgd_lr, beta=beta, alpha=alpha, weight_decay=weight_decay, epsilon=epsilon,
                        step=0, up_bound=100000, low_bound=0.00001, max_v=sgd_lr, min_v=sgd_lr)
        super(bound_ASGDoptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(bound_ASGDoptimizer, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            epsilon = group['epsilon']
            beta = group['beta']
            lr = group['lr']
            sgd_lr = group['sgd_lr']
            alpha = group['alpha']
            group['step'] += 1
            group['up_bound'] = sgd_lr * (1 + 1 / (group['step'] * alpha))
            group['low_bound'] = sgd_lr * (1 - 1 / (1 + group['step'] * alpha))
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
                adaptive_v = lr / (param_state['v_buffer'] ** 0.5 + epsilon)
                max_v = torch.max(adaptive_v)
                if max_v > group['up_bound'] and max_v > group['max_v']:
                    group['max_v'] = max_v
                min_v = torch.min(adaptive_v)
                if min_v < group['low_bound'] and min_v < group['min_v']:
                    group['min_v'] = min_v
                if 'I_buffer' not in param_state:
                    param_state['I_buffer'] = d_p.clone() * (1 - momentum)
                else:
                    I_buf = param_state['I_buffer']
                    I_buf.mul_(momentum).add_(1 - momentum, d_p)
            if group['max_v'] > group['up_bound']:
                sgd_rate = (group['max_v'] - group['up_bound']) / (group['max_v'] - 1 + 10 ^ -8)
                if group['min_v'] < group['low_bound']:
                    sgd_rate = max(sgd_rate, (group['low_bound'] - group['min_v']) / (1 - group['min_v'] + 10 ^ -8))
            elif group['min_v'] < group['low_bound']:
                sgd_rate = (group['low_bound'] - group['min_v']) / (1 - group['min_v'] + 10 ^ -8)
            else:
                sgd_rate = 0
            for p in group['params']:
                param_state = self.state[p]
                p.data -= sgd_rate * sgd_lr * param_state['I_buffer'] * (1 - momentum ** param_state['time_buffer'])
                + (1 - sgd_rate) * lr * param_state['I_buffer'] * (1 - momentum ** param_state['time_buffer']) / \
                ((param_state['v_buffer'] / (1 - beta ** param_state['time_buffer'])) ** 0.5 + group['epsilon'])
            group['max_v'] = group['sgd_lr']
            group['min_v'] = group['sgd_lr']
        return loss


class bound_alpha_adamoptimizer(Optimizer):
    def __init__(self, params, lr=required, sgd_lr=0.01, momentum=0.9, beta=0.99, alpha=[1.5,0.01], weight_decay=0,epsilon=0.0000001):
        defaults = dict(lr=lr, momentum=momentum, sgd_lr=sgd_lr, beta=beta, alpha=alpha, weight_decay=weight_decay, epsilon=epsilon,
                        step=0, up_bound=100000, low_bound=0.00001, max_v=sgd_lr, min_v=sgd_lr)
        super(bound_alpha_adamoptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(bound_alpha_adamoptimizer, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            epsilon = group['epsilon']
            beta = group['beta']
            lr = group['lr']
            sgd_lr = group['sgd_lr']
            alpha = group['alpha']
            group['step'] += 1
            group['up_bound'] = torch.tensor(sgd_lr * (1 + 1 / (group['step'] * alpha[1])))
            group['low_bound'] = torch.tensor(sgd_lr * (1 - 1 / (1 + group['step'] * alpha[1])))
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
                adaptive_v = lr / ((param_state['v_buffer'] / (1 - beta ** param_state['time_buffer'])) ** 0.5 + epsilon)
                max_v = torch.max(adaptive_v)
                if max_v > group['up_bound'] and max_v > group['max_v']:
                    group['max_v'] = max_v
                min_v = torch.min(adaptive_v)
                if min_v < group['low_bound'] and min_v < group['min_v']:
                    group['min_v'] = min_v
                if 'I_buffer' not in param_state:
                    param_state['I_buffer'] = d_p.clone() * (1 - momentum)
                else:
                    I_buf = param_state['I_buffer']
                    I_buf.mul_(momentum).add_(1 - momentum, d_p)
            print(group['max_v'])
            if group['max_v'] > group['up_bound']:
                decade_alpha = torch.log(group['max_v']) / (torch.log(group['up_bound']) + epsilon)
                if group['min_v'] < group['low_bound']:
                    decade_alpha = max(decade_alpha, torch.log(group['min_v']) / torch.log(group['low_bound']))
            elif group['min_v'] < group['low_bound']:
                decade_alpha = torch.log(group['min_v']) / (torch.log(group['low_bound']) - epsilon)
            else:
                decade_alpha = 1
            for p in group['params']:
                param_state = self.state[p]
                p.data -= (sgd_lr / (sgd_lr / lr) ** (1 / decade_alpha)) * param_state['I_buffer'] * (1 - momentum ** param_state['time_buffer']) / \
                ((param_state['v_buffer'] / (1 - beta ** param_state['time_buffer'])) ** (1 / (alpha[0] * decade_alpha)) + group['epsilon'])
            print(group['max_v'])
            group['max_v'] = group['sgd_lr']
            group['min_v'] = group['sgd_lr']
            print(group['max_v'])
        return loss


class Global_Adamoptimizer(Optimizer):
    def __init__(self, params, lr=required, momentum=0.9, beta=0.99, weight_decay=0,epsilon=0.0000001):
        defaults = dict(lr=lr, momentum=momentum, beta=beta, weight_decay=weight_decay, epsilon=epsilon)

        super(Global_Adamoptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Global_Adamoptimizer, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            epsilon = group['epsilon']
            beta = group['beta']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if 'v_buffer' not in param_state:
                    param_state['v_buffer'] = (1-beta) * torch.mean(d_p.clone() ** 2)
                    param_state['time_buffer'] = 1
                else:
                    param_state['v_buffer'] = param_state['v_buffer'] * beta + (1 - beta) * torch.mean(d_p.clone() ** 2)
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

                p.data.add_(-group['lr'], (I_buf / (1 - momentum ** param_state['time_buffer'])) / (v_buf ** 0.5 + epsilon))

        return loss

class lb_Adamoptimizer(Optimizer):
    def __init__(self, params, lr=required, momentum=0.9, beta=0.99, weight_decay=0,epsilon=0.0000001):
        defaults = dict(lr=lr, momentum=momentum, beta=beta, weight_decay=weight_decay, epsilon=epsilon)

        super(lb_Adamoptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(lb_Adamoptimizer, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
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
                param_state['grad1'] = (I_buf / (1 - momentum ** param_state['time_buffer'])) / (v_buf ** 0.5 + epsilon)
                grad_length += torch.sum(param_state['grad1'].detach() ** 2)
            bound_l2norm = torch.clamp(torch.sqrt(grad_length), 30)
            grad_decade_rate = group['lr'] / bound_l2norm
            for p in group['params']:
                param_state = self.state[p]
                p.data.add_(-grad_decade_rate, param_state['grad1'])
        return loss


class lb_SGDoptimizer(Optimizer):
    def __init__(self, params, lr=required, momentum=0.9, weight_decay=0):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)

        super(lb_SGDoptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(lb_SGDoptimizer, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
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
            bound_l2norm = torch.clamp(torch.sqrt(grad_length), min(1, group['time_buffer'] * 0.0005))
            grad_decade_rate = group['lr'] / bound_l2norm
            for p in group['params']:
                param_state = self.state[p]
                p.data.add_(-grad_decade_rate, param_state['grad1'])

        return loss


class alpha2ascent_lbAdamoptimizer(Optimizer):
    def __init__(self, params, lr=required, momentum=0.9, beta=0.99, alpha=[2,2], weight_decay=0, epsilon=0.0000001):
        defaults = dict(lr = lr, momentum=momentum, beta=beta, alpha=alpha,  weight_decay=weight_decay, epsilon=epsilon)

        super(alpha2ascent_lbAdamoptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(alpha2ascent_lbAdamoptimizer, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
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
            grad_length = torch.tensor(0).cuda().float()
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if 'v_buffer' not in param_state:
                    param_state['v_buffer'] = (1-beta) * (torch.abs(d_p.clone()) ** alpha[0])
                else:
                    param_state['v_buffer'] = param_state['v_buffer'] * beta + (1-beta) * (torch.abs(d_p.clone()) ** alpha[0])
                v_buf = param_state['v_buffer'] / (1 - beta ** group['time_buffer'])
                if momentum != 0:
                    if 'I_buffer' not in param_state:
                        param_state['I_buffer'] = d_p.detach() * (1 - momentum)
                    else:
                        param_state['I_buffer'].mul_(momentum).add_(1 - momentum, d_p.detach())
                else:
                    raise ValueError('Please using RMSprop instead')
                alpha2 = alpha[1] + group['time_buffer'] / 3000
                param_state['grad1'] = (param_state['I_buffer'] / (1 - momentum ** group['time_buffer'])) / (v_buf ** (1 / alpha2) + epsilon)
                grad_length += torch.sum(param_state['grad1'] ** 2)
            bound_l2norm = torch.clamp(torch.sqrt(grad_length), min(1,group['time_buffer'] * 0.0005))
            grad_decade_rate = group['lr'] / bound_l2norm
            for p in group['params']:
                param_state = self.state[p]
                p.data.add_(-grad_decade_rate, param_state['grad1'])


        return loss

class direction_Adamoptimizer(Optimizer):
    def __init__(self, params, lr=required, momentum=0.9, beta=0.99, weight_decay=0,epsilon=0.0000001):
        defaults = dict(lr=lr, momentum=momentum, beta=beta, weight_decay=weight_decay, epsilon=epsilon)

        super(direction_Adamoptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(direction_Adamoptimizer, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
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
