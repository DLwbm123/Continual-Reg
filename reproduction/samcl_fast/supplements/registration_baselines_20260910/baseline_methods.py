"""Registration-compatible adapters; never modify the existing source checkout."""
import numpy as np
import torch
from core.methods.ewc_on import EwcOn
from core.methods.si import SI
from core.methods.gpm import GPM
from core.methods.sgd import Sgd


def single(batch, j):
    return {k: [v[j]] if k in ('names', 'keypoints') else v[[j]] for k, v in batch.items()}


class RegistrationEWC(EwcOn):
    def end_task(self, loader):
        fish = torch.zeros_like(self.net.get_params())
        count = 0
        for batch in loader:
            for j in range(len(batch['imgs'])):
                inputs = single(batch, j)
                self.opt.zero_grad()
                output = self(inputs)
                loss = self.get_metrics(inputs, output)['loss_final']
                loss.backward()
                fish += self.net.get_grads().detach().square()
                count += 1
        if not count or not torch.isfinite(fish).all():
            raise ValueError('Invalid empirical Fisher')
        fish /= count
        self.fish = fish if self.fish is None else self.cfg.method.ewc.gamma * self.fish + fish
        self.checkpoint = self.net.get_params().detach().clone()
        self.opt.zero_grad(set_to_none=True)


class RegistrationSI(SI):
    @property
    def device(self):
        return self.cfg.var.obj_operator.device

    def observe(self, inputs, not_aug_inputs=None):
        return super().observe(inputs, None, not_aug_inputs)


class RegistrationGPM(GPM):
    def get_layer_gradient_for_each_layer_reduced_conv(self, loader):
        params = [p for p in self.net.parameters() if p.dim() == self.cfg.dataset.dim + 2]
        all_grads = [[] for _ in params]
        for batch in loader:
            for j in range(len(batch['imgs'])):
                inputs = single(batch, j)
                output = self(inputs)
                loss = self.get_metrics(inputs, output, replay=True)['loss_final']
                # One first-order call gives the same numeric layer gradients;
                # no second derivatives are used by the source GPM algorithm.
                gradients = torch.autograd.grad(loss, params, allow_unused=True)
                for i, (p, grad) in enumerate(zip(params, gradients)):
                    grad = torch.zeros_like(p) if grad is None else grad
                    if not torch.isfinite(grad).all():
                        raise ValueError('Nonfinite GPM representation')
                    all_grads[i].append(grad.detach().reshape(p.shape[0], -1).cpu().numpy().copy())
        if not all_grads[0]:
            raise ValueError('No GPM boundary samples')
        return [np.concatenate(values, axis=0).T for values in all_grads]

    def begin_task(self, loader=None):
        self.feature_mat = [torch.as_tensor(u, device=self.cfg.var.obj_operator.device)
                            for u in self.feature_list]

    def observe(self, inputs, not_aug_inputs=None):
        self.opt.zero_grad()
        output = self(inputs)
        self.metrics = self.get_metrics(inputs, output)
        self.metrics['loss_final'].backward()
        if self.current_task:
            k = 0
            for p in self.net.parameters():
                if p.dim() == self.cfg.dataset.dim + 2:
                    grad = p.grad.view(p.shape[0], -1)
                    basis = self.feature_mat[k]
                    # Equivalent G - G(UU^T), without allocating the dense D x D matrix.
                    grad.sub_((grad @ basis) @ basis.T)
                    k += 1
        self.opt.step()
        return output


CLASSES = {'sequential': Sgd, 'ewc': RegistrationEWC, 'si': RegistrationSI, 'gpm': RegistrationGPM}
FIELDS = {'sequential': (), 'ewc': ('checkpoint', 'fish'),
          'si': ('checkpoint', 'big_omega', 'small_omega'),
          'gpm': ('current_task', 'feature_list', 'sin_value_list')}


def method_state(model, method):
    return {key: getattr(model, key) for key in FIELDS[method]}


def restore_method(model, method, state, device):
    if set(state) != set(FIELDS[method]):
        raise ValueError('Missing method-specific resume state')
    for key, value in state.items():
        setattr(model, key, value.to(device) if torch.is_tensor(value) else value)
    if method == 'gpm':
        model.begin_task()
