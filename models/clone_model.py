import torch
import torch.nn as nn
import copy
from torch.nn.parallel.parallel_apply import parallel_apply


class CloneModel(nn.Module):
    def __init__(self, module, batch_size):
        super(CloneModel, self).__init__()
        self.replicas = [module]
        self.batch_size = batch_size
        for i in range(batch_size):
            self.replicas += copy.deepcopy(module)

    def forward(self, *inputs, **kwargs):
        inputs, kwargs = self.scatter(inputs, kwargs)
        for i in range(1, self.batch_size):
            self.replicas[i].load_state_dict(self.replicas[0].state_dict())
        outputs = parallel_apply(self.replicas, inputs, kwargs)
        return self.gather(outputs)

    def scatter(self, inputs, kwargs):
        x = inputs[0]
        xs = torch.split(x, 1)
        kwargs = None
        return [xs], kwargs

    def gather(self, outputs):
        pass
