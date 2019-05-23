import torch
import cuquant as qdq
import numpy as np


def test_qdq_gpu():
    if not torch.cuda.is_available():
        return
    x = torch.randn(1000).cuda().uniform_(-1, 1)
    q = qdq.qdq_gpu(x)
    dq = np.unique(q.cpu().numpy())
    print('x', x)
    print('q', q)
    print('unique q', dq)
    print('# unique q', len(dq))


if __name__ == '__main__':
    test_qdq_gpu()
