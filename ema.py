import mindspore.nn as nn
import mindspore.ops as P
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P

_ema_op = C.MultitypeFuncGraph("grad_ema_op")


@_ema_op.register("Tensor", "Tensor", "Tensor")
def _ema_weights(factor, ema_weight, weight):
    """Apply grad sum to cumulative gradient."""
    add = P.Assign()
    return add(ema_weight, ema_weight * factor + weight * (1 - factor))


class EMACell(nn.Cell):
    def __init__(self, weights, ema_decay=0.9999):
        super(EMACell, self).__init__()
        self.ema_weights = weights.clone(prefix="ema_weights")
        self.ema_decay = Tensor(ema_decay, mstype.float32)
        self.hyper_map = C.HyperMap()
        self.print = P.Print()

    def construct(self, weights):
        success = self.hyper_map(F.partial(_ema_op, self.ema_decay), self.ema_weights, weights)
        return success
