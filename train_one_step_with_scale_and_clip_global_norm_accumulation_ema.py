import mindspore.nn as nn
from mindspore import Parameter, Tensor
from mindspore import dtype as mstype
from mindspore.common import RowTensor
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P

from .ema import EMACell

_sum_op = C.MultitypeFuncGraph("grad_sum_op")
assignadd = P.AssignAdd()
assignadd.add_prim_attr("primitive_target", "CPU")

_grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()
reciprocal.add_prim_attr("primitive_target", "CPU")


@_sum_op.register("Tensor", "Tensor")
def _cumulative_grad(grad_sum, grad):
    """Apply grad sum to cumulative gradient."""

    return assignadd(grad_sum, grad)


@_grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * F.cast(reciprocal(scale), F.dtype(grad))


@_grad_scale.register("Tensor", "RowTensor")
def tensor_grad_scale_row_tensor(scale, grad):
    return RowTensor(grad.indices,
                     grad.values * F.cast(reciprocal(scale), F.dtype(grad.values)),
                     grad.dense_shape)


class TrainOneStepWithLossScaleCellGlobalNormClipAccumulationEMA(nn.TrainOneStepWithLossScaleCell):
    """1
    Encapsulation class of SSD network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default: 1.0.
        use_global_nrom(bool): Whether apply global norm before optimizer. Default: False
    """

    def __init__(self, network, optimizer,
                 scale_sense=1.0, use_global_norm=True,
                 clip_global_norm_value=1.0,
                 **kwargs):
        super(TrainOneStepWithLossScaleCellGlobalNormClipAccumulationEMA, self).__init__(network, optimizer,
                                                                                         scale_sense)
        self.use_global_norm = use_global_norm
        self.clip_global_norm_value = clip_global_norm_value
        self.print = P.Print()
        self.assignadd = P.AssignAdd()
        self.accumulation_step = int(kwargs.get("accumulation_step", 1))
        if self.accumulation_step > 1:
            self._grad_sum = optimizer.parameters.clone(prefix="grad_sum", init='zeros')
            self.cur_step_num = Parameter(Tensor(0, mstype.int64), requires_grad=False)
        self.enable_ema = kwargs.get("enable_ema", False)
        if self.enable_ema:
            ema_decay = kwargs.get("ema_decay", 0.9999)
            self._ema_cell = EMACell(self.weights, ema_decay=ema_decay)

    def construct(self, *inputs):
        weights = self.weights
        loss = self.network(*inputs)
        scaling_sens = self.scale_sense

        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)
        if self.accumulation_step == 1:
            scaling_sens_filled = C.ones_like(loss) * F.cast(scaling_sens, F.dtype(loss))
            grads = self.grad(self.network, weights)(*inputs, scaling_sens_filled)
            grads = self.hyper_map(F.partial(_grad_scale, scaling_sens), grads)
            # apply grad reducer on grads
            grads = self.grad_reducer(grads)
            # get the overflow buffer
            cond = self.get_overflow_status(status, grads)
            overflow = self.process_loss_scale(cond)
            # if there is no overflow, do optimize
            if not overflow:
                if self.use_global_norm:
                    grads = C.clip_by_global_norm(grads, clip_norm=self.clip_global_norm_value)
                loss = F.depend(loss, self.optimizer(grads))
                if self.enable_ema:
                    self._ema_cell(self.weights)
            else:
                self.print("=============Over Flow, skiping=============")
        else:
            loss = loss / self.accumulation_step
            scaling_sens_filled = C.ones_like(loss) * F.cast(scaling_sens, F.dtype(loss))
            grads = self.grad(self.network, weights)(*inputs, scaling_sens_filled)
            grads = self.hyper_map(F.partial(_grad_scale, scaling_sens), grads)
            # apply grad reducer on grads
            grads = self.grad_reducer(grads)
            # get the overflow buffer
            cond = self.get_overflow_status(status, grads)
            overflow = self.process_loss_scale(cond)
            # if there is no overflow, do optimize
            if not overflow:
                if self.use_global_norm:
                    grads = C.clip_by_global_norm(grads, clip_norm=self.clip_global_norm_value)
                self.hyper_map(F.partial(_sum_op), self._grad_sum, grads)
                self.assignadd(self.cur_step_num, 1)
                if self.cur_step_num % self.accumulation_step == 0:
                    loss = F.depend(loss, self.optimizer(self._grad_sum))
                    self.hyper_map(F.partial(_sum_op), self._grad_sum, -self._grad_sum)
                    if self.enable_ema:
                        self._ema_cell(self.weights)
            else:
                self.print(self.cur_step_num, "=============Over Flow, skiping=============")
        return loss
