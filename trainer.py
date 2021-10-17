import mindspore
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore import dtype as mstype
from mindspore.communication.management import get_group_size
from mindspore.context import ParallelMode
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.parallel._auto_parallel_context import auto_parallel_context

from .ema import EMACell

grad_scale = C.MultitypeFuncGraph("grad_scale")
_sum_op = C.MultitypeFuncGraph("grad_sum_op")
_clear_op = C.MultitypeFuncGraph("clear_op")


@_sum_op.register("Tensor", "Tensor")
def _cumulative_grad(grad_sum, grad):
    """Apply grad sum to cumulative gradient."""
    add = P.AssignAdd()
    return add(grad_sum, grad)


@_clear_op.register("Tensor", "Tensor")
def _clear_grad_sum(grad_sum, zero):
    """Apply zero to clear grad_sum."""
    success = True
    success = F.depend(success, F.assign(grad_sum, zero))
    return success


@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * P.Reciprocal()(scale)


class TrainingWrapper(nn.Cell):
    """
    Encapsulation class of SSD network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default: 1.0.
        use_global_nrom(bool): Whether apply global norm before optimizer. Default: False
    """

    def __init__(self, network, optimizer, sens=1.0, use_global_norm=True, clip_global_norm_value=1.0,
                 accumulation_step=1, **kwargs):
        super(TrainingWrapper, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        # self.weights = ms.ParameterTuple(optimizer.parameters)
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = float(sens)
        self.reducer_flag = False
        self.grad_reducer = None
        self.use_global_norm = use_global_norm
        self.clip_global_norm_value = clip_global_norm_value / accumulation_step
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            if auto_parallel_context().get_device_num_is_set():
                degree = context.get_auto_parallel_context("device_num")
            else:
                degree = get_group_size()
            self.grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)
        self.hyper_map = C.HyperMap()
        self.print = P.Print()
        self.accumulation_step = int(accumulation_step)
        self.cur_step_num = mindspore.Parameter(Tensor(0, mstype.int64), requires_grad=False)
        self._grad_sum = optimizer.parameters.clone(prefix="grad_sum", init='zeros')
        self._zeros = optimizer.parameters.clone(prefix="zeros", init='zeros')

        # ema weight update
        self.enable_ema = kwargs.get("enable_ema", False)
        if self.enable_ema:
            ema_decay = kwargs.get("ema_decay", 0.9999)
            self._ema_cell = EMACell(self.weights, ema_decay=ema_decay)
        # self.hyper_map(F.partial(_clear_op), self._grad_sum, self._zeros)
        # self.save_path = "ema_weight.ckpt"

    def construct(self, *args):
        """opt"""
        self.cur_step_num = self.cur_step_num + 1
        weights = self.weights
        loss = self.network(*args)
        if self.accumulation_step == 1:
            sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
            grads = self.grad(self.network, weights)(*args, sens)
            if self.reducer_flag:
                # apply grad reducer on grads
                grads = self.grad_reducer(grads)
            if self.sens > 1:
                grads = self.hyper_map(F.partial(grad_scale, F.scalar_to_array(self.sens)), grads)
            if self.use_global_norm:
                grads = C.clip_by_global_norm(grads, clip_norm=self.clip_global_norm_value)
            self.optimizer(grads)
            if self.enable_ema:
                self._ema_cell(self.weights)

        else:
            loss = loss / self.accumulation_step
            sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
            grads = self.grad(self.network, weights)(*args, sens)
            if self.reducer_flag:
                # apply grad reducer on grads
                grads = self.grad_reducer(grads)
            if self.sens > 1:
                grads = self.hyper_map(F.partial(grad_scale, F.scalar_to_array(self.sens)), grads)
            if self.use_global_norm:
                grads = C.clip_by_global_norm(grads, clip_norm=self.clip_global_norm_value)
            # assign to self._grad_sum
            self.hyper_map(F.partial(_sum_op), self._grad_sum, grads)
            if self.cur_step_num % self.accumulation_step == 0:
                # optimizer
                self.optimizer(self._grad_sum)
                # clear grads
                self.hyper_map(F.partial(_clear_op), self._grad_sum, self._zeros)
                if self.enable_ema:
                    self._ema_cell(self.weights)
        return loss
