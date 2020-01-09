# coding: utf-8
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: disable=too-many-lines
"""Signum optimizer."""
from __future__ import absolute_import
from ..ndarray import (NDArray, zeros, clip, sqrt, cast, multi_sum_sq, multi_lars)
from ..ndarray import (signsgd_update, signum_update)
from .optimizer import Optimizer, register

__all__ = ['Signum']


@register
class Signum(Optimizer):
    r"""The Signum optimizer that takes the sign of gradient or momentum.

    The optimizer updates the weight by::

        rescaled_grad = rescale_grad * clip(grad, clip_gradient) + wd * weight
        state = momentum * state + (1-momentum)*rescaled_grad
        weight = (1 - lr * wd_lh) * weight - lr * sign(state)

    References
    ----------
    Jeremy Bernstein, Yu-Xiang Wang, Kamyar Azizzadenesheli & Anima Anandkumar. (2018).
    signSGD: Compressed Optimisation for Non-Convex Problems. In ICML'18.

    See: https://arxiv.org/abs/1802.04434

    For details of the update algorithm see
    :class:`~mxnet.ndarray.signsgd_update` and :class:`~mxnet.ndarray.signum_update`.

    This optimizer accepts the following parameters in addition to those accepted
    by :class:`.Optimizer`.

    Parameters
    ----------
    momentum : float, optional
       The momentum value.
    wd_lh : float, optional
       The amount of decoupled weight decay regularization, see details in the original paper at:\
       https://arxiv.org/abs/1711.05101
    """
    def __init__(self, learning_rate=0.01, momentum=0.9, wd_lh=0.0, **kwargs):
        super(Signum, self).__init__(learning_rate=learning_rate, **kwargs)
        self.momentum = momentum
        self.wd_lh = wd_lh

    def create_state(self, index, weight):
        momentum = None
        if self.momentum != 0.0:
            momentum = zeros(weight.shape, weight.context, dtype=weight.dtype, stype=weight.stype)
        return momentum

    def _update_impl(self, index, weight, grad, state):
        assert(isinstance(weight, NDArray))
        assert(isinstance(grad, NDArray))
        self._update_count(index)
        lr = self._get_lr(index)
        wd = self._get_wd(index)

        kwargs = {'rescale_grad': self.rescale_grad}
        if self.momentum > 0:
            kwargs['momentum'] = self.momentum
        if self.clip_gradient:
            kwargs['clip_gradient'] = self.clip_gradient
        if self.wd_lh:
            kwargs['wd_lh'] = self.wd_lh

        if state is not None:
            signum_update(weight, grad, state, out=weight,
                          lr=lr, wd=wd, **kwargs)
        else:
            signsgd_update(weight, grad, out=weight,
                           lr=lr, wd=wd, **kwargs)

    def update(self, index, weight, grad, state):
        self._update_impl(index, weight, grad, state)