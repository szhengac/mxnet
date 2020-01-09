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
"""NAG optimizer."""
from __future__ import absolute_import
import warnings
import numpy
from ..ndarray import (NDArray, zeros, clip, sqrt, cast, multi_sum_sq, multi_lars)
from ..ndarray import (sgd_update, mp_sgd_update, nag_mom_update, mp_nag_mom_update)
from .optimizer import Optimizer, register

__all__ = ['NAG']


@register
class NAG(Optimizer):
    """Nesterov accelerated gradient.

    This optimizer updates each weight by::

        state = momentum * state + grad + wd * weight
        weight = weight - (lr * (grad + momentum * state))

    Parameters
    ----------
    momentum : float, optional
       The momentum value.
    multi_precision: bool, optional
        Flag to control the internal precision of the optimizer.
        False: results in using the same precision as the weights (default),
        True: makes internal 32-bit copy of the weights and applies gradients
        in 32-bit precision even if actual weights used in the model have lower precision.
        Turning this on can improve convergence and accuracy when training with float16.
    """
    def __init__(self, momentum=0.0, **kwargs):
        super(NAG, self).__init__(**kwargs)
        self.momentum = momentum

    def create_state_multi_precision(self, index, weight):
        weight_master_copy = None
        if self.multi_precision and weight.dtype == numpy.float16:
            weight_master_copy = weight.astype(numpy.float32)
            return (self.create_state(index, weight_master_copy), weight_master_copy)
        if weight.dtype == numpy.float16 and not self.multi_precision:
            warnings.warn("Accumulating with float16 in optimizer can lead to "
                          "poor accuracy or slow convergence. "
                          "Consider using multi_precision=True option of the "
                          "NAG optimizer")
        return self.create_state(index, weight)

    def create_state(self, index, weight):
        momentum = None
        if self.momentum != 0.0:
            momentum = zeros(weight.shape, weight.context, dtype=weight.dtype)
        return momentum

    def _update_impl(self, index, weight, grad, state, multi_precision=False):
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

        if not multi_precision:
            if state is not None:
                nag_mom_update(weight, grad, state, out=weight, lr=lr, wd=wd, **kwargs)
            else:
                sgd_update(weight, grad, out=weight, lr=lr, wd=wd, **kwargs)
        else:
            if state[0] is not None:
                mp_nag_mom_update(weight, grad, state[0], state[1], out=weight,
                                  lr=lr, wd=wd, **kwargs)
            else:
                mp_sgd_update(weight, grad, state[1], out=weight,
                              lr=lr, wd=wd, **kwargs)

    def update(self, index, weight, grad, state):
        self._update_impl(index, weight, grad, state, multi_precision=False)

    def update_multi_precision(self, index, weight, grad, state):
        use_multi_precision = self.multi_precision and weight.dtype == numpy.float16 \
                                and isinstance(state, (tuple, list))
        self._update_impl(index, weight, grad, state,
                          multi_precision=use_multi_precision)
