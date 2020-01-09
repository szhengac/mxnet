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
"""DCASGD optimizer."""
from __future__ import absolute_import
from ..ndarray import (NDArray, zeros, clip, sqrt, cast, multi_sum_sq, multi_lars)
from .optimizer import Optimizer, register

__all__ = ['DCASGD']


@register
class DCASGD(Optimizer):
    """The DCASGD optimizer.

    This class implements the optimizer described in *Asynchronous Stochastic Gradient Descent
    with Delay Compensation for Distributed Deep Learning*,
    available at https://arxiv.org/abs/1609.08326.

    This optimizer accepts the following parameters in addition to those accepted
    by :class:`.Optimizer`.

    Parameters
    ----------
    momentum : float, optional
       The momentum value.

    lamda : float, optional
       Scale DC value.
    """
    def __init__(self, momentum=0.0, lamda=0.04, **kwargs):
        super(DCASGD, self).__init__(**kwargs)
        self.momentum = momentum
        self.weight_previous = {}
        self.lamda = lamda

    def create_state(self, index, weight):
        if self.momentum == 0.0:
            return (None,
                    weight.copy())  # previous weight
        else:
            return (zeros(weight.shape, weight.context, dtype=weight.dtype), # momentum
                    weight.copy())  # previous weight

    def update(self, index, weight, grad, state):
        assert(isinstance(weight, NDArray))
        assert(isinstance(grad, NDArray))
        self._update_count(index)
        lr = self._get_lr(index)
        wd = self._get_wd(index)

        grad = grad * self.rescale_grad
        if self.clip_gradient is not None:
            grad = clip(grad, -self.clip_gradient, self.clip_gradient)

        mom, previous_weight = state
        if mom:
            mom[:] *= self.momentum
            mom[:] += -lr * (grad + wd * weight + self.lamda \
                             * grad * grad * (weight - previous_weight))
        else:
            assert(self.momentum == 0.0)
            mom = -lr * (grad + wd * weight + self.lamda \
                         * grad * grad * (weight - previous_weight))
        previous_weight[:] = weight
        weight[:] += mom