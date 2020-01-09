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
"""Adamax optimizer."""
from __future__ import absolute_import
from ..ndarray import (NDArray, zeros, clip, sqrt, cast, maximum, abs as NDabs,
                       multi_sum_sq, multi_lars)
from .optimizer import Optimizer, register

__all__ = ['NDabs', 'Adamax']


# pylint: enable=line-too-long
@register
class Adamax(Optimizer):
    """The AdaMax optimizer.

    It is a variant of Adam based on the infinity norm
    available at http://arxiv.org/abs/1412.6980 Section 7.

    The optimizer updates the weight by::

        grad = clip(grad * rescale_grad + wd * weight, clip_gradient)
        m = beta1 * m_t + (1 - beta1) * grad
        u = maximum(beta2 * u, abs(grad))
        weight -= lr / (1 - beta1**t) * m / u

    This optimizer accepts the following parameters in addition to those accepted
    by :class:`.Optimizer`.

    Parameters
    ----------
    beta1 : float, optional
        Exponential decay rate for the first moment estimates.
    beta2 : float, optional
        Exponential decay rate for the second moment estimates.
    """
    def __init__(self, learning_rate=0.002, beta1=0.9, beta2=0.999, **kwargs):
        super(Adamax, self).__init__(learning_rate=learning_rate, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2

    def create_state(self, index, weight):
        return (zeros(weight.shape, weight.context, dtype=weight.dtype),  # mean
                zeros(weight.shape, weight.context, dtype=weight.dtype))  # variance

    def update(self, index, weight, grad, state):
        assert(isinstance(weight, NDArray))
        assert(isinstance(grad, NDArray))
        self._update_count(index)
        lr = self._get_lr(index)
        wd = self._get_wd(index)

        t = self._index_update_count[index]
        lr /= (1. - self.beta1**t)

        # preprocess grad
        grad = grad * self.rescale_grad + wd * weight
        if self.clip_gradient is not None:
            grad = clip(grad, -self.clip_gradient, self.clip_gradient)

        # update m_t and u_t
        m_t, u_t = state
        m_t[:] *= self.beta1
        m_t[:] += (1. - self.beta1) * grad
        u_t[:] = maximum(self.beta2 * u_t, NDabs(grad))

        # update weight
        weight[:] -= lr * m_t / u_t
