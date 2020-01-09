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
"""AdaDelta optimizer."""
from __future__ import absolute_import
from ..ndarray import (NDArray, zeros, clip, sqrt, cast, multi_sum_sq, multi_lars)
from .optimizer import Optimizer, register

__all__ = ['AdaDelta']


@register
class AdaDelta(Optimizer):
    """The AdaDelta optimizer.

    This class implements AdaDelta, an optimizer described in  *ADADELTA: An adaptive
    learning rate method*, available at https://arxiv.org/abs/1212.5701.

    This optimizer updates each weight by::

        grad = clip(grad * rescale_grad + wd * weight, clip_gradient)
        acc_grad = rho * acc_grad + (1. - rho) * grad * grad
        delta = sqrt(acc_delta + epsilon) / sqrt(acc_grad + epsilon) * grad
        acc_delta = rho * acc_delta + (1. - rho) * delta * delta
        weight -= (delta + wd * weight)

    This optimizer accepts the following parameters in addition to those accepted
    by :class:`.Optimizer`.

    Parameters
    ----------
    rho: float
        Decay rate for both squared gradients and delta.
    epsilon : float
        Small value to avoid division by 0.
    """
    def __init__(self, rho=0.90, epsilon=1e-5, **kwargs):
        super(AdaDelta, self).__init__(**kwargs)
        self.rho = rho
        self.epsilon = epsilon

    def create_state(self, index, weight):
        return (zeros(weight.shape, weight.context),  # accumulated g
                zeros(weight.shape, weight.context))  # accumulated delta

    def update(self, index, weight, grad, state):
        assert(isinstance(weight, NDArray))
        assert(isinstance(grad, NDArray))
        wd = self._get_wd(index)
        self._update_count(index)

        # preprocess grad
        grad *= self.rescale_grad
        if self.clip_gradient is not None:
            grad = clip(grad, - self.clip_gradient, self.clip_gradient)

        # accumulated g and delta initlization
        acc_g, acc_delta = state

        # update g, delta
        acc_g[:] *= self.rho
        acc_g[:] += (1. - self.rho) * grad * grad
        current_delta = sqrt(acc_delta + self.epsilon) / sqrt(acc_g + self.epsilon) * grad
        acc_delta[:] *= self.rho
        acc_delta[:] += (1. - self.rho) * current_delta * current_delta

        # update weight
        weight[:] -= current_delta + wd * weight