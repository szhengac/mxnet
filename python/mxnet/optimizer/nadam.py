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
"""Nadam optimizer."""
from __future__ import absolute_import
from ..ndarray import (NDArray, zeros, clip, sqrt, cast, multi_sum_sq, multi_lars)
from .optimizer import Optimizer, register

__all__ = ['Nadam']


@register
class Nadam(Optimizer):
    """The Nesterov Adam optimizer.

    Much like Adam is essentially RMSprop with momentum,
    Nadam is Adam RMSprop with Nesterov momentum available
    at http://cs229.stanford.edu/proj2015/054_report.pdf.

    This optimizer accepts the following parameters in addition to those accepted
    by :class:`.Optimizer`.

    Parameters
    ----------
    beta1 : float, optional
        Exponential decay rate for the first moment estimates.
    beta2 : float, optional
        Exponential decay rate for the second moment estimates.
    epsilon : float, optional
        Small value to avoid division by 0.
    schedule_decay : float, optional
        Exponential decay rate for the momentum schedule
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 schedule_decay=0.004, **kwargs):
        super(Nadam, self).__init__(learning_rate=learning_rate, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.schedule_decay = schedule_decay
        self.m_schedule = 1.

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

        # preprocess grad
        grad = grad * self.rescale_grad + wd * weight
        if self.clip_gradient is not None:
            grad = clip(grad, -self.clip_gradient, self.clip_gradient)

        # warming momentum schedule
        momentum_t = self.beta1 * (1. - 0.5 * (pow(0.96, t * self.schedule_decay)))
        momentum_t_1 = self.beta1 * (1. - 0.5 * (pow(0.96, (t + 1) * self.schedule_decay)))
        self.m_schedule = self.m_schedule * momentum_t
        m_schedule_next = self.m_schedule * momentum_t_1

        # update m_t and v_t
        m_t, v_t = state
        m_t[:] *= self.beta1
        m_t[:] += (1. - self.beta1) * grad
        v_t[:] *= self.beta2
        v_t[:] += (1. - self.beta2) * grad * grad

        grad_prime = grad / (1. - self.m_schedule)
        m_t_prime = m_t / (1. - m_schedule_next)
        v_t_prime = v_t / (1. - pow(self.beta2, t))
        m_t_bar = (1. - momentum_t) * grad_prime + momentum_t_1 * m_t_prime

        # update weight
        weight[:] -= lr * m_t_bar / (sqrt(v_t_prime) + self.epsilon)
