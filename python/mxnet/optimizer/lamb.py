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
"""Lamb optimizer."""
from __future__ import absolute_import
import numpy
from ..ndarray import (NDArray, zeros, clip, sqrt, cast, multi_sum_sq, multi_lars)
from ..ndarray import (lamb_update_phase1, lamb_update_phase2,
                       mp_lamb_update_phase1, mp_lamb_update_phase2)
from .optimizer import Optimizer, register

__all__ = ['LAMB']

@register
class LAMB(Optimizer):
    """LAMB Optimizer.
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-6,
                 lower_bound=None, upper_bound=None, bias_correction=True, **kwargs):
        super(LAMB, self).__init__(learning_rate=learning_rate, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.bias_correction = bias_correction


    def create_state(self, index, weight):
        stype = weight.stype
        return (zeros(weight.shape, weight.context, dtype=numpy.float32, stype=stype),
                zeros(weight.shape, weight.context, dtype=numpy.float32, stype=stype))

    def _update_impl(self, index, weight, grad, state, multi_precision=False):
        assert(isinstance(weight, NDArray))
        assert(isinstance(grad, NDArray))
        self._update_count(index)
        lr = self._get_lr(index)
        wd = self._get_wd(index)
        t = self._index_update_count[index]

        kwargs = {'beta1': self.beta1, 'beta2': self.beta2, 'epsilon': self.epsilon,
                  'bias_correction': self.bias_correction, 't': t,
                  'rescale_grad': self.rescale_grad}

        if self.clip_gradient:
            kwargs['clip_gradient'] = self.clip_gradient

        if multi_precision:
            mean, var = state[1]
            weight32 = state[0]
            g = mp_lamb_update_phase1(weight, grad, mean, var, weight32, wd=wd, **kwargs)

            kwargs = {}
            if self.lower_bound:
                kwargs['lower_bound'] = self.lower_bound
            if self.upper_bound:
                kwargs['upper_bound'] = self.upper_bound
            r_1 = weight32.norm()
            r_2 = g.norm()
            mp_lamb_update_phase2(weight, g, r_1, r_2, weight32, lr=lr, out=weight, **kwargs)
        else:
            mean, var = state
            g = lamb_update_phase1(weight, grad, mean, var, wd=wd, **kwargs)

            kwargs = {}
            if self.lower_bound:
                kwargs['lower_bound'] = self.lower_bound
            if self.upper_bound:
                kwargs['upper_bound'] = self.upper_bound
            r_1 = weight.norm()
            r_2 = g.norm()
            lamb_update_phase2(weight, g, r_1, r_2, lr=lr, out=weight, **kwargs)

    def update(self, index, weight, grad, state):
        self._update_impl(index, weight, grad, state, multi_precision=False)

    def update_multi_precision(self, index, weight, grad, state):
        if not isinstance(index, (tuple, list)):
            use_multi_precision = self.multi_precision and weight.dtype == numpy.float16
        else:
            use_multi_precision = self.multi_precision and weight[0].dtype == numpy.float16
        self._update_impl(index, weight, grad, state,
                          multi_precision=use_multi_precision)
