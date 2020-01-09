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
"""FTRL optimizer."""
from __future__ import absolute_import
from ..ndarray import (NDArray, zeros, clip, sqrt, cast, multi_sum_sq, multi_lars)
from ..ndarray import ftrl_update
from .optimizer import Optimizer, register

__all__ = ['Ftrl']


#pylint: disable=invalid-name
#pylint: disable=line-too-long
@register
class Ftrl(Optimizer):
    """The Ftrl optimizer.

    Referenced from *Ad Click Prediction: a View from the Trenches*, available at
    http://dl.acm.org/citation.cfm?id=2488200.

    eta :
        .. math::
           \\eta_{t,i} = \\frac{learningrate}{\\beta+\\sqrt{\\sum_{s=1}^tg_{s,i}^2}}

    The optimizer updates the weight by::

        rescaled_grad = clip(grad * rescale_grad, clip_gradient)
        z += rescaled_grad - (sqrt(n + rescaled_grad**2) - sqrt(n)) * weight / learning_rate
        n += rescaled_grad**2
        w = (sign(z) * lamda1 - z) / ((beta + sqrt(n)) / learning_rate + wd) * (abs(z) > lamda1)

    If the storage types of weight, state and grad are all ``row_sparse``, \
    **sparse updates** are applied by::

        for row in grad.indices:
            rescaled_grad[row] = clip(grad[row] * rescale_grad, clip_gradient)
            z[row] += rescaled_grad[row] - (sqrt(n[row] + rescaled_grad[row]**2) - sqrt(n[row])) * weight[row] / learning_rate
            n[row] += rescaled_grad[row]**2
            w[row] = (sign(z[row]) * lamda1 - z[row]) / ((beta + sqrt(n[row])) / learning_rate + wd) * (abs(z[row]) > lamda1)

    The sparse update only updates the z and n for the weights whose row_sparse
    gradient indices appear in the current batch, rather than updating it for all
    indices. Compared with the original update, it can provide large
    improvements in model training throughput for some applications. However, it
    provides slightly different semantics than the original update, and
    may lead to different empirical results.

    For details of the update algorithm, see :class:`~mxnet.ndarray.ftrl_update`.

    This optimizer accepts the following parameters in addition to those accepted
    by :class:`.Optimizer`.

    Parameters
    ----------
    lamda1 : float, optional
        L1 regularization coefficient.
    learning_rate : float, optional
        The initial learning rate.
    beta : float, optional
        Per-coordinate learning rate correlation parameter.
    """

    def __init__(self, lamda1=0.01, learning_rate=0.1, beta=1, **kwargs):
        super(Ftrl, self).__init__(**kwargs)
        self.lamda1 = lamda1
        self.beta = beta
        self.lr = learning_rate

    def create_state(self, index, weight):
        return (zeros(weight.shape, weight.context, stype=weight.stype),  # z
                zeros(weight.shape, weight.context, stype=weight.stype))  # n

    def update(self, index, weight, grad, state):
        assert(isinstance(weight, NDArray))
        assert(isinstance(grad, NDArray))
        self._update_count(index)
        wd = self._get_wd(index)
        lr = self._get_lr(index)

        kwargs = {'lamda1': self.lamda1, 'beta': self.beta, 'rescale_grad': self.rescale_grad}
        if self.clip_gradient:
            kwargs['clip_gradient'] = self.clip_gradient

        # accumulated g and delta initialization
        z, n = state
        ftrl_update(weight, grad, z, n, out=weight,
                    lr=lr, wd=wd, **kwargs)