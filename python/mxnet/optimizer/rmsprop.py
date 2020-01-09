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
"""RMSProp optimizer."""
from __future__ import absolute_import
from ..ndarray import (NDArray, zeros, clip, sqrt, cast, multi_sum_sq, multi_lars)
from ..ndarray import (rmsprop_update, rmspropalex_update)
from .optimizer import Optimizer, register

__all__ = ['RMSProp']


@register
class RMSProp(Optimizer):
    """The RMSProp optimizer.

    Two versions of RMSProp are implemented:

    If ``centered=False``, we follow
    http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf by
    Tieleman & Hinton, 2012.
    For details of the update algorithm see :class:`~mxnet.ndarray.rmsprop_update`.

    If ``centered=True``, we follow http://arxiv.org/pdf/1308.0850v5.pdf (38)-(45)
    by Alex Graves, 2013.
    For details of the update algorithm see :class:`~mxnet.ndarray.rmspropalex_update`.

    This optimizer accepts the following parameters in addition to those accepted
    by :class:`.Optimizer`.

    Parameters
    ----------
    gamma1: float, optional
        A decay factor of moving average over past squared gradient.
    gamma2: float, optional
        A "momentum" factor. Only used if `centered`=``True``.
    epsilon : float, optional
        Small value to avoid division by 0.
    centered : bool, optional
        Flag to control which version of RMSProp to use.::

            True: will use Graves's version of `RMSProp`,
            False: will use Tieleman & Hinton's version of `RMSProp`.

    clip_weights : float, optional
        Clips weights into range ``[-clip_weights, clip_weights]``.
    """
    def __init__(self, learning_rate=0.001, gamma1=0.9, gamma2=0.9,
                 epsilon=1e-8, centered=False, clip_weights=None, **kwargs):
        super(RMSProp, self).__init__(learning_rate=learning_rate, **kwargs)
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.centered = centered
        self.epsilon = epsilon
        self.clip_weights = clip_weights

    def create_state(self, index, weight):
        if self.centered:
            return (
                zeros(weight.shape, weight.context, stype=weight.stype),  # n
                zeros(weight.shape, weight.context, stype=weight.stype),  # g
                zeros(weight.shape, weight.context, stype=weight.stype))  # delta
        else:
            return (zeros(weight.shape, weight.context, stype=weight.stype),)  # n

    def update(self, index, weight, grad, state):
        assert(isinstance(weight, NDArray))
        assert(isinstance(grad, NDArray))
        self._update_count(index)
        lr = self._get_lr(index)
        wd = self._get_wd(index)

        kwargs = {'gamma1': self.gamma1, 'epsilon': self.epsilon,
                  'rescale_grad': self.rescale_grad}
        if self.centered:
            kwargs['gamma2'] = self.gamma2
        if self.clip_gradient:
            kwargs['clip_gradient'] = self.clip_gradient
        if self.clip_weights:
            kwargs['clip_weights'] = self.clip_weights

        if not self.centered:
            (n, ) = state
            rmsprop_update(
                weight, grad, n, out=weight, lr=lr, wd=wd, **kwargs)
        else:
            n, g, delta = state
            rmspropalex_update(weight, grad, n, g, delta, out=weight,
                               lr=lr, wd=wd, **kwargs)