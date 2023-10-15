# SPDX-FileCopyrightText: Copyright (c) 2023, HUAWEI TECHNOLOGIES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""Moving window filter to smooth out sensor readings."""

import collections


class MovingWindowFilter(object):
    """A stable O(1) moving filter for incoming data streams.
  We implement the Neumaier's algorithm to calculate the moving window average,
  which is numerically stable.
  """

    def __init__(self, window_size: int):
        """Initializes the class.
    Args:
      window_size: The moving window size.
    """
        assert window_size > 0
        self._window_size = window_size
        self._value_deque = collections.deque(maxlen=window_size)
        # The moving window sum.
        self._sum = 0
        # The correction term to compensate numerical precision loss during calculation.
        self._correction = 0

    def _neumaier_sum(self, value: float):
        """Update the moving window sum using Neumaier's algorithm.

    For more details please refer to:
    https://en.wikipedia.org/wiki/Kahan_summation_algorithm#Further_enhancements

    Args:
      value: The new value to be added to the window.
    """

        new_sum = self._sum + value
        if abs(self._sum) >= abs(value):
            # If self._sum is bigger, low-order digits of value are lost.
            self._correction += (self._sum - new_sum) + value
        else:
            # low-order digits of sum are lost
            self._correction += (value - new_sum) + self._sum

        self._sum = new_sum

    def calculate_average(self, new_value: float) -> float:
        """Computes the moving window average in O(1) time.

    Args:
      new_value: The new value to enter the moving window.

    Returns:
      The average of the values in the window.

    """
        deque_len = len(self._value_deque)
        if deque_len < self._value_deque.maxlen:
            pass
        else:
            # The left most value to be subtracted from the moving sum.
            self._neumaier_sum(-self._value_deque[0])

        self._neumaier_sum(new_value)
        self._value_deque.append(new_value)

        return (self._sum + self._correction) / self._window_size
