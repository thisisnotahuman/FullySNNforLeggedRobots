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

import math
import numpy as np
import transformations


def getEulerFromQuaternion(q):
    q = quat2wxyz(q)
    return transformations.euler_from_quaternion(q, axes='sxyz')

def getQuaternionFromEuler(rpy):
    """
    return: wxyz
    """
    return transformations.quaternion_from_euler(*rpy)

def invertTransform(t, q):
    q = quat2wxyz(q)
    try:
        q = transformations.quaternion_inverse(q)
    except ValueError:
        print("ValueError: not a valid quaternion")
    q = quat2xyzw(q)
    return (-t[0], -t[1], -t[2]), q

def multiplyTransforms(t1, q1, t2, q2):
    q1 = quat2wxyz(q1)
    q2 = quat2wxyz(q2)
    T1 = transformations.quaternion_matrix(q1)
    T2 = transformations.quaternion_matrix(q2)
    T1[:3, 3] = t1
    T2[:3, 3] = t2
    T = T1.dot(T2)
    p = T[:3, 3]
    q = transformations.quaternion_from_matrix(T)
    q = quat2xyzw(q)
    return p, q

def getMatrixFromQuaternion(q):
    q = quat2wxyz(q)
    return transformations.quaternion_matrix(q)[:3, :3]

def quat2wxyz(q):
    return (q[3], q[0], q[1], q[2])

def quat2xyzw(q):
    return (q[1], q[2], q[3], q[0])