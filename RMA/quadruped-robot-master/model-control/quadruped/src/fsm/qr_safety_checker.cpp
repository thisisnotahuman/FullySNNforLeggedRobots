// The MIT License

// Copyright (c) 2022
// Robot Motion and Vision Laboratory at East China Normal University
// Contact: tophill.robotics@gmail.com

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "fsm/qr_safety_checker.hpp"


template <typename T>
bool qrSafetyChecker<T>::CheckSafeOrientation() {
  if (abs(data->quadruped->GetBaseRollPitchYaw()[0]) >= 0.5 ||
      abs(data->quadruped->GetBaseRollPitchYaw()[1]) >= 0.5) {
        printf("Orientation safety check failed!\n");
    return false;
  } else {
    return true;
  }
}


template <typename T>
bool qrSafetyChecker<T>::CheckPDesFoot() {

  return true;
}


template <typename T>
bool qrSafetyChecker<T>::CheckForceFeedForward() {

    bool safeForceFeedForward = true;

    /* Check commanded torque of every motor.
     * Work as clip function.
     * The value depends on users. */
    for (int leg = 0; leg < 4; leg++) {
        for (int motorId(0); motorId< 3; ++motorId) {
            if (data->legCmd[leg*3+motorId].tua>23) {
                data->legCmd[leg*3+motorId].tua = 23;
            } else if (data->legCmd[leg*3+motorId].tua<-23) {
                data->legCmd[leg*3+motorId].tua = -23;
            }
        }
    }

    return safeForceFeedForward;
}


template class qrSafetyChecker<float>;
