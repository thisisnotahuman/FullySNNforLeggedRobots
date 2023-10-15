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

from absl import app
from absl import flags
from inputs import get_gamepad
import threading
import time

FLAGS = flags.FLAGS
MAX_ABS_RX = 32768
MAX_ABS_RY = 32768


def _interpolate(raw_reading, max_raw_reading, new_scale):
    return raw_reading / max_raw_reading * new_scale


class Gamepad:
    """Interface for reading commands from Logitech F710 Gamepad.
      The control works as following:
      1) Press LB+RB at any time for emergency stop
      2) Use the left joystick for forward/backward/left/right walking.
      3) Use the right joystick for rotation around the z-axis.
    """
    def __init__(self, vel_scale_x=.4, vel_scale_y=.4, vel_scale_rot=1.):
        """Initialize the gamepad controller.
      Args:
        vel_scale_x: maximum absolute x-velocity command.
        vel_scale_y: maximum absolute y-velocity command.
        vel_scale_rot: maximum absolute yaw-dot command.
      """
        self._vel_scale_x = vel_scale_x
        self._vel_scale_y = vel_scale_y
        self._vel_scale_rot = vel_scale_rot
        self._lb_pressed = False
        self._rb_pressed = False

        # Controller states
        self.vx, self.vy, self.wz = 0., 0., 0.
        self.estop_flagged = False
        self.is_running = True

        self.read_thread = threading.Thread(target=self.read_loop)
        self.read_thread.start()

    def read_loop(self):
        """The read loop for events.
        This funnction should be executed in a separate thread for continuous
        event recording.
        """
        while self.is_running and not self.estop_flagged:
            events = get_gamepad()
            for event in events:
                self.update_command(event)

    def update_command(self, event):
        """Update command based on event readings."""
        if event.ev_type == 'Key' and event.code == 'BTN_TL':
            self._lb_pressed = bool(event.state)
        elif event.ev_type == 'Key' and event.code == 'BTN_TR':
            self._rb_pressed = bool(event.state)
        elif event.ev_type == 'Absolute' and event.code == 'ABS_X':
            # Left Joystick L/R axis
            self.vy = _interpolate(-event.state, MAX_ABS_RX, self._vel_scale_y)
        elif event.ev_type == 'Absolute' and event.code == 'ABS_Y':
            # Left Joystick F/B axis; need to flip sign for consistency
            self.vx = _interpolate(-event.state, MAX_ABS_RY, self._vel_scale_x)
        elif event.ev_type == 'Absolute' and event.code == 'ABS_RX':
            self.wz = _interpolate(event.state, MAX_ABS_RX,
                                   self._vel_scale_rot)

        if self._lb_pressed and self._rb_pressed:
            self.estop_flagged = True
            self.vx, self.vy, self.wz = 0., 0., 0.

    def get_command(self, time_since_reset):
        del time_since_reset  # unused
        return (1.0 * self.vx, 0.5 * self.vy,
                0), -1.5 * self.wz, self.estop_flagged

    def stop(self):
        self.is_running = False


def main(_):
    gamepad = Gamepad()
    while True:
        print("Vx: {}, Vy: {}, Wz: {}, Estop: {}".format(
            gamepad.vx, gamepad.vy, gamepad.wz, gamepad.estop_flagged))
        time.sleep(0.1)


if __name__ == "__main__":
    app.run(main)
