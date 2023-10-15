#include "robot_types.h"

std::ostream& operator <<(std::ostream& c, JointCmd& data)
{
    c << data.pos << " " << data.kp << " " << data.vel << " " << data.kd << " " << data.tor;
    return c;
} 