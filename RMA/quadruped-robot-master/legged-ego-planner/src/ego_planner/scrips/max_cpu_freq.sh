# Set CPU frequency to maximum
sudo echo "***** Set CPU frequency to maximum *****"
sudo echo "- Previous frequency of cores"
sudo cat /sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq
sudo echo "- Previous CPU governor of cores"
sudo cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor
sudo echo

sudo echo "- The list of available frequencies"
sudo cat /sys/devices/system/cpu/cpufreq/policy0/scaling_available_frequencies
sudo echo userspace > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor
sudo echo 2265600 > /sys/devices/system/cpu/cpufreq/policy0/scaling_setspeed
sudo echo 115200 > /sys/devices/system/cpu/cpufreq/policy0/scaling_min_freq
sudo echo 2265600 > /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq

sudo echo
sudo echo "- Changed frequency of cores"
sudo cat /sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq
sudo echo "- Changed CPU governor of cores"
sudo cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor

