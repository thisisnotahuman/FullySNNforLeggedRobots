#!/bin/bash
# Set GPU frequency to maximum
sudo echo "***** Set GPU frequency to maximum *****"
sudo echo "- Previous frequency of GPU"
sudo cat /sys/devices/17000000.gv11b/devfreq/17000000.gv11b/cur_freq
sudo echo "- The list of available frequencies"
sudo cat /sys/devices/17000000.gv11b/devfreq/17000000.gv11b/available_frequencies
sudo echo 1109250000 > /sys/devices/17000000.gv11b/devfreq/17000000.gv11b/max_freq
sudo echo 1109250000 > /sys/devices/17000000.gv11b/devfreq/17000000.gv11b/min_freq
sudo echo "- Changed frequency of GPU"
sudo cat /sys/devices/17000000.gv11b/devfreq/17000000.gv11b/cur_freq

