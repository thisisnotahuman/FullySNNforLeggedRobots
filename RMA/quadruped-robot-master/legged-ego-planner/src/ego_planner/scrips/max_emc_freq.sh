#!/bin/bash
# Set EMC frequency to maximum
sudo echo "***** Set EMC frequency to maximum *****"
sudo echo "- Previous frequency of EMC"
sudo cat /sys/kernel/debug/bpmp/debug/clk/emc/rate
sudo echo

sudo echo 1 > /sys/kernel/debug/bpmp/debug/clk/emc/mrq_rate_locked
sudo echo 1 > /sys/kernel/debug/bpmp/debug/clk/emc/state
sudo cat /sys/kernel/debug/bpmp/debug/clk/emc/max_rate
sudo echo 2133000000 > /sys/kernel/debug/bpmp/debug/clk/emc/rate

sudo echo "- Changed frequency of EMC"
sudo cat /sys/kernel/debug/bpmp/debug/clk/emc/rate

