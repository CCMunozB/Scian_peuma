#!/usr/bin/env python3

import pyudev
import subprocess
import time

def check_device(device):
    """Check if the device is a CH341 device"""
    return device.get('ID_VENDOR_ID') == '1a86' and device.get('ID_MODEL_ID') == '7523'

def run_program(is_ch341):
    """Run the appropriate program based on device type"""
    if is_ch341:
        print("CH341 device detected - running program A")
        # Replace with your actual command for CH341 device
        subprocess.run(["python","/home/electroscian/Documents/PEUMA/Scian_peuma/peuma_bis.py"])
    else:
        print("Non-CH341 device detected - running program B")
        # Replace with your actual command for other devices
        #subprocess.run(["/path/to/program_for_others"])
        print("Device is not a CH341 device, exiting.")

def main():
    context = pyudev.Context()
    monitor = pyudev.Monitor.from_netlink(context)
    monitor.filter_by(subsystem='usb', device_type='usb_device')
    
    print("Monitoring USB 1-1 for device connections...")
    
    for device in iter(monitor.poll, None):
        # Check if this is the USB 1-1 port
        if device.device_node == '/dev/bus/usb/001/009' or '1-9' in device.device_path:
            if device.action == 'add':
                time.sleep(1)  # Give the system a moment to fully recognize the device
                is_ch341 = check_device(device)
                run_program(is_ch341)
            elif device.action == 'remove':
                print("Device disconnected from USB 1-9")

if __name__ == '__main__':
    main()