#!/bin/bash
# USB Mass Storage Gadget Setup Script for RPi Zero 2W (Bookworm 64-bit)

set -e  # Exit on any error

echo "=== Setting up USB Mass Storage Gadget ==="

# Configuration
IMAGE_FILE="/usb.img"
IMAGE_SIZE_MB=32K  # Size of the image in MB

# Check if running on Raspberry Pi Zero 2W
if ! grep -q "Zero 2" /proc/device-tree/model 2>/dev/null; then
    echo "Warning: This script is designed for Raspberry Pi Zero 2W"
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create the disk image if it doesn't exist
if [ ! -f "$IMAGE_FILE" ]; then
    echo "Creating disk image ($IMAGE_SIZE_MB MB)..."
    sudo dd if=/dev/zero of="$IMAGE_FILE" bs=1M count=$IMAGE_SIZE_MB status=progress
    sudo chmod +777 "$IMAGE_FILE"
    echo "Formatting as FAT32..."
    sudo mkfs.vfat "$IMAGE_FILE"
else
    echo "Disk image already exists at $IMAGE_FILE"
fi

# Clean up any existing gadget configurations
echo "Cleaning up existing gadget configurations..."
sudo rmmod g_mass_storage 2>/dev/null || true
sudo rmmod libcomposite 2>/dev/null || true
sudo rmmod dwc2 2>/dev/null || true

# Remove any existing gadget directory
if [ -d "/sys/kernel/config/usb_gadget/pi_zero" ]; then
    echo "Removing existing gadget..."
    sudo rmdir /sys/kernel/config/usb_gadget/pi_zero/configs/c.1/mass_storage.usb0 2>/dev/null || true
    sudo rmdir /sys/kernel/config/usb_gadget/pi_zero/configs/c.1/strings/0x409 2>/dev/null || true
    sudo rmdir /sys/kernel/config/usb_gadget/pi_zero/configs/c.1 2>/dev/null || true
    sudo rmdir /sys/kernel/config/usb_gadget/pi_zero/functions/mass_storage.usb0 2>/dev/null || true
    sudo rmdir /sys/kernel/config/usb_gadget/pi_zero/strings/0x409 2>/dev/null || true
    sudo rm -rf /sys/kernel/config/usb_gadget/pi_zero 2>/dev/null || true
fi

# Wait a moment for cleanup to complete
sleep 2

# Try the simple method first (most reliable on Bookworm)
echo "Attempting simple method (g_mass_storage module)..."
if sudo modprobe g_mass_storage file="$IMAGE_FILE" stall=0 removable=1; then
    echo "✓ Successfully loaded g_mass_storage module"
    echo "The USB gadget should now be active. Check your PC for the new drive."
    exit 0
fi

echo "Simple method failed, trying libcomposite method..."

# Load necessary modules
sudo modprobe libcomposite
sudo modprobe usb_f_mass_storage

# Create gadget structure
sudo mkdir -p /sys/kernel/config/usb_gadget/pi_zero
cd /sys/kernel/config/usb_gadget/pi_zero

# Set USB identifiers
echo 0x1d6b | sudo tee idVendor > /dev/null
echo 0x0104 | sudo tee idProduct > /dev/null
echo 0x0200 | sudo tee bcdUSB > /dev/null
echo 0x0100 | sudo tee bcdDevice > /dev/null

# Create strings directories and files
sudo mkdir -p strings/0x409
echo "1234567890" | sudo tee strings/0x409/serialnumber > /dev/null
echo "Raspberry Pi" | sudo tee strings/0x409/manufacturer > /dev/null
echo "Pi Zero 2W Mass Storage" | sudo tee strings/0x409/product > /dev/null

# Create configuration
sudo mkdir -p configs/c.1/strings/0x409
echo "Mass Storage" | sudo tee configs/c.1/strings/0x409/configuration > /dev/null
echo 250 | sudo tee configs/c.1/MaxPower > /dev/null

# Create mass storage function
sudo mkdir -p functions/mass_storage.usb0
echo 0 | sudo tee functions/mass_storage.usb0/lun.0/cdrom > /dev/null
echo 0 | sudo tee functions/mass_storage.usb0/lun.0/ro > /dev/null
echo 0 | sudo tee functions/mass_storage.usb0/lun.0/nofua > /dev/null
echo "$IMAGE_FILE" | sudo tee functions/mass_storage.usb0/lun.0/file > /dev/null

# Link function to configuration
sudo ln -s functions/mass_storage.usb0 configs/c.1/

# Enable the gadget
UDC=$(ls /sys/class/udc | head -n1)
if [ -n "$UDC" ]; then
    echo "$UDC" | sudo tee UDC > /dev/null
    echo "✓ Gadget successfully activated using $UDC"
else
    echo "✗ Error: No USB Device Controller found"
    exit 1
fi

echo "=== Setup complete! ==="
echo "The Pi should now appear as a USB drive on your PC."
echo "Image file: $IMAGE_FILE"
