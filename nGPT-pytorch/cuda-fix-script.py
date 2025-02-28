#!/usr/bin/env python3
"""
CUDA Diagnostic and Fix Script
This script diagnoses CUDA issues and attempts to fix common problems that
might occur when moving a drive between machines.
"""

import os
import sys
import subprocess
import platform
import shutil
import json
from pathlib import Path
import re

def print_header(message):
    """Print a formatted header message."""
    print("\n" + "=" * 80)
    print(f" {message} ".center(80, "="))
    print("=" * 80)

def run_command(command, desc=None):
    """Run a shell command and return the output."""
    if desc:
        print(f"\n> {desc}")
        print(f"  $ {command}")
    
    try:
        result = subprocess.run(
            command, shell=True, check=False, 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True
        )
        if result.stdout:
            print(result.stdout.strip())
        if result.stderr:
            print(f"STDERR: {result.stderr.strip()}")
        return result
    except Exception as e:
        print(f"Error executing '{command}': {e}")
        return None

def check_system_info():
    """Check and display system information."""
    print_header("SYSTEM INFORMATION")
    print(f"Platform: {platform.platform()}")
    print(f"Python version: {sys.version}")
    run_command("nvidia-smi", "Checking NVIDIA Driver with nvidia-smi")
    run_command("lspci | grep -i nvidia", "Checking for NVIDIA devices with lspci")
    run_command("lsmod | grep nvidia", "Checking loaded NVIDIA kernel modules")
    run_command("grep -i nvidia /var/log/syslog | tail -20", "Checking system logs for NVIDIA messages")

def check_cuda_environment():
    """Check CUDA-related environment variables and paths."""
    print_header("CUDA ENVIRONMENT")
    
    # Check common CUDA environment variables
    for var in ['CUDA_VISIBLE_DEVICES', 'LD_LIBRARY_PATH', 'CUDA_HOME', 'PATH']:
        print(f"{var}={os.environ.get(var, 'not set')}")
    
    # Check for CUDA installations
    possible_cuda_paths = [
        "/usr/local/cuda",
        "/usr/lib/cuda",
        "/opt/cuda",
    ]
    
    print("\nDetected CUDA installations:")
    for path in possible_cuda_paths:
        if os.path.exists(path):
            version_file = Path(path) / "version.txt"
            if version_file.exists():
                with open(version_file, "r") as f:
                    version = f.read().strip()
                print(f"  {path}: {version}")
            else:
                print(f"  {path}: (version file not found)")
        
    # Find all libcuda.so files
    run_command("find /usr -name 'libcuda.so*' 2>/dev/null", "Locating CUDA libraries")
    
    # Check PyTorch CUDA detection
    print("\nPyTorch CUDA detection:")
    run_command("""python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('Device count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)" """)

def check_pytorch_config():
    """Check PyTorch configuration."""
    print_header("PYTORCH CONFIGURATION")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"PyTorch config:")
        for k, v in torch.__config__.show().items():
            print(f"  {k}: {v}")
            
        # Check how PyTorch was built
        build_info = {
            "CUDA built": torch.backends.cuda.is_built(),
            "CUDA available": torch.cuda.is_available(),
            "cuDNN built": torch.backends.cudnn.is_available(),
            "cuDNN version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "N/A"
        }
        
        for k, v in build_info.items():
            print(f"  {k}: {v}")
            
    except ImportError:
        print("PyTorch is not installed.")
    except Exception as e:
        print(f"Error checking PyTorch configuration: {e}")

def fix_cuda_visible_devices():
    """Fix CUDA_VISIBLE_DEVICES environment variable."""
    print_header("FIXING CUDA_VISIBLE_DEVICES")
    
    # Check if variable is set and needs fixing
    current_value = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    print(f"Current CUDA_VISIBLE_DEVICES: {current_value}")
    
    # Try to determine available devices
    try:
        result = subprocess.run(
            "nvidia-smi --list-gpus | wc -l", 
            shell=True, check=True, 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True
        )
        gpu_count = int(result.stdout.strip())
        print(f"Detected {gpu_count} GPU(s)")
        
        if gpu_count > 0:
            # Create a list of device indices
            device_indices = list(range(gpu_count))
            new_value = ",".join(map(str, device_indices))
            
            # Set the environment variable for this process
            os.environ['CUDA_VISIBLE_DEVICES'] = new_value
            print(f"Setting CUDA_VISIBLE_DEVICES to '{new_value}' for this session")
            
            # Instructions for the user
            print("\nTo fix this permanently, add the following to your ~/.bashrc file:")
            print(f"export CUDA_VISIBLE_DEVICES=\"{new_value}\"")
            
            # Attempt to add to .bashrc if user agrees
            response = input("\nWould you like me to add this to your ~/.bashrc? (y/n): ")
            if response.lower() == 'y':
                bashrc_path = os.path.expanduser("~/.bashrc")
                with open(bashrc_path, "a") as f:
                    f.write(f"\n# Added by CUDA fix script\n")
                    f.write(f"export CUDA_VISIBLE_DEVICES=\"{new_value}\"\n")
                print(f"Added to {bashrc_path}")
            
            return True
        else:
            print("No GPUs detected with nvidia-smi. Cannot fix CUDA_VISIBLE_DEVICES.")
            return False
            
    except Exception as e:
        print(f"Error fixing CUDA_VISIBLE_DEVICES: {e}")
        return False

def fix_driver_mismatch():
    """Attempt to fix driver/CUDA version mismatches."""
    print_header("CHECKING DRIVER/CUDA COMPATIBILITY")
    
    # Get current driver version
    try:
        result = subprocess.run(
            "nvidia-smi --query-gpu=driver_version --format=csv,noheader", 
            shell=True, check=True, 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True
        )
        driver_version = result.stdout.strip()
        print(f"Current NVIDIA driver version: {driver_version}")
        
        # Get PyTorch CUDA version
        import torch
        torch_cuda_version = torch.version.cuda if torch.cuda.is_available() else None
        print(f"PyTorch CUDA version: {torch_cuda_version}")
        
        if not torch_cuda_version:
            print("PyTorch was not built with CUDA support or CUDA is not available.")
            print("Consider reinstalling PyTorch with CUDA support:")
            print("  conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia")
            print("  or")
            print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            return False
            
        # Check compatibility (this is a simplified check)
        cuda_major = int(torch_cuda_version.split('.')[0]) if torch_cuda_version else 0
        driver_major = float(driver_version.split('.')[0]) if driver_version else 0
        
        if cuda_major == 11 and driver_major < 450:
            print("WARNING: CUDA 11.x requires NVIDIA driver >= 450.x")
            print("Consider upgrading your NVIDIA driver.")
        elif cuda_major == 12 and driver_major < 525:
            print("WARNING: CUDA 12.x requires NVIDIA driver >= 525.x")
            print("Consider upgrading your NVIDIA driver.")
            
        return True
        
    except Exception as e:
        print(f"Error checking driver compatibility: {e}")
        return False

def fix_train_script():
    """Fix the train.py script to disable AMP if CUDA is not available."""
    print_header("FIXING TRAIN.PY SCRIPT")
    
    script_path = os.path.expanduser("~/Desktop/toroidGPT/nGPT-pytorch/train.py")
    
    if not os.path.exists(script_path):
        print(f"Script not found at {script_path}")
        script_path = input("Please enter the full path to train.py: ")
        if not os.path.exists(script_path):
            print("Script not found. Cannot fix.")
            return False
    
    print(f"Fixing {script_path}")
    
    # Create a backup
    backup_path = f"{script_path}.bak"
    shutil.copy2(script_path, backup_path)
    print(f"Created backup at {backup_path}")
    
    # Read the script
    with open(script_path, "r") as f:
        content = f.read()
    
    # Look for the assert line and replace it
    if "assert not (USE_AMP and not torch.cuda.is_available())" in content:
        new_content = content.replace(
            "assert not (USE_AMP and not torch.cuda.is_available())",
            "if USE_AMP and not torch.cuda.is_available():\n    print('WARNING: AMP enabled but CUDA not available. Disabling AMP.')\n    USE_AMP = False"
        )
        
        # Write modified script
        with open(script_path, "w") as f:
            f.write(new_content)
        
        print("Successfully modified train.py to handle missing CUDA gracefully")
        return True
    else:
        print("Could not find the assert line in train.py. Manual inspection needed.")
        return False

def check_ollama():
    """Check Ollama installation and configuration."""
    print_header("CHECKING OLLAMA")
    
    run_command("which ollama", "Checking if ollama is installed")
    run_command("ollama --version", "Checking ollama version")
    
    # Check if ollama service is running
    run_command("systemctl status ollama 2>&1 || echo 'Ollama service not found'", "Checking ollama service status")
    
    # Check GPU usage with ollama
    print("\nChecking GPU usage with Ollama:")
    run_command("ollama list", "Listing ollama models")
    
    # Try to get Ollama's GPU configuration
    try:
        run_command("cat ~/.ollama/config.json 2>/dev/null || echo 'Config file not found'", "Checking Ollama config")
    except:
        print("Could not check Ollama config")

def update_system_gpu_config():
    """Update system GPU configuration."""
    print_header("UPDATING SYSTEM GPU CONFIGURATION")
    
    # Check if blacklist file exists and disable it
    blacklist_file = "/etc/modprobe.d/blacklist-nvidia.conf"
    if os.path.exists(blacklist_file):
        print(f"Found {blacklist_file} - this might be blocking the NVIDIA driver")
        backup_file = f"{blacklist_file}.bak"
        
        response = input(f"Would you like to backup and rename this file? (y/n): ")
        if response.lower() == 'y':
            try:
                shutil.copy2(blacklist_file, backup_file)
                os.rename(blacklist_file, f"{blacklist_file}.disabled")
                print(f"Renamed {blacklist_file} to {blacklist_file}.disabled")
                print(f"Created backup at {backup_file}")
            except Exception as e:
                print(f"Error disabling blacklist file: {e}")
                print("Try running this script with sudo")
    
    # Check if nvidia modules are loaded
    modules_loaded = run_command("lsmod | grep -c nvidia", "Checking if NVIDIA modules are loaded")
    if modules_loaded and modules_loaded.stdout.strip() == "0":
        print("NVIDIA modules are not loaded. Attempting to load them...")
        
        response = input("Would you like to try loading NVIDIA modules? (y/n): ")
        if response.lower() == 'y':
            try:
                run_command("sudo modprobe nvidia", "Loading nvidia module")
                run_command("sudo modprobe nvidia_uvm", "Loading nvidia_uvm module")
            except Exception as e:
                print(f"Error loading modules: {e}")
                print("Try running this script with sudo")

def fix_recommendations():
    """Provide recommendations based on diagnostics."""
    print_header("RECOMMENDATIONS")
    
    print("""
1. Update NVIDIA drivers:
   sudo apt update
   sudo apt install --reinstall nvidia-driver-{version}
   
2. Check CUDA installation:
   If your PyTorch requires a specific CUDA version, install it:
   wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
   sudo sh cuda_11.8.0_520.61.05_linux.run

3. Reinstall PyTorch with the correct CUDA version:
   # For CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   # OR with conda
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

4. If your script uses AMP (Automatic Mixed Precision) but CUDA is unavailable:
   Modify the script to fallback to CPU when CUDA is unavailable.

5. Verify your GPU is supported by your CUDA version
   Check compatibility at: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
    """)

def main():
    """Main function."""
    print_header("CUDA DIAGNOSTIC AND FIX SCRIPT")
    print("This script will diagnose CUDA issues and attempt to fix them.")
    
    # Run diagnostics
    check_system_info()
    check_cuda_environment()
    check_pytorch_config()
    check_ollama()
    
    # Ask before attempting fixes
    response = input("\nWould you like to attempt automatic fixes? (y/n): ")
    if response.lower() != 'y':
        print("Exiting without applying fixes.")
        return
    
    # Apply fixes
    fix_cuda_visible_devices()
    fix_driver_mismatch()
    fix_train_script()
    update_system_gpu_config()
    
    # Final recommendations
    fix_recommendations()
    
    print_header("FIX SCRIPT COMPLETED")
    print("""
Next steps:
1. Source your updated environment: source ~/.bashrc
2. Restart your system if driver changes were made
3. Try running your script again
""")

if __name__ == "__main__":
    main()
