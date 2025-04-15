import platform
import subprocess
import os

if platform.system() == "Windows":
    script = os.path.join("install", "install_win.bat")
    print(f"Running Windows install script: {script}")
    subprocess.run([script], shell=True)
elif platform.system() == "Linux":
    script = os.path.join("install", "install_linux.sh")
    print(f"Running Linux install script: {script}")
    subprocess.run(["bash", script])
else:
    print("Unsupported OS for automatic torch installation.")
