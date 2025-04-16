R-GIANT INSTALLATION AND USAGE GUIDE

This guide explains how to install and run the R-GIANT package for connectome extraction and graph generation using diffusion and structural MRI data. It covers setting up the Python environment, installing dependencies including GPU-accelerated PyTorch and its extensions, and running the command line interface for data preprocessing and graph assembly.


SECTION 1: SYSTEM REQUIREMENTS


- Python 3.11 (recommended)

- Windows or Linux system with NVIDIA GPU

- Compatible NVIDIA driver (for CUDA 11.8: driver version >= 520.61.05)

- Git installed (optional but useful for cloning the repository)


SECTION 2: CLONE OR DOWNLOAD THE REPOSITORY


You can either clone the repository or download it as a ZIP and extract it.

To clone:

git clone https://github.com/PietervdHoeven/R-GIANT.git cd R-GIANT


SECTION 3: SET UP A PYTHON VIRTUAL ENVIRONMENT


To avoid dependency conflicts, create a virtual environment for the project.

On Windows:

python -m venv rgiant-venv rgiant-venv\Scripts\activate

On Linux/macOS:

python3 -m venv rgiant-venv source rgiant-venv/bin/activate


SECTION 4: INSTALL GPU-ACCELERATED TORCH AND EXTENSIONS


Install PyTorch and all necessary geometric libraries compiled for CUDA 11.8.

This can be done with either:

python scripts/torch_install.py

OR:

pip install torch==2.3.0+cu118 torchvision==0.18.0+cu118 torchaudio==2.3.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html

pip install torch-scatter==2.1.2+pt23cu118 -f https://data.pyg.org/whl/torch-2.3.0+cu118.html pip install torch-sparse==0.6.18+pt23cu118 -f https://data.pyg.org/whl/torch-2.3.0+cu118.html pip install torch-cluster==1.6.3+pt23cu118 -f https://data.pyg.org/whl/torch-2.3.0+cu118.html pip install torch-spline-conv==1.2.2+pt23cu118 -f https://data.pyg.org/whl/torch-2.3.0+cu118.html pip install torch-geometric==2.6.1

These wheels include the necessary CUDA binaries, so there is no need to install the full CUDA Toolkit. Just make sure the NVIDIA driver on your system is compatible with CUDA 11.8. Both approaches work for windows and linux.


SECTION 5: INSTALL REQUIRED PYTHON DEPENDENCIES


Install the package and all dependencies. Run this command from the R-GIANT root directory:

pip install -e .

This installs R-GIANT in editable mode along with core dependencies defined in pyproject.toml.


SECTION 6: OPTIONAL - VERIFY INSTALLATION


To verify that everything was installed successfully, run:

pip list

You should see rgiant, torch, dipy, nibabel, HD-BET, and other packages listed.

You can test whether you are CUDA enabled by opening a python console and running the following:

import torch
print(torch.cuda.is_available())        # Should return True
print(torch.cuda.get_device_name(0))    # Should return the GPU device name


SECTION 8: USAGE


Once installed, you can use the command line interface provided by cli.py.

Example: run cleaning pipeline

rgiant-cli clean --participant-id 0001 --session-id 0757 --data-dir /path/to/data --clear-temp

Example: run connectome pipeline

rgiant-cli connectome --participant-id 0001 --session-id 0757 --data-dir /path/to/data

Example: extract node features

rgiant-cli nodes --participant-id 0001 --session-id 0757 --data-dir /path/to/data

Example: assemble graph

rgiant-cli graph --participant-id 0001 --session-id 0757 --data-dir /path/to/data

To view all available commands:

rgiant-cli --help

To view help for a specific subcommand:

rgiant-cli connectome --help


SECTION 9: CLEAN UNINSTALL


To deactivate and remove the virtual environment:

deactivate (in your terminal)

Then delete the virtual environment folder:

rm -r rgiant-venv (Linux/macOS) rmdir /s rgiant-venv (Windows)