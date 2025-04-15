@echo off
echo Installing CUDA-enabled torch and PyG for Windows...

pip install torch==2.3.0+cu118 -f https://download.pytorch.org/whl/cu118/torch_stable.html
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv ^
  -f https://data.pyg.org/whl/torch-2.3.0+cu118.html
pip install torch-geometric==2.6.1