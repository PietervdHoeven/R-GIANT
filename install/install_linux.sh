# Install PyTorch with CUDA 11.8 support
echo "Installing PyTorch 2.3.0 with CUDA 11.8..."
pip install torch==2.3.0+cu118 torchvision==0.18.0+cu118 torchaudio==2.3.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# Install PyTorch Geometric and extensions for CUDA 11.8
echo "Installing torch-geometric extensions..."
pip install torch-scatter==2.1.2+pt23cu118 -f https://data.pyg.org/whl/torch-2.3.0+cu118.html
pip install torch-sparse==0.6.18+pt23cu118 -f https://data.pyg.org/whl/torch-2.3.0+cu118.html
pip install torch-cluster==1.6.3+pt23cu118 -f https://data.pyg.org/whl/torch-2.3.0+cu118.html
pip install torch-spline-conv==1.2.2+pt23cu118 -f https://data.pyg.org/whl/torch-2.3.0+cu118.html
pip install torch-geometric==2.6.1