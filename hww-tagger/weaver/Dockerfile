FROM anibali/pytorch:1.8.1-cuda11.1

RUN sudo apt-get update
RUN sudo apt-get install -y vim screen

RUN pip install numpy pandas 
RUN pip install uproot3 awkward0 lz4 xxhash tables onnxruntime-gpu
RUN pip install scikit-learn scikit-hep scipy PyYAML tqdm 
RUN pip install coffea vector

RUN pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://pytorch-geometric.com/whl/torch-1.8.1+cu111.html

# Set the default command to python3.
CMD ["python3"]
