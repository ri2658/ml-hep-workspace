FROM gitlab-registry.nrp-nautilus.io/prp/jupyter-stack/prp:latest

LABEL maintainer="Raghav Kansal <rkansal@ucsd.edu>"

RUN sudo apt-get update
RUN sudo apt-get install -y vim screen wget
RUN sudo apt-get install -y gcc

RUN pip install uproot3 awkward0 lz4 xxhash tables onnxruntime-gpu
RUN pip install scikit-learn scikit-hep scipy PyYAML tqdm
RUN pip install coffea vector

RUN pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.9.0+cu102.html

# Set the default command to python3.
CMD ["python3"]
