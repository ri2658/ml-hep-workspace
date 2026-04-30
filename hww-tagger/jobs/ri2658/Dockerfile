# Use PyTorch base image with CUDA support
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

RUN pip install --upgrade pip

RUN pip install numpy pandas
RUN pip install scipy scikit-learn
RUN pip install scikit-hep PyYAML tqdm

RUN pip install weaver-core>=0.4

WORKDIR /opt/repo

# Default command
CMD ["python3"]