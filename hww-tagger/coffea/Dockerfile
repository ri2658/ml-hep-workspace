FROM python:3

RUN apt-get update
RUN apt-get install -y vim screen

RUN pip install numpy matplotlib mplhep awkward uproot coffea vector 

# Set the default command to python3.
CMD ["python3"]
