# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Install OpenVINO and other dependencies
RUN apt-get update && apt-get install -y lsb-release
RUN wget https://apt.repos.intel.com/openvino/2022/GPG-PUB-KEY-INTEL-OPENVINO-2022 && apt-key add GPG-PUB-KEY-INTEL-OPENVINO-2022
RUN echo "deb https://apt.repos.intel.com/openvino/2022 all main" > /etc/apt/sources.list.d/intel-openvino-2022.list
RUN apt-get update && apt-get install -y intel-openvino-dev-ubuntu20-2022.1.313
RUN /opt/intel/openvino/bin/setupvars.sh

# Define environment variables, if needed
ENV DISPLAY=:0

# Run your Python application when the container launches
CMD [ "python", "app.py" ]
