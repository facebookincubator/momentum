# Use Ubuntu 24.04 as the base image
FROM ubuntu:24.04

# Install necessary packages
RUN apt-get update && \
    apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*

# Run the Pixi installation command
RUN curl -fsSL https://pixi.sh/install.sh | bash

# Update PATH for Pixi
ENV PATH="/root/.pixi/bin:$PATH"

# Install FBX SDK
RUN curl -fsSL https://damassets.autodesk.net/content/dam/autodesk/www/files/fbx202037_fbxsdk_gcc_linux.tar.gz -o fbx_sdk.tar.gz && \
    mkdir /opt/fbx_sdk && \
    tar -xzf fbx_sdk.tar.gz -C /opt/fbx_sdk && \
    rm fbx_sdk.tar.gz
RUN chmod ugo+x /opt/fbx_sdk/fbx202037_fbxsdk_linux && \
    mkdir -p /usr/fbx202037_fbxsdk_linux && \
    printf "yes\nyes\n" | /opt/fbx_sdk/fbx202037_fbxsdk_linux /usr/fbx202037_fbxsdk_linux

# Copy the current directory contents into the container
COPY . /opt/momentum

# Set the working directory inside the container
WORKDIR /opt/momentum

RUN MOMENTUM_BUILD_IO_FBX=ON pixi run build

# Set the default command
CMD ["bash"]
