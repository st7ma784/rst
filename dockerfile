# Use the python image to save us time in setting up the environment later 
FROM ubuntu:20.04 
# Find an image that has python installed but behaves like ubuntu
#FROM python:3.8-alpine

#TO DO : HOW TO BUILD THIS ON ARM

# Step 1: Update the package list and install necessary packages
# RUN echo 'APT::Install-Suggests "0";' >> /etc/apt/apt.conf.d/00-docker
# RUN echo 'APT::Install-Recommends "0";' >> /etc/apt/apt.conf.d/00-docker
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libhdf5-serial-dev \
    libncurses-dev \
    libnetcdf-dev \
    python3-pip \
    libpng-dev \
    libx11-dev \
    libxext-dev \
    netpbm \
    build-essential \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \ 
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Step 2 : Install the necessary cdf package
ADD https://spdf.gsfc.nasa.gov/pub/software/cdf/dist/latest/linux/cdf39_1-dist-cdf.tar.gz ./app/cdf39_1-dist-cdf.tar.gz
# Unpack and install the package (see README.install for more information):
WORKDIR /app
SHELL ["/bin/bash", "-c"]
RUN tar -xzvf /app/cdf39_1-dist-cdf.tar.gz && \
    cd cdf39_1-dist && \
    make OS=linux ENV=gnu all && \
    make test && \
    make INSTALLDIR=/usr/local/cdf install && \
    cd /app && \
    rm -rf cdf39_1-dist && \
    rm cdf39_1-dist-cdf.tar.gz
ENV CDF_PATH=/usr/local/cdf
# Step 3:  install RST

# Set the working directory
COPY . /app/rst

# Open rst/.profile/base.bash to check paths are correctly set:
# XPATH, NETCDF_PATH, CDF_PATH To check if the paths are set correctly locate the following header files: For NETCDF_PATH locate netcdf.h For CDF PATH locate cdf.h
RUN sed -i 's|XPATH=.*|XPATH=/usr/local|' /app/rst/.profile/base.bash && \
    sed -i 's|NETCDF_PATH=.*|NETCDF_PATH=/usr|' /app/rst/.profile/base.bash && \
    sed -i 's|CDF_PATH=.*|CDF_PATH=/usr/local/cdf|' /app/rst/.profile/base.bash

# # Load the RST environment variables. Open and edit your ~/.bashrc file to include:
ENV RSTPATH=/app/rst

# # where the INSTALL LOCATION is the path with the RST repository that has been copied to your computer. In order to load the environment variables you just setup, you'll need to close your current terminal and open a new terminal, or from the command line type:
# # Run make.build from the command line. You may need to change directory to $RSTPATH/build/script. This runs a helper script that sets up other compiling code.
ENV PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/local/bin:/usr/bin/:/root/bin:/root/script:/app/rst/build/bin:/app/rst/build/script:/app/rst/bin:/app/rst/script"
#HATE THIS LINE TOO !


# Load the RST environment variables
RUN source /app/rst/.profile.bash && \
    cd /app/rst/build/script && \
    make.build && \
    make.code

    
##ad app/rst/.profile.bash to the .bashrc file
RUN echo "source /app/rst/.profile.bash" >> ~/.bashrc

RUN source ~/.bashrc
# # # # # Set the working directory
# # # # # Define the entry point for the container
WORKDIR /app

ENV MAPDATA=/app/rst/tables/general/map_data
ENV BNDDATA=/app/rst/tables/general/bnd_data

ENV ISTP_PATH="/data/istp"

ENV SD_HDWPATH=/app/rst//tables/superdarn/hdw/
ENV SD_TDIFFPATH=/app/rst/tables/superdarn/tdiff/
ENV SD_RADAR=/app/rst/tables/superdarn/radar.dat

ENV AACGM_DAT_PREFIX=/app/rst/tables/analysis/aacgm/aacgm_coeffs
ENV IGRF_PATH=/app/rst/tables/analysis/mag/
ENV SD_MODEL_TABLE=/app/rst/tables/superdarn/model

ENV AACGM_v2_DAT_PREFIX=/app/rst/tables/analysis/aacgm/aacgm_coeffs-13-
ENV IGRF_COEFFS=/app/rst/tables/analysis/mag/magmodel_1590-2020.txt

ENV COLOR_TABLE_PATH=/app/rst/tables/base/key/
CMD ["bash"]