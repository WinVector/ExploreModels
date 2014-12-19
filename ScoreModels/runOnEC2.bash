#!/bin/bash

# run this script on an Amazon 64 bit Amazon Linux 
# recommend a 64 bit instance with at least 8GB of ram
# installs dev tools and Python sklearn on a fresh machine
# also downloads the data
# as of 12-17-2014 this defaulting to Python 2.6, so written for that
# did not work on micro instance, did work on c3.2xlarge instance


# Start getting data (in background):
echo "starting background data download"
wget http://persoal.citius.usc.es/manuel.fernandez.delgado/papers/jmlr/data.tar.gz 1>/dev/null 2>&1 &


echo "installing software"
# Configure machine
sudo yum -y groupinstall "Development Tools"
# from http://dacamo76.com/blog/2012/12/07/installing-scikit-learn-on-amazon-ec2/
sudo yum -y install gcc-c++ python-devel atlas-sse3-devel lapack-devel
sudo easy_install pip
sudo pip install -U liac-arff
sudo pip install -U pytz numpy pandas scipy scikit-learn


# Wait for data load to finish
echo "waiting for data to finish downloading"
wait
echo "expect: 252603f2a5bf8d7975a392cdf5f84fb1c5d9b5c2  data.tar.gz"
shasum data.tar.gz 


# run the job
echo "running the job"
python processArffs.py data.tar.gz > processArffs_log.txt 2>&1

wait
echo "job done"



