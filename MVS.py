#!/usr/bin/python
#! -*- encoding: utf-8 -*-

# This file is part of OpenMVG (Open Multiple View Geometry) C++ library.

# Python implementation of the bash script written by Romuald Perrot
# Created by @vins31
# Modified by Pierre Moulon
#
# this script is for easy use of OpenMVG
#
# usage : python openmvg.py image_dir output_dir
#
# image_dir is the input directory where images are located
# output_dir is where the project must be saved
#
# if output_dir is not present script will create it
#

# Indicate the openMVG binary directory
OPENMVG_SFM_BIN = "/root/autodl-tmp/openMVG/build/Linux-x86_64-RELEASE"

# Indicate the openMVG camera sensor width directory
CAMERA_SENSOR_WIDTH_DIRECTORY = "/root/autodl-tmp/openMVG/src/software/SfM" + "/../../openMVG/exif/sensor_width_database"
MVS_PATCH = "/root/autodl-tmp/openMVS_build/bin"
import os
import subprocess
import sys

if len(sys.argv) < 3:
    print ("Usage %s image_dir output_dir" % sys.argv[0])
    sys.exit(1)

input_dir = sys.argv[1]
output_dir = sys.argv[2]
matches_dir = os.path.join(output_dir, "matches")
reconstruction_dir = os.path.join(output_dir, "reconstruction_sequential")
camera_file_params = os.path.join(CAMERA_SENSOR_WIDTH_DIRECTORY, "sensor_width_camera_database.txt")

print ("Using input dir  : ", input_dir)
print ("      output_dir : ", output_dir)

# Create the ouput/matches folder if not present
if not os.path.exists(output_dir):
  os.mkdir(output_dir)
if not os.path.exists(matches_dir):
  os.mkdir(matches_dir)



# Create the reconstruction if not present
if not os.path.exists(reconstruction_dir):
    os.mkdir(reconstruction_dir)



print ("8.mvs")
pRecons = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_openMVG2openMVS"),  "-i", "/root/autodl-tmp/openMVG/build/software/SfM/matches_sequentialshiyan24/reconstruction_sequential/sfm_data.bin", "-o", "/root/autodl-tmp/openMVG/build/software/SfM/matches_sequentialshiyan24/MVS/scene.mvs","-d","\\shi7\\128yuanbufengai"] )
pRecons.wait()

print ("8.DensifyPointCloud ")
pRecons1 = subprocess.Popen( [os.path.join(MVS_PATCH, "DensifyPointCloud"),  "-i", "/root/autodl-tmp/openMVG/build/software/SfM/matches_sequential1shiyan/MVS/scene.mvs", "-o", "/root/autodl-tmp/openMVG/build/software/SfM/matches_sequential1shiyan/MVS/sceneDensifyPointCloud.mvs"] )
pRecons1.wait()

print ("9.ReconstructMesh")
pMatches1 = subprocess.Popen( [os.path.join(MVS_PATCH, "ReconstructMesh"), "-i","/root/autodl-tmp/openMVG/build/software/SfM/matches_sequential1shiyan/MVS/scene.mvs" ,"-o","/root/autodl-tmp/openMVG/build/software/SfM/matches_sequential1shiyan/MVS/ReconstructMesh.mvs"
] )
pMatches1.wait()


print("10.RefineMesh")
pMatches2 = subprocess.Popen([os.path.join(MVS_PATCH, "RefineMesh"), "-i", "/root/autodl-tmp/openMVG/build/software/SfM/matches_sequential1shiyan/MVS/ReconstructMesh.mvs", "-o", "/root/autodl-tmp/openMVG/build/software/SfM/matches_sequential1shiyan/MVS/RefineMesh.mvs"])
pMatches2.wait()

print("11.TextureMesh")
pMatches3 = subprocess.Popen([os.path.join(MVS_PATCH, "TextureMesh"), "-i", "/root/autodl-tmp/openMVG/build/software/SfM/matches_sequential1shiyan/MVS/RefineMesh.mvs", "-o", "/root/autodl-tmp/openMVG/build/software/SfM/matches_sequential1shiyan/MVS/TextureMesh.mvs"])
pMatches3.wait()
