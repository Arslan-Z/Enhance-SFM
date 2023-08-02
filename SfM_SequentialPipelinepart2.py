#!/usr/bin/python
# -*- encoding: utf-8 -*-

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

import os
import subprocess
import sys
import time

if len(sys.argv) < 3:
    print("Usage %s image_dir output_dir" % sys.argv[0])
    sys.exit(1)

input_dir = sys.argv[1]
output_dir = sys.argv[2]
matches_dir = os.path.join(output_dir, "matches")
reconstruction_dir = os.path.join(output_dir, "reconstruction_sequential")
camera_file_params = os.path.join(CAMERA_SENSOR_WIDTH_DIRECTORY, "sensor_width_camera_database.txt")

print("Using input dir  : ", input_dir)
print("      output_dir : ", output_dir)

# Create the output/matches folder if not present
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
if not os.path.exists(matches_dir):
    os.mkdir(matches_dir)

# Create the output_terminal.txt file
terminal_output_file = os.path.join(matches_dir, "output_terminal2.txt")

# Record the start time
start_time = time.time()

# Execute subprocesses with output redirection
with open(terminal_output_file, "w") as terminal_output:
    print("5. Filter matches")
    pFiltering = subprocess.Popen(
        [
            os.path.join(OPENMVG_SFM_BIN, "openMVG_main_GeometricFilter"),
            "-i",
            matches_dir + "/sfm_data.json",
            "-m",
            matches_dir + "/matches.putative.txt",
            "-g",
            "f",
            "-o",
            matches_dir + "/matches.f.txt",
        ],
        stdout=terminal_output,
        stderr=terminal_output,
    )
    pFiltering.wait()

    # Create the reconstruction if not present
    if not os.path.exists(reconstruction_dir):
        os.mkdir(reconstruction_dir)

    print("6. Do Sequential/Incremental reconstruction")
    pRecons = subprocess.Popen(
        [
            os.path.join(OPENMVG_SFM_BIN, "openMVG_main_SfM"),
            "--sfm_engine",
            "INCREMENTAL",
            "--input_file",
            matches_dir + "/sfm_data.json",
            "--match_dir",
            matches_dir,
            "--output_dir",
            reconstruction_dir,
        ],
        stdout=terminal_output,
        stderr=terminal_output,
    )
    pRecons.wait()

    print("7. Colorize Structure")
    pRecons = subprocess.Popen(
        [
            os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeSfM_DataColor"),
            "-i",
            reconstruction_dir + "/sfm_data.bin",
            "-o",
            os.path.join(reconstruction_dir, "colorized.ply"),
        ],
        stdout=terminal_output,
        stderr=terminal_output,
    )
    pRecons.wait()

# Record the end time
end_time = time.time()

# Calculate the duration
duration = end_time - start_time

# Write the start time, end time, and duration to the output_terminal.txt file
with open(terminal_output_file, "a") as terminal_output:
    terminal_output.write(f"\nStart time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n")
    terminal_output.write(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}\n")
    terminal_output.write(f"Duration: {duration:.2f} seconds\n")