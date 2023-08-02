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

# Create the ouput/matches folder if not present
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
if not os.path.exists(matches_dir):
    os.mkdir(matches_dir)

# Create the output_terminal.txt file
terminal_output_file = os.path.join(matches_dir, "output_terminal.txt")

# Record the start time
start_time = time.time()

# Execute subprocesses with output redirection
#!/usr/bin/python
# -*- encoding: utf-8 -*-

# ... (省略部分代码)

# Execute subprocesses with output redirection
with open(terminal_output_file, "w") as terminal_output:
    print("1. Intrinsics analysis")
    pIntrisics = subprocess.Popen(
        [
            os.path.join(OPENMVG_SFM_BIN, "openMVG_main_SfMInit_ImageListing"),
            "-i",
            input_dir,
            "-o",
            matches_dir,
            "-d",
            camera_file_params,
            "-k",
            "1434.864830475782;0.0;854.3231671264116;0.0;1450.195933644484;492.3926398186667;0.0;0.0;1.0",
            "-g",
            "1",
        ],
        stdout=terminal_output,
        stderr=terminal_output,
    )
    pIntrisics.wait()

    print("2. Compute features")
    pFeatures = subprocess.Popen(
        [
            os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeFeatures"),
            "-i",
            matches_dir + "/sfm_data.json",
            "-o",
            matches_dir,
            "-m",
            "SIFT",
            # ... (省略部分代码)
        ],
        stdout=terminal_output,
        stderr=terminal_output,
    )
    pFeatures.wait()

    print("3. Compute matching pairs")
    pPairs = subprocess.Popen(
        [
            os.path.join(OPENMVG_SFM_BIN, "openMVG_main_PairGenerator"),
            "-i",
            matches_dir + "/sfm_data.json",
            "-o",
            matches_dir + "/pairs.bin",
            # ... (省略部分代码)
        ],
        stdout=terminal_output,
        stderr=terminal_output,
    )
    pPairs.wait()

    print("4. Compute matches")
    pMatches = subprocess.Popen(
        [
           os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeMatches"),
            "-i",
            matches_dir + "/sfm_data.json",
            "-p",
            matches_dir + "/pairs.bin",
            "-o",
            matches_dir + "/matches.putative.txt",
        ],
        stdout=terminal_output,
        stderr=terminal_output,
    )
    pMatches.wait()

# ... (省略部分代码)

# Record the end time
end_time = time.time()

# Calculate the duration
duration = end_time - start_time

# Write the start time, end time, and duration to the output_terminal.txt file
with open(terminal_output_file, "a") as terminal_output:
    terminal_output.write(f"\nStart time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n")
    terminal_output.write(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}\n")
    terminal_output.write(f"Duration: {duration} seconds\n")

print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
print(f"Duration: {duration} seconds")