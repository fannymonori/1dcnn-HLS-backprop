# Codes for an FPGA-only design of 1D-CNN backpropagation

This folder contains the HLS codes, the testbench, and the C++ host code necessary for running the end-to-end training of a small 1D-CNN network on AMD's FPGAs. Network architecture is based on [1]. The training of the network runs fully on the FPGA.

Step-by-step instructions for running the HLS kernel:

Create Vitis HLS project and add files:
1. Open Vitis Unified IDE
2. Create new HLS project
3. Add all files in hls/kernel/src folder under 'Sources'.
4. Select 'dcnn1d_top' as top function.
5. Add all files in hls/testbench/src folder under 'Test Bench'.

Generate dataset files to input to the network:

6. Download wine spoilage dataset [2] from: https://data.mendeley.com/datasets/vpc887d53s/1 (licensed under CC-BY)
7. Generate file for training input to the testbench network by running the Python code found under python/src/run.py. Output of file is 'train_samples_wine.txt'. The file can be added to 'Test Bench' in Vitis.

Run the HLS code:

8. Run C Simulation to verify functionality.
9.  Run C Synthesis.
10.  Run Co-simulation to verify functionality as it runs on hardware.
11.  Run Implementation.

Running on hardware:

1. To run on the target hardware, set-up a platform project in Vitis.
2. Create an HLS kernel, using the code under hls/
3. Put the code from the host_code folder and the 'train_samples_wine.txt' in the Application project.
4. Set-up the hardware configurations and run.

#

[1] Y. Wang et al., “An optimized deep convolutional neural network for dendrobium classification based on electronic nose,” Sensors and Actuators A: Physical, vol. 307, 111874, 2020

[2] RODRIGUEZ GAMBOA, JUAN CARLOS; Albarracin Estrada, Eva Susana (2019), “Electronic nose dataset for detection of wine spoilage thresholds”, Mendeley Data, V1, doi: 10.17632/vpc887d53s.1