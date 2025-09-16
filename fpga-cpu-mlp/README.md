# Codes for the FPGA-CPU design of the MLP training

This repository contains code for training a one-hidden-layer fully-connected neural network on the sensor drift dataset [1] with ap_fixed precision. The repository includes the HLS design as well as the testbench, the C++ host code for running it on a target board, and a Python code to generate the data necessary for the simulations.

Step-by-step instructions for running the HLS kernel:

Create Vitis HLS project and add files:
1. Open Vitis Unified IDE
2. Create new HLS project
3. Add all files in hls/kernel/ folder under 'Sources'.
4. Select 'dcnn1d_top' as top function.
5. Add all files in hls/testbench/ and in third-party/cnpy/ folders under 'Test Bench'.

Generate dataset files to input to the network:

6. Download sensor drift dataset from: https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset (licensed under CC-BY 4.0)
7. Generate file for training input to the testbench network by running the Python code found under python/run_mlp_train.py. One output file is the 'gas_mlp_data.npz', which contains the initial weights of the model. The Python code also saves training and test data in the "saved_files". All of these files should be added to 'Test Bench' in Vitis.

Run the HLS code:

8. Run C Simulation to verify functionality.
9. Run C Synthesis.
10. Run Co-simulation to verify functionality as it runs on hardware.
11. Run Implementation.

#

[1] Vergara, A. (2012). Gas Sensor Array Drift Dataset. UCI Machine Learning Repository. https://doi.org/10.24432/C5RP6W

This project includes the following open-source code, which is licenced under the MIT licence: https://github.com/rogersce/cnpy
