# Overview
This project implements a speech processing system in C++ that includes functionalities such as audio recording, playback, Linear Predictive Coding (LPC), Hidden Markov Models (HMM), and K-Means clustering for speech feature analysis.

# Features
    ## Audio Recording and Playback: Record audio for a defined duration and play it back using Windows APIs.

    ## Linear Predictive Coding (LPC): Analyze speech signals to calculate cepstral coefficients.

    ## Hidden Markov Models (HMM): Perform speech recognition with HMM algorithms, including forward, backward, and re-estimation steps.

    ## K-Means Clustering: Generate a codebook for speech feature classification.

    ## Live Testing: Recognize live-recorded audio samples against trained models.

# Prerequisites
    Windows Environment: The application uses Windows-specific APIs for audio processing.

    C++ Compiler: A Windows-compatible C++ compiler (e.g., Visual Studio) is required.

    ## Libraries:
        winmm.lib for audio recording and playback.

# File Structure
    Source Code: LPC.cpp contains the main implementation of the system.

# Input Files:
    Universe.txt: Contains the universe of feature vectors for codebook generation.

# HMM model files
    (_og_A.txt, _og_B.txt, _og_pi.txt, etc.).

# Output Files:
    Output/codebook.txt: Stores the generated codebook.

    Output/Ci/: Stores the cepstral coefficients of processed audio.

# How to Use
    ## Compile the Code:
        Use a C++ compiler like Visual Studio to compile LPC.cpp.
        Ensure all required libraries are linked (e.g., winmm.lib).

Run the Program:

Choose from three main options:

Retrain models using pre-recorded data.

Test the system with existing test files.

Perform live speech recognition using audio capture.

Recording and Playback:

The program records audio for a specified duration (default: 3 seconds).

Recorded audio is played back immediately after recording.

Training:

Processes input files to compute LPC features and generate a codebook using K-Means clustering.

Trains HMM models for each digit or class based on the processed features.

Testing:

Tests trained models with predefined test files or live-recorded audio.

Outputs classification results with probabilities.

Output Files:

Cepstral coefficients are saved in the Output/Ci/ directory.

The generated codebook is saved in Output/codebook.txt.

Live Recognition:

Records live audio, processes it, and matches it against trained models to identify the corresponding class or contact.

Notes
Modify file paths in the source code to match your directory structure if needed.

Ensure that all input files (e.g., Universe.txt, HMM model files) are present in their respective locations before running the program.

Troubleshooting
If the program fails to open input files, check the file paths specified in the code.

For issues with recording or playback, ensure that your system supports Windows Multimedia APIs (winmm.lib).

To make this downloadable, you can save it as a Markdown file named README.md and include it in your project repository or zip file.

Download Instructions
Clone the Repository: Use Git to clone the repository containing this README and the source code.

Extract the Zip File: If you downloaded a zip file, extract it to a directory on your computer.

Open README.md: Use a Markdown viewer or any text editor to view the README file.

Feel free to customize this README based on additional requirements or specific configurations of your project!

Download Link
You can provide a direct link to the downloadable zip file or repository here.

Example Use Cases
Speech Recognition: Use this system to recognize spoken digits or words by training HMM models on recorded audio samples.

Codebook Generation: Apply K-Means clustering to generate a codebook for efficient speech feature classification.

Known Issues
Compatibility: The code is designed for Windows environments due to its reliance on Windows Multimedia APIs.

File Paths: Ensure that all file paths in the code match your local directory structure.

Improved Accuracy: Experiment with different LPC orders or HMM configurations to enhance speech recognition accuracy.

Acknowledgments
Special thanks to Prof. P.K. Das for their guidance on this project.