# KinectV2 Depth Image Keypoint Detection with CNN 🧠📸

This repository contains a project to detect keypoints and rotations in 3D space from depth images using a Convolutional Neural Network (CNN). This includes training a model, creating datasets and annotations, and testing the model on new images.
(Basicaly, it take depth informations from the KinectV2 and try to guess where the Foot,Knees and Hips are with there rotation)

I have already trained a small model to test out, and ended up with a 1.6gb model that could predict (A bit poorly cause i used a small dataset) 1 image in 0.03s on Collab TPU v2 CPU
That means, if some optimization is done, i actually thing that we can have a less than 1gb model that could run extremely fast on GPU, DirectML on AMD and CUDA on Nvidia
(30fps would be amazing)

The AI see only Normalized (0-255) Depth grayscale images to guess where the body part are, i think that founding another way to normalize depth would made the training much better
(I'm a noob in AI stuff tbh)

## Table of Contents

- [Overview](#overview)
- [Dependencies](#dependencies)
- [Setup](#setup)
- [Dataset Creation](#dataset-creation)
- [Training the Model](#training-the-model)
- [Testing the Model](#testing-the-model)
- [Usage](#usage)
- [License](#license)
- [Contribution](#contribution)
- [Questions](#questions)

## Overview

This project uses a CNN to predict the 3D positions and rotations of keypoints (feet, knees, hips) from depth images captured by a Kinect V2 sensor. The project includes scripts for:

- Create Depth Image screenshot for data
- Annotating depth images to create a dataset.
- Training a CNN model with the dataset.
- Testing the trained model on new depth images.

## Dependencies

To run this project, you will need the following dependencies:

- [Python 3.12.3](https://www.python.org/downloads/release/python-3123/)
- PyTorch
- OpenCV
- NumPy
- [Custom PyKinect2 Fork](https://github.com/Reiko69420/PyKinect2-Updated)

You can install the necessary Python packages using the following command:

  ```
  pip install torch torchvision opencv-python numpy
  ```

But you need to also install the [Custom PyKinect2 Fork](https://github.com/Reiko69420/PyKinect2-Updated)
Go in the repo and read the ReadMe to install it, it's pretty easy

## Dataset Creation

To create a dataset:

1. **Capture Depth Images:**
   Use your Kinect V2 sensor to capture depth images and save them to a directory.
   There is a python file named "create_depth_data.py" that when started, will take 1 depth image every seconds and add it to a folder named "depth_images"
   (Until you stop the python code)

2. **Annotate Images:**
   Use the provided annotation script to label keypoints and rotations in your depth images. Run the script and follow the on-screen instructions:

    ```
    python annotate_images.py
    ```

   This will generate a JSON file with the annotations. This JSon files just need to be converted to another json,
   just run convert_annotations_3d.py and everything will be converted in results.json

## Training the Model

To train the model, use the following script:

  ```
  python train_model.py
  ```


This script will:

- Load the annotated dataset.
- Train the CNN model using the dataset.
- Save the trained model to `keypoint_cnn_model.pth`.

(The script will train the model for 100 Epochs, and it's not really enough depending on your dataset)

## Testing the Model

To test the trained model on new depth images, use the following script:

  ```
  python test_model.py
  ```

This script will:

- Load the trained model.
- Preprocess and predict keypoints and rotations for each depth image in the specified directory.
- Print the results and prediction time for each image.

## Usage

Here is an example of how to run the testing script:

  ```
  python test_model.py --image_dir path/to/your/test/depth_images --model_path keypoint_cnn_model.pth
  ```

This command will process all depth images in the specified directory and output the predicted keypoints and rotations along with the time taken for each prediction.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contribution

Feel free to contribute to this project by opening issues or submitting pull requests! 🎉
I know it's not really good and a lot of things can be improved!

## Questions

If you have any questions or need further assistance, just open a "Issue"
