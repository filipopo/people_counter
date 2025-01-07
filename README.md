# People Counter

People counter using a VGG-16-based CSRNet model trained with the ShanghaiTech dataset. The project includes a web API and frontend interface for image uploads, enabling users to estimate crowd counts efficiently. Built with FastAPI and PyTorch, this app provides accurate predictions and user-friendly interactions

## Project Structure

`app` contains the FastAPI application code, including the API implementation and a simple HTML frontend for image uploads

`training` includes scripts for preprocessing the dataset, training the CSRNet model, and validating predictions

## CSRNet Overview

CSRNet, based on VGG-16 architecture, is a dilated convolutional neural network designed for understanding highly congested scenes. It processes crowd-counting tasks by learning hierarchical representations of visual features, achieving state-of-the-art accuracy on datasets like ShanghaiTech

## API

The application provides a REST API for predicting crowd counts in uploaded images

<details>
 <summary>/ - POST</summary>

  Predicts the number of people in a given image, for example

  Request:

  `curl -F "image=@people.jpg" http://127.0.0.1:8000`

  Response:

  ```
  {
    "Image": "people.jpg",
    "Predicted count": 3
  }
  ```
</details>

## Setup Instructions

<details>
  <summary>Training</summary>

  Perform these steps in the `training` folder

  Follow `make_dataset.ipynb` to generate the ground_truth folders

  Train the CSRNet model

  `python train.py ../dataset/part_A_train.json ../dataset/part_A_val.json`

  This should create PartAmodel_best.pth.tar and PartBmodel_best.pth.tar

  Follow `val.ipynb` to try the validation
</details>

<details>
  <summary>Running the web app</summary>

  Perform these steps in the `app` folder

  Install the dependencies

  `pip install -r requirements.txt`

  Run the web app

  `fastapi run api.py --port 8000`

  Now you can visit the website at http://127.0.0.1:8000
</details>

## Notes

The model requires a Nvidia GPU with CUDA support for efficient training but I've edited the code to allow training with just the CPU

## Thanks to

The CSRNet authors: https://arxiv.org/abs/1802.10062 https://github.com/leeyeehoo/CSRNet-pytorch

ShanghaiTech Dataset: https://drive.google.com/open?id=16dhJn7k4FWVwByRsQAEpl9lwjuV03jVI
