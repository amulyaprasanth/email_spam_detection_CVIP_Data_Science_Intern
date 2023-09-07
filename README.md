# Email Spam Detection Project

This project uses TensorFlow to build a machine learning model for predicting whether an email is spam or not. The model has been deployed on the Render free tier, allowing users to classify emails as spam or not through a web interface.

## Table of Contents

- [Introduction](#introduction)
- [Usage](#usage)
- [Deployment](#deployment)
- [Model Training](#model-training)
- [Dataset](#dataset)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Email spam is a common problem, and this project aims to tackle it using machine learning techniques. The project's core functionality is built on TensorFlow, a popular deep learning framework. It includes a trained machine learning model capable of classifying emails as spam or not based on their content.


## Usage

To use the spam detection model, follow these steps:

1. Go to the [demo website](https://email-spam-detection-qycu.onrender.com).
2. Wait for some time, if the page doesn't load because the application will be inactive because of render free tier. So it will take time to fire up the application.

3. Enter the text of the email you want to classify.

4. Click the "Predict" button.

5. The model will then predict whether the email is spam or not and display the result on the webpage.

## Deployment

The project is deployed on the Render free tier. Render provides a simple and reliable platform for hosting web applications and services.

To deploy this project on Render:

1. Sign up for a Render account if you don't already have one.

2. Create a new web service on Render.

3. Configure the service to use the appropriate environment and dependencies specified in `requirements.txt`.

4. Deploy the project code to the service.

5. Access the public URL of your Render service to use the email spam detection tool.

## Model Training

The machine learning model used for email spam detection is trained on a labeled dataset. The training code and details can be found in the `notebooks/model_trails.ipynb` Jupyter Notebook in the project repository.

## Dataset

The project uses a labeled dataset of emails, where each email is tagged as either spam or not spam (ham). This dataset is used for training and evaluating the machine learning model. The dataset is not included in this repository, but you can obtain a similar dataset from sources like [Kaggle](https://www.kaggle.com/uciml/sms-spam-collection-dataset).

## Contributing

Contributions to this project are welcome. If you'd like to contribute, please fork the repository, create a new branch, and submit a pull request. Make sure to follow best practices for code quality and maintainability.

## License

This project is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute the code as per the terms of this license.
