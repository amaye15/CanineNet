# CanineNet: Dog Breed Classification for 'Le Refuge'

## Project Overview
As a volunteer at 'Le Refuge,' a local animal protection association, I embarked on a mission to develop a machine learning tool that could aid in the classification of dog breeds from their vast image database. The project was inspired by my own experience finding my beloved pet, Snooky, through this association. To give back, I aimed to streamline their data management process by implementing an advanced breed classification algorithm.

## Technical Approach
The project involves the development of a supervised image classification model using Convolutional Neural Networks (CNN). The approach includes:
- Preprocessing images with techniques like whitening, equalization, and resizing.
- Data augmentation strategies such as mirroring and cropping.
- Implementing two core methodologies:
    1. Designing a custom CNN model, inspired by existing architectures, with optimized hyperparameters and data augmentation to enhance model performance.
    2. Employing transfer learning, modifying a pre-trained network to suit our specific classification needs.

## Data
The Stanford Dogs Dataset serves as the primary training data, ensuring a robust and diverse set of images for accurate model training.

## Resource Management
Considering the resource-intensive nature of CNN training, the project offers solutions for computational limitations:
- Limiting the dataset to three dog breeds for initial testing and model design.
- Utilizing GPU computing resources or cloud platforms like Google Colab for efficient training.

## Deliverables
- A Jupyter Notebook for image analysis and preprocessing.
- A Notebook detailing the creation and training of the custom CNN model, including hyperparameter tuning and data augmentation.
- A Notebook for transfer learning-based model training.
- A local prediction program (via a notebook or a Python program with Streamlit) that inputs dog images and outputs predicted breeds.
- A presentation for the database manager at 'Le Refuge,' outlining the deployment strategy for the chosen solution.

## Contribution and Usage
This project is open for collaboration and aims to assist not only 'Le Refuge' but also other animal protection associations with similar needs. Instructions for setup, usage, and contribution guidelines are provided within this repository.
