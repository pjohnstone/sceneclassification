# sceneclassification

Project containing three separate classifiers for the task of scene recognition, built using the OpenIMAJ library.

The goal of the classifier is to take an input test image and classify it as one of 15 scenes.

Three classifiers were made:
1. K-Nearest-Neighbour using tiny image feature (21.6% accuracy)
2. LibLinearAnnotator creating 15 one-versus-all classifiers using bag-of-visual-words feature based on densely sampled pixel patch local features (63.7% accuracy)
3. Pyramid Histogram Of Words feature extractor used with a homogenous kernel map and LibLinearAnnotator (78.9% accuracy)
## Build Instructions

- Import as Maven Project and install dependencies
- Download training and testing datasets at http://comp3204.ecs.soton.ac.uk/cw/training.zip and http://comp3204.ecs.soton.ac.uk/cw/testing.zip
- Navigate to the package corresponding to the run number (run 1, 2, or 3)
- Update the ``filepath`` variable in the App classes to point to the location of the training dataset
- Run App.java

The accuracy of the classifier will be printed out after the program finishes executing.

In order to predict the classifications of unseen data, follow the instructions in the corresponding App.java files.
