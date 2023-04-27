# CS 539 Sign Language Classifier and Typing Program

This is a repository for a *COMP SCI 539: Introduction to Neural Networks* 
end-of-the-semester class project. Over the course of the semester, we touched 
on a variety of topics, including concepts present in this project such as 
classification, neural networks, and transfer learning.

For many in the group, this was the first time using git and GitHub. As a 
result, the repository is not super organized. However, we have done our best to
make the code we have created easy to explore and play with.

There were 3 stages of our model development, each of which improved on aspects
of the prior.
1. `Original Model`

    This was the first try at a neural net classifier. We trained a neural net
    to try to classify 200x200 images, but this was found to not be a very 
    generalizable approach. On top of that it was very slow to train fully.

2. `Transfer Model`

    This improvement utilized a method known as transfer learning. This is where
    another pre-trained neural network is used in conjunction with a smaller
    neural network to perform object classification. Results were still not very 
    generalizable in a live setting.

3. `Joint Detection`

    For this stage, a Google Mediapipe hand detection model was used. This 
    pre-trained model gave the X, Y, and Z positions of 21 joints in the hand 
    and the handedness for 64 total points. This was fed into 64 inputs of a 
    neural network that we trained to ultimately perform very generalizable and 
    live-feed hand sign classification.

