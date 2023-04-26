# CS 539 Sign Language Classifier and Typing Program

This is a repository for a *COMP SCI 539: Introduction to Neural Networks* 
end-of-the-semester class project. Over the course of the semester, we touched 
on a variety of topics, including concepts present in this project such as 
classification, neural networks, and transfer learning.

For many in the group, this was the first time using git and GitHub. As a 
result, the repository is not super organized. However, we have done our best to
make the code we have created easy to explore and play with.

Here are some key files that require a bit of explanation:
- `requirements.txt`
    - This file contains the `pip freeze` of the virtual environment (`venv`) 
    used in the development of these programs. It is necessary to set up your
    Python environment and ensure that all of these dependencies are installed.
    It would be wise to use a `venv` like we have.
- `PrintStringLive.py`
    - This is the main program to showcase the effective utility of the model 
    that we have developed.
    - Upon execution with the command `python PrintStringLive.py`, a camera view 
    should open. The user can then sign with the ability to see what the camera
    sees and watch as the signed letters are typed out in the terminal of 
    execution.
    - Note that in order to type two of the same character in a row, the hand
    must be removed from the frame so that the program can distinguish multiple 
    instances of the same letter. For the sake of technical simplicity, it was 
    decided that this was the easiest way to handle this.


