# Project Overview

![Image Captioning Model](https://video.udacity-data.com/topher/2018/March/5ab588e3_image-captioning/image-captioning.png)

Image Captioning Model

## Project Overview

In this project, you will create a neural network architecture to automatically generate captions from images.

After using the Microsoft Common Objects in COntext [(MS COCO) dataset](http://cocodataset.org/#home) to train your network, you will test your network on novel images!

## Project Instructions

The project is structured as a series of Jupyter notebooks that are designed to be completed in sequential order:

- 0_Dataset.ipynb
- 1_Preliminaries.ipynb
- 2_Training.ipynb
- 3_Inference.ipynb

You can find these notebooks in the Udacity workspace that appears in the concept titled **Project: Image Captioning**. This workspace provides a Jupyter notebook server directly in your browser.

You can read more about workspaces (and how to toggle GPU support) in the following concept (**Introduction to GPU Workspaces**). This concept will show you how to toggle GPU support in the workspace.

> **You MUST enable GPU mode for this project and submit your project after you complete the code in the workspace.**

> A completely trained model is expected to take between 5-12 hours to 
> train well on a GPU; it is suggested that you look at early patterns in 
> loss (what happens in the first hour or so of training) as you make 
> changes to your model, so that you only have to spend this large amount 
> of time training your *final* model.

Should you have any questions as you go, please post in the Student Hub!

## Evaluation

Your project will be reviewed by a Udacity reviewer against the CNN project [rubric](https://review.udacity.com/#!/rubrics/1427/view). Review this rubric thoroughly, and self-evaluate your project before submission. As in the first project, **you'll find that only some of the notebooks and files are graded**. All criteria found in the rubric must meet specifications for you to pass.

## Ready to submit your project?

It is a known issue that the COCO dataset is not well supported for 
download on Windows, and so you are required to complete the project in 
the GPU workspace. This will also allow you to bypass setting up an AWS 
account and downloading the dataset, locally. If you would like to refer
 to the project code, you may look at [this version (in PyTorch 0.4.0)](https://github.com/udacity/CVND---Image-Captioning-Project) at the linked Github repo.

Once you've completed your project, you may **only submit from the workspace** for this project, see the page: **[submit from here] Project: Image Captioning, Pytorch 0.4**. Click Submit, a button that appears on the bottom right of the *workspace*, to submit your project.

For submitting from the workspace, directly, please make sure that 
you have deleted any large files and model checkpoints in your notebook 
directory before submission or your project file may be too large to 
download and grade.

### GPU Workspaces

**Note: To load the COCO data in the workspace, you *must have GPU mode enabled*.**

In the next section, you'll learn more about these types of workspaces.

# LSTM Decoder

In the project, we pass all our inputs as a sequence to an LSTM. A 
sequence looks like this: first a feature vector that is extracted from 
an input image, then a start word, then the next word, the next word, 
and so on!

### Embedding Dimension

The LSTM is defined such that, as it sequentially looks at inputs, it expects that each individual input in a sequence is of a **consistent size** and so we *embed* the feature vector and each word so that they are `embed_size`.

### Sequential Inputs

So, an LSTM looks at inputs sequentially. In PyTorch, there are two ways to do this.

The first is pretty intuitive: for all the inputs in a sequence, 
which in this case would be a feature from an image, a start word, the 
next word, the next word, and so on (until the end of a sequence/batch),
 you loop through each input like so:

```
for i in inputs:
    # Step through the sequence one element at a time.
    # after each step, hidden contains the hidden state.
    out, hidden = lstm(i.view(1, 1, -1), hidden)
```

The second approach, which this project uses, is to **give the LSTM our *entire sequence*** and have it produce a set of outputs and the last hidden state:

```
# the first value returned by LSTM is all of the hidden states throughout
# the sequence. the second is just the most recent hidden state

# Add the extra 2nd dimension
inputs = torch.cat(inputs).view(len(inputs), 1, -1)
hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))  # clean out hidden state
out, hidden = lstm(inputs, hidden)
```




