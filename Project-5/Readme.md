# Developing an Image Classifier with Deep Learning

In this first part of the project, you'll work through a Jupyter 
notebook to implement an image classifier with TensorFlow. We'll provide
 some tips and guide you, but for the most part the code is left up to 
you. As you work through this project, please [refer to the rubric](https://review.udacity.com/#!/rubrics/2697/view) for guidance towards a successful submission.

Remember that your code should be your own, please do not plagiarize ([see here](https://udacity.zendesk.com/hc/en-us/articles/360001451091-What-is-plagiarism-) for more information).

This notebook will be required as part of the project submission. 
After you finish it, make sure you download it as an HTML file and 
include it with the files you write in the next part of the project.

We've provided you a workspace with a GPU for working on this 
project. If you'd instead prefer to work on your local machine, you can 
find the files in the [GitHub repo](https://github.com/udacity/intro-to-ml-tensorflow).

If you are using the workspace, be aware that saving large files can 
create issues with backing up your work. You'll be saving a model 
checkpoint in Part 1 of this project which can be multiple GBs in size 
if you use a large classifier network. Dense networks can get large very
 fast since you are creating N x M weight matrices for each new layer. 
In general, it's better to avoid wide layers and instead use more hidden
 layers, this will save a lot of space. Keep an eye on the size of the 
checkpoint you create. You can open a terminal and enter `ls -lh` to see the sizes of the files. If your checkpoint is greater than 1 GB,
 reduce the size of your classifier network and re-save the checkpoint.

# Part 2 - Building the Command Line Application

Now that you've built and trained a deep neural network on the flower
 data set, it's time to convert it into an application that others can 
use. Your application should be a Python script that run from the 
command line. For testing, you should use the saved Keras model you 
saved in the first part.

### Specifications

The project submission must include a `predict.py` file 
that uses a trained network to predict the class for an input image. 
Feel free to create as many other files as you need. Our suggestion is 
to create a module just for utility functions like preprocessing images.
 **Make sure to include all files necessary to run the `predict.py` file in your submission.**

The `predict.py` module should predict the top flower names from an image along with their corresponding probabilities.

**Basic usage:**

```
$ python predict.py /path/to/image saved_model
```

**Options:**

- `--top_k` : Return the top KKK most likely classes:

```
$ python predict.py /path/to/image saved_model --top_k KKK
```

- `--category_names` : Path to a JSON file mapping labels to flower names:

```
$ python predict.py /path/to/image saved_model --category_names map.json
```

The best way to get the command line input into the scripts is with the [argparse module](https://docs.python.org/3/library/argparse.html) in the standard library. You can also find [a nice tutorial for argparse here](https://pymotw.com/3/argparse/).

### Examples

For the following examples, we assume we have a file called `orchid.jpg` in a folder named`/test_images/` that contains the image of a flower. We also assume that we have a Keras model saved in a file named `my_model.h5`.

**Basic usage:**

```
$ python predict.py ./test_images/orchid.jpg my_model.h5
```

**Options:**

- Return the top 3 most likely classes:

```
$ python predict.py ./test_images/orchid.jpg my_model.h5 --top_k 3
```

- Use a `label_map.json` file to map labels to flower names:

```
$ python predict.py ./test_images/orchid.jpg my_model.h5 --category_names label_map.json
```

# Workspace

We have provided a Command Line Interface workspace for you to run 
and test your code. TensorFlow 2.0 and TensorFlow Hub are already 
installed.

### Images for Testing

In the workspace we have provided 4 images in the `./test_images/` folder for you to check your `prediction.py` module. The 4 images are:

- cautleya_spicata.jpg
- hard-leaved_pocket_orchid.jpg
- orange_dahlia.jpg
- wild_pansy.jpg
