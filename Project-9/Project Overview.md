# Project: Machine Translation

Machine translation is a popular topic in research with new papers 
coming out every year. Over the years of research, different methods 
were created, like [rule-based](https://en.wikipedia.org/wiki/Rule-based_machine_translation), [statistical](https://en.wikipedia.org/wiki/Statistical_machine_translation), and [example-based](https://en.wikipedia.org/wiki/Example-based_machine_translation) machine translation. With all this effort, it’s still an unsolved 
problem. However, neural networks have made a large leap forward in 
machine translation.

In this notebook, you will build a deep neural network that functions
 as part of an end-to-end machine translation pipeline. Your completed 
pipeline will accept English text as input and return the French 
translation.

After you complete the workspace, make sure you review the [rubric](https://review.udacity.com/#!/rubrics/4672/view) to ensure your submission meets all requirements to pass.

To make your project stand out, we provide you with a few optional challenges:

1. This project focuses on learning various network architectures for 
   machine translation, but we don't evaluate the models according to best 
   practices by splitting the data into separate test & training sets 
   -- so the model accuracy is overstated. Use the `sklearn.model_selection.train_test_split()` function to create separate training & test datasets, then retrain 
   each of the models using only the training set and evaluate the 
   prediction accuracy using the holdout test set. Does the "best" model 
   change?
2. To enhance your model, try to fine-tune the hyperparameters and experiment with extra layers. Does your model improve a lot?
