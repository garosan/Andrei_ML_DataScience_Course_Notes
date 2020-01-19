## Complete Machine Learning and Data Science: Zero to Mastery   
https://www.udemy.com/course/complete-machine-learning-and-data-science-zero-to-mastery/


Syllabus
--------
- Introduction
- Machine Learning 101
- Machine Learning and Data Science framework
- The 2 Paths
- Data Science Environment Setup
- Pandas: Data Analysis
- NumPy
- Matplotlib + Seaborn: Plotting and Data Visualization
- Scikit-learn: Creating Machine Learning Models


Introduction
------------

### Course Outline
- Machine Learning 101
- Python
- Work Environment
- NumPy
- Data analysis (Pandas)
- Data visualization (matplotlib)
- Scikit-learn
- Supervised learning (Regression, classification, time series)
- Neural networks (Deep learning, Transfer learning)
- Data engineering
- Storytelling & Communication

### Classroom

https://discord.gg/nVmbHYY

Machine Learning 101
--------------------

### What is Machine Learning?


### How did we get here

### Types of Machine Learning

You guessed it: 

- Supervised (has labels)
	- Classification
	- Regression
- Unsupervised (doesn't have labels)
	- Clustering
	- Association rule learning
- Reinforcement
	- Skill acquisition
	- Real time learning

### 2nd explanation of ML

input => algorithm => output

input => output => figure out the algorithm

Machine Learning and Data Science framework
-------------------------------------------

### Introducing our framework

1. Data collection
2. Data modelling
3. Deployment

1. Create a framework
2. Match to data science and machine learning tools
3. Make projects

### 6 step Machine Learning framework

https://www.mrdbourke.com/a-6-step-field-guide-for-building-machine-learning-projects/

1. Problem definition
	- What problem are we trying to solve? Supervised/Unsupervised?
2. Data 
	- What kind of data do we have? Structured/Unstructured?
3. Evaluation
	- What defines success for us?
4. Features
	- What do we already know about the data?
5. Modelling
	- Based on our problem and data, what model should we use?
6. Experimentation
	- How could we improve? What can we try next?

### Types of Machine Learning problems

When shouldn't we use ML?
	- Will a simple hand-coded instruction based system work?

#### Main types of ML

- Supervised learning
	- Classification (discrete)
		- Binary classification
		- Multi-class classification
- Unsupervised learning (continuous)
	- Finding patterns in unsupervised data
	- Clustering
- Transfer learning
	- Ex. Applying knowledge of cat images recognition to dog images recognition
- Reinforcement learning
	- Rewards and punishments (generally used for games)

### Types of data

- Structured
	- Rows & columns, excel files
- Unstructured
	- Images, videos, audio
- Static
- Streaming data
	- Ex. News data about stock prices

### Types of evaluation

Different types of metrics


| Classification  | Regression     									| Recommendation   	|
| :-------------  | :---------- 										|	:-----------  		|
|  Accuracy  			| Mean absolute error (MAE)  			| Precision at K  	|
|  Precission    	| Mean squared error (MSE) 				|										|
|  Recall					| Root mean squared error (RMSE) 	| 									|


### Features in data

What do we already know about the data?

Feature variables. Ex: weight, sex, heart rate.
Feature target. Heart disease?

Numerical features: weight, heart rate
Categorical features: Sex, heart disease?
Derived features: Visited hospital in last year?

Feature engineering: Looking at different features in our data and creating new ones.

A feature that is not present in most of the records isn't useful.

Feature coverage. Ideally all samples have the same features present.

### Modelling - Splitting data

The most important concept in ML: Training, validation and test sets.

- Training (\~70%). Train your model on this data.
- Validation (\~15%). Tune your model on this.
- Test (\~15). Test and compare on this.

### Modelling - Picking the model

1st step in a ML journey is not to write a ML algorithm, but to know which algorithms are
better suited for which problems.

Examples:

- Structured data
	- Catboots
	- XGBoost
	- Random forest
- Unstructured data
	- Deep learning
	- Transfer learning

Use x (data) to predict y (labels)

Goal: Minimise time between experiments

| Experiment  | Accuracy | Training time |
| :---------: | :------: | :-----------: |
|  1  				| 87%  		 | 3 min 				 |
|  2    			| 91% 		 | 92 min				 |
|  3					| 94% 	   | 176 min			 |


To remember:
- Some models work better than others on different problems
- Start small and build up (add complexity) as you need

### Modelling - Tuning

- Ideally, we want to tune the model on validation data.
- If there is no validation data, we use training data.
- Hyperparameter optimization is tuning this parameters to enhance our model.
- A hyperparameter is a parameter whose value is set before the learning process begins. By contrast, the values of other parameters are derived via training.

### Modelling - Comparison

- Generalization means adapting to data the model hasn't seen before (test set).
- Underfitting: If model performs dramatically better on training set than on test set.
- Overfitting: If model performs dramatically better on test set than on training set.

How underfitting, overfitting and balanced fitting look in graphs:

![alt text](https://miro.medium.com/max/2400/1*UCd6KrmBxpzUpWt3bnoKEA.png "Overfitting and Underfitting")

Source: https://towardsdatascience.com/underfitting-and-overfitting-in-machine-learning-and-how-to-deal-with-it-6fe4a8a49dbf

Reasons for overfitting:
- Data leakage: Some train data leaks into test data

Common fixes for over and underfitting:
- Try a more advanced model
- Increase model hyperparameters
- Reduce amount of features
- Train longer

### Experimentation

Last step of the ML framework. Iterative process

### Tools we will use

- Anaconda (includes everything needed to get up and running, like:)
	- Jupyter notebooks
	- numpy
	- pandas
	- matplotlib
	- scikit-learn

Of which,
- Data analysis tools:
	- pandas
	- matplotlib
	- numpy
- Machine Learning tools:
	- Tensorflow
	- PyTorch
	- CatBoost
	- XGBoost
	- Scikit-learn

An extra resource on general AI topics:
https://www.elementsofai.com/


The 2 Paths
-----------

Are you familiar with Python or not?

ZTM blog: https://zerotomastery.io/blog/
