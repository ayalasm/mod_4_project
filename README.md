# WineGression

We analyze various physiochemical properties of wines from the Vinho Verde region in Portugal, and how these factors can be used to predict a wine's alcohol content (an important factor in reviewers' quality score). 

We find relationships between density, pH, and fixed acidtity (tartaric acid) as predictors for alcohol content based on samples from about 1,600 red wines.

With a lack of homoscedasticity in the extremes of our data, our linear regression model is not a useful tool for predicting alcohol content in wines whose predictor values are very high or very low.

## Contributors
Marco Ayala-Sanchez [(github)](https://github.com/ayalasm)

Dave Bletsch [(github)](https://github.com/davebletsch)

## Background
This is our third Flatiron School project (NYC Data Science), for module 4 - statistical modeling with linear regression.

See the [presentation](https://docs.google.com/presentation/d/1guZte2N5jead28mWM0vNwVgsPdDVIoYIegC3ZYloIdU/edit?usp=sharing) and conclusions on Google Slides or view the ppt file in our repository.

## Data
We sourced our data from a complete data set via Kaggle.com based on a viticultural [study](https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009) from 2009.

### Model
Ultimately, we used a linear regression model to predict alcohol content. Neither a Ridge or Lasso penalization improved the model's error, so the orignal OLS model was used.

### Feature Engineering
A heatmap of our variables showed no unexpected multicollinearity. 

![heatmap](https://github.com/ayalasm/mod_4_project/blob/dave/multicollinearity%20heatmap.png)

However, upon plotting grouped data, several clear variable interactions emerged. The relevance to our model only proved significant in 2 of these interactions (ph and fixed acidity, residual sugar and density).

![OLS regression results](https://github.com/ayalasm/mod_4_project/blob/dave/OLS%20results%20with%20interaction%20for%20README.png)

## Application
This model might be useful in helping to determine when grapes reach their full potential. If we know when the grapes contain the ideal amount of measurable substances, this model can predict the alcohol content of the finished wine (again, a factor that seems to influence a taster's preference).

## How to use this Repo
The main analysis is in mod_4_notebook.ipynb

You can find our presentation is labeled "Presentation.pptx"
