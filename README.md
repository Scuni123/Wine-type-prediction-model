# Wine-type-prediction-model
This is my first ML model I made using scikit-learn as the backbone. The whole idea was the practice all the ML basics and build upon the foundation I already created for myself.
I spent time learning the basics for Python, Pandas, MatPlotLib and Machine Learning statsitical methods so now it was time to try and make my own model.

## The data
Scikit learn already has databases built into it, so I wanted to work with one that I hadn't already read about so I couldn't really "cheat" and copy it. I ended up going with the "load_wine" dataset. This data has 178 samples with 13 features and 3 classes.

> I will admit that getting the data from scikit learn really glosses over a big part of DataScience which is getting the data and organzing/cleaning it. That part was done for me, but I had tried practicing this technique myself before with random Country GDP data I found on the internet.

## Exploratory Data Analysis

So, I wanted to get a better idea of what the data actually looks like and what features are relavant. I started off with basic functions such as .describe, .head(), and.shape. This helped me see which features are a part of the data set.

Next, I wanted to try and make a PCA graph as a bird's eye view of how the features interact with one another. After this, I compared some key features against the classes in order to see which features deserved a spotlight in the model I made.

> After the data set has been pre-processed and selected features were put into X and the classes into y, the data was ready to move onto the next step.

## Data splitting

So, it is very bad pratice to train your model on the data that you will testing it on. In order to work around this, I used scikit-learns train_test_split in order to break the data in 80% training and 20% testing.

Now I have the variables and the data split for the model.

## Choosing an appropriate model

Based off of all the ML statistial anysis methods I learned about and review last week, I had a few in mind to try. Some of the common ones I had in mind and read about were:
 * K-Nearest neighbor (KNN)
 * Support Vector Machine (SVM)
 * Random Forest
 * Logistical regression
 * Linear regression

These are all ML methods that are avaiable through scikit learn. I tested them out and found the Random forest and logistical regression to give 100% accuracy when using training data to predict y test values. At this point I could stop, but I wanted to test out the GridSearchCV grid in order to work on other skills.

## Cross validation and hyperparameter optimization
I chose to go with the RFC model. I then used the .get_params() function in order to see all the features I could try and change to optimize the model further (in theory).

The general format I choose for testing features was as follows:
~~~
grid =GridSearchCV(
    estimator = model_RFC,
    param_grid = {'max_depth': [5, 6, 7, 8, 9, 10, 11, 12 , 13, 14, 15]},
    cv=10       
)
grid.fit(X_train, y_train)
pd.DataFrame(grid.cv_results_)
~~~

I tried this for both max_depth and n_estimators. From there, I chose the best results for each and then retrained my model using the optimized hyperparameters to end up with m final model:

~~~
model_RFC_update = RandomForestClassifier(n_estimators=500, max_depth = 9)
model_RFC_update.fit(X_train, y_train)
predictions_RFC_update = model_RFC_update.predict(X_test)
print(accuracy_score(y_test, predictions_RFC_update))
~~~

Again, this resulted in a perfect model, but I'm glad I got to try more techniques.
