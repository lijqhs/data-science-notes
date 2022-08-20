# Data Analysis with Python  <!-- omit in toc -->

- [Datasets](#datasets)
  - [Exporting to different formats in Python](#exporting-to-different-formats-in-python)
  - [Jupyter Notebook: Import data](#jupyter-notebook-import-data)
- [Preprocessing Data in Python](#preprocessing-data-in-python)
  - [How to deal with missing data](#how-to-deal-with-missing-data)
  - [Data Formatting in Python](#data-formatting-in-python)
  - [Data Normalization in Python](#data-normalization-in-python)
  - [Binning](#binning)
  - [Turning categorical variables into quantitative variables in Python](#turning-categorical-variables-into-quantitative-variables-in-python)
  - [Jupyter Notebook: Preprocessing data](#jupyter-notebook-preprocessing-data)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
  - [Descriptive Statistics - Describe()](#descriptive-statistics---describe)
  - [Grouping data](#grouping-data)
    - [`groupby`](#groupby)
    - [`pivot`](#pivot)
    - [Heatmap](#heatmap)
  - [Correlation](#correlation)
  - [Correlation - Statistics](#correlation---statistics)
    - [Pearson Correlation](#pearson-correlation)
    - [Correlation Heatmap](#correlation-heatmap)
  - [Association between two categorical variables: Chi-Square](#association-between-two-categorical-variables-chi-square)
  - [Jupyter Notebook: Exploratory Data Analysis (EDA)](#jupyter-notebook-exploratory-data-analysis-eda)
- [Model Development](#model-development)
  - [Linear Regression and Multiple Linear Regression](#linear-regression-and-multiple-linear-regression)
  - [Model Evaluation using Visualization](#model-evaluation-using-visualization)
    - [Regression Plot](#regression-plot)
    - [Residual Plot](#residual-plot)
    - [Distribution Plots](#distribution-plots)
  - [Polynomial Regression and Pipelines](#polynomial-regression-and-pipelines)
  - [Measures for In-Sample Evaluation](#measures-for-in-sample-evaluation)
    - [Mean Squared Error (MSE)](#mean-squared-error-mse)
    - [R-squared](#r-squared)
  - [Jupyter Notebook: Model Development](#jupyter-notebook-model-development)
- [Model Evaluation and Refinement](#model-evaluation-and-refinement)
  - [Function `cross_val_score()`](#function-cross_val_score)
  - [Function `cross_val_predict()`](#function-cross_val_predict)
  - [Overfitting, Underfitting and Model Selection](#overfitting-underfitting-and-model-selection)
  - [Ridge Regression](#ridge-regression)
  - [Grid Search](#grid-search)
  - [Jupyter Notebook: Model Evaluation and Refinement](#jupyter-notebook-model-evaluation-and-refinement)
  - [Jupyter Notebook: House Sales in King Count USA](#jupyter-notebook-house-sales-in-king-count-usa)


## Datasets

Understanding Datasets

<img src="res/data-attr.png" width="500"> 

Data source: https://archive.ics.uci.edu/ml/machine-learning-databases/autos/


<img src="res/import-data.png" width="500"> 


### Exporting to different formats in Python

| Data Format | Read | Save |
| :--: | :--: | :--: |
| `csv` | `pd.read_csv()` | `df.to_csv()` |
| `json` | `pd.read_json()` | `df.to_json()` |
| `Excel` | `pd.read_excel()` | `df.to_excel()` |
| `sql` | `pd.read_sql()` | `df.to_sql()` |


Basic insights from the data
- Understand your data before you begin any analysis
- Should check: 
  - data types
    - `df.dtypes`
  - data distribution
    - `df.describe()`
    - `df.describe(include="all")`, provides full summary statistics
      - `unique`
      - `top`
      - `freq`
- Locate potential issues with the data
  - potential info and type mismatch
  - compatibility with python methods

<img src="res/db-api.png" width="500"> 


### [Jupyter Notebook: Import data](res/1-Review-Introduction.ipynb)

<br/>
<div align="right">
    <b><a href="#top">↥ back to top</a></b>
</div>
<br/>


## Preprocessing Data in Python

- Identify and handle missing values
- Data formatting
- Data normalization (centering / scaling)
- Data binning
- Turning categorical values to numeric variables

### How to deal with missing data

- Check with the data collection source
- Drop the missing values
  - drop the variable
  - drop the data entry
- Replace the missing values
  - replace it with an average (of similar datapoints)
  - replace it by frequency
  - replace it based on other functions
- Leave it as missing data

```python
df.dropna(subset=["price"], axis=0, inplace=True)
```
is equivalent to 

```python
df = df.dropna(subset=["price"], axis=0)
```

### Data Formatting in Python

Non-formatted:
- confusing
- hard to aggregate
- hard to compare

Formatted:
- more clear
- easy to aggregate
- easy to compare

Correcting data types
- use `df.dtypes()` to identify data type
- use `df.astype()` to convert data type
  - e.g. `df["price"] = df["price"].astype("int")`

### Data Normalization in Python

Approaches for normalization:
- Simple feature scaling: x<sub>new</sub> = x<sub>old</sub>/x<sub>max</sub>
  - `df["length"] = df["length"] / df["length"].max()`
- Min-Max: x<sub>new</sub> = (x<sub>old</sub>-x<sub>min</sub>)/(x<sub>max</sub>-x<sub>min</sub>)
  - `df["length"] = (df["length"]-df["length"].min()) / (df["length"].max()-df["length"].min())`
- Z-score: x<sub>new</sub> = (x<sub>old</sub>-μ)/σ
  - `df["length"] = (df["length"]-df["length"].mean()) / df["length"].std()`

### Binning

<img src="res/binning.png" width="450"> 

```python
bins = np.linspace(min(df["price"]), max(df["price"]), 4)
group_names = ["Low", "Medium", "High"]
df["price-binned"] = pd.cut(df["price"], bins, labels=group_names, include_lowest=True)
```

### Turning categorical variables into quantitative variables in Python

<img src="res/onehot1.png" width="450"> 

<img src="res/onehot2.png" width="450"> 

### [Jupyter Notebook: Preprocessing data](res/2-Review-Data-Wrangling.ipynb)


<br/>
<div align="right">
    <b><a href="#top">↥ back to top</a></b>
</div>
<br/>

## Exploratory Data Analysis (EDA)

- Question:
  - "What are the characteristics which have the most impact on the car price?"
- Preliminary step in data analysis to:
  - Summarize main characteristics of the data
  - Gain better understanding of the data set
  - Uncover relationships between variables
  - Extract important variables

Learning Objectives:
- Descriptive Statistics
- GroupBy
- Correlation
- Correlation - Statistics


### Descriptive Statistics - Describe()

- Summarize statistics using pandas `describe()` method
  - `df.describe()`
- Summarize categorical data is by using the `value_counts()` method
- Box Plot
- Scatter Plot
  - each observation represented as a point
  - scatter plot show the relationship between two variables
    - predictor/independent variables on x-axis
    - target/dependent variables on y-axis

### Grouping data

#### `groupby`

- use `df.groupby()` method:
  - can be applied on categorical variables
  - group data into categories
  - single or multiple variables

<img src="res/groupby.png" width="500"> 

A table of this form isn't the easiest to read and also not very easy to visualize.

To make it easier to understand, we can transform this table to a pivot table by using the `pivot` method.

#### `pivot`

<img src="res/pivot.png" width="500"> 

The price data now becomes a rectangular grid, which is easier to visualize. This is similar to what is usually done in Excel **spreadsheets**. Another way to represent the pivot table is using a **heat map** plot.

#### Heatmap

<img src="res/heatmap.png" width="500"> 

### Correlation

<img src="res/corr-positive.png" width="500"> 

<img src="res/corr-negative.png" width="500"> 

<img src="res/corr-weak.png" width="500"> 

### Correlation - Statistics
#### Pearson Correlation

<img src="res/pearson.png" width="500"> 

<img src="res/corr-dependence.png" width="500"> 


The correlation reflects the noisiness and direction of a linear relationship (top row), but not the slope of that relationship (middle), nor many aspects of nonlinear relationships (bottom). N.B.: the figure in the center has a slope of 0 but in that case the correlation coefficient is undefined because the variance of Y is zero.

#### Correlation Heatmap

<img src="res/corr-heatmap.png" width="500"> 


<br/>
<div align="right">
    <b><a href="#top">↥ back to top</a></b>
</div>
<br/>

### Association between two categorical variables: Chi-Square

**Categorical variables**
- use the Chi-square Test for Association (denoted as 𝜒2)
- The test is intended to test how likely it is that an observed distribution is due to chance

**Chi-Square Test of association**
- The Chi-square tests a null hypothesis that the variables are independent.
- The Chi-square does not tell you the type of relationship that exists between both variables; but only that a relationship exists.

See also: [Chi-Square Test of Independence](https://libguides.library.kent.edu/spss/chisquare)

<img src="res/chi-square.png" width="500"> 

<img src="res/chi-square2.png" width="500"> 

<img src="res/chi-square3.png" width="500"> 

<img src="res/chi-square4.png" width="500"> 

### [Jupyter Notebook: Exploratory Data Analysis (EDA)](res/3-Review-Exploratory-Data-Analysis.ipynb)


<br/>
<div align="right">
    <b><a href="#top">↥ back to top</a></b>
</div>
<br/>

## Model Development

- simple linear regression
- multiple linear regression
- polynomial regression

### Linear Regression and Multiple Linear Regression

<img src="res/slr.png" width="500"> 

<img src="res/slr1.png" width="500"> 

<img src="res/mlr.png" width="500"> 

<img src="res/mlr1.png" width="500"> 


### Model Evaluation using Visualization

#### Regression Plot

Regression plot gives us a good estimate of:
- the relationship between two variables
- the strength of the correlation
- the direction of the relationship (positive or negative)

Regression plot shows us a combination of:
- the scatterplot: where each point represents a different `y`
- the fitted linear regression line (y&#770;)

```python
import seaborn as sns

sns.regplot(x="highway-mpg", y="price", data=df)
plt.ylim(0,)
```

#### Residual Plot

<img src="res/residual-plot.png" width="500"> 

We expect to see the results to have **zero mean**, distributed **evenly** around the `x` axis with similar variance.

```python
import seaborn as sns

sns.residplot(df["highway-mpg"], df["price"])
```

#### Distribution Plots

A distribution plot counts the predicted value versus the actual value. These plots are extremely useful for visualizing models with more than one independent variable or feature.

<img src="res/distribution-plot.png" width="500"> 


```python
import seaborn as sns

ax1 = sns.distplot(df["price"], hist=False, color="r", label="Actual Value")

sns.distplot(Yhat, hist=False, color="b", label="Fitted Value", ax=ax1)
```


<br/>
<div align="right">
    <b><a href="#top">↥ back to top</a></b>
</div>
<br/>

### Polynomial Regression and Pipelines

<img src="res/poly-reg.png" width="500"> 

<img src="res/poly-reg1.png" width="500"> 

<img src="res/poly-reg2.png" width="500"> 

<img src="res/poly-reg3.png" width="500"> 

Numpy's polyfit function cannot perform this type of regression. We use the preprocessing library in scikit-learn to create a polynomial feature object.


```python
from sklearn.preprocessing import PolynomialFeatures

pr = PolynomialFeatures(degree=2, include_bias=False)
x_poly = pr.fit_transform(x[['horsepower', 'curb-weight']])
```

<img src="res/poly-reg4.png" width="500"> 

As the dimension of the data gets larger, we may want to normalize multiple features in scikit-learn. Instead we can use the preprocessing module to simplify many tasks. For example, we can standardize each feature simultaneously. We import `StandardScaler`.


```python
from sklearn.preprocessing import StandardScaler
SCALE = StandardScaler()
SCALE.fit(x_data[['horsepower', 'highway-mpg']])
x_scale = SCALE.transform(x_data[['horsepower', 'highway-mpg']])
```

We can simplify our code by using a pipeline library.

<img src="res/pipeline.png" width="500"> 


```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

Input = [('scale', StandardScaler()), ('polynomial', PolynomialFeatures(degree=2),...), ('model', LinearRegression())]
pipe = Pipeline(Input)
pipe.fit(df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y)
yhat = pipe.predict(X[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
```


<br/>
<div align="right">
    <b><a href="#top">↥ back to top</a></b>
</div>
<br/>

### Measures for In-Sample Evaluation

Measures for In-Sample Evaluation
- A way to numerically determine how good the model fits on dataset
- Two important measures to determine the fit of a model:
  - Mean Squared Error (MSE)
  - R-squared (R<sup>2</sup>)

#### Mean Squared Error (MSE)

```python
from sklearn.metrics import mean_square_error

mean_square_error(df['price'], Y_predict_simple_fit)
```

#### R-squared

- The [Coefficient of Determination](https://en.wikipedia.org/wiki/Coefficient_of_determination) or R-squared (R<sup>2</sup>)
- Is a measure to determine how close the data is to the fitted regression line.
- R<sup>2</sup>: the percentage of variation of the target variable (Y) that is explained by the linear model.
- think about as comparing a regression model to a simple model i.e. the mean of the data points

R<sup>2</sup>=(1-(MSE of regression line)/(MSE of the average of the data))

<img src="res/r-square.png" width="500"> 

- Generally the values of the MSE are between 0 and 1
- We can calculate the R<sup>2</sup> as follows

```python
X = df[['highway-mpg']]
Y = df['price']
lm.fit(X, Y)
lm.score(X, Y)  # 0.496591188
```
We can say that approximately **49.695%** of the variation of price is explained by this simple linear model.

<img src="res/mlr-slr.png" width="500"> 

### [Jupyter Notebook: Model Development](res/4-Review-Model-Development.ipynb)


<br/>
<div align="right">
    <b><a href="#top">↥ back to top</a></b>
</div>
<br/>

## Model Evaluation and Refinement

Training/Testing Sets
- Split dataset into:
  - Training set (70%)
  - Testing set (30%)
- Build and train the model with a training set
- Use testing set to assess the performance of a predictive model
- When we have completed testing our model we should use all the data to train the model to get the best performance

<img src="res/train-test-split.png" width="500"> 

### Function `cross_val_score()`

One of the most common out of sample evaluation metrics is [cross-validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)). 
- In this method, the dataset is split into K equal groups. 
- Each group is referred to as a fold. For example, four folds. Some of the folds can be used as a training set which we use to train the model and the remaining parts are used as a test set, which we use to test the model. 
- For example, we can use three folds for training, then use one fold for testing. This is repeated until each partition is used for both training and testing. 
- At the end, we use the average results as the estimate of out-of-sample error. 
- The evaluation metric depends on the model, for example, the r squared. 

The simplest way to apply cross-validation is to call the `cross_val_score` function, which performs multiple out-of-sample evaluations.

```python
from sklearn.model_selection import cross_val_score

score = cross_val_score(lr, x_data, y_data, cv=3)
np.mean(scores)
```

### Function `cross_val_predict()`

- It returns the prediction that was obtained for each element when it was in the test set
- Has a similar interface to `cross_val_score()`

```python
from sklearn.model_selection import cross_val_predict

yhat = cross_val_predict(lr2e, x_data, y_data, cv=3)
```

### Overfitting, Underfitting and Model Selection

<img src="res/model-selection.png" width="500"> 

<img src="res/model-selection-r-square.png" width="500"> 

Calculate different R-squared values as follows:

```python
Rsqu_test = []
order = [1,2,3,4]

for n in order:
  pr = PolynomialFeatures(degree=n)
  x_train_pr = pr.fit_transform(x_train[['horsepower']])
  x_test_pr = pr.fit_transform(x_test[['horsepower']])
  lr.fit(x_train_pr, y_train)
  Rsqu_test.append(lr.score(x_test_pr, y_test))
```


<br/>
<div align="right">
    <b><a href="#top">↥ back to top</a></b>
</div>
<br/>


### Ridge Regression

Ridge regression is a regression that is employed in a Multiple regression model when Multicollinearity occurs. Multicollinearity is when there is a strong relationship among the independent variables. Ridge regression is very common with polynomial regression. 

<img src="res/ridge-regression.png" width="500"> 


The column corresponds to the different polynomial coefficients, and the rows correspond to the different values of alpha. 
- As alpha increases, the parameters get smaller. This is most evident for the higher order polynomial features. 
- But Alpha must be selected carefully. 
  - If alpha is too large, the coefficients will approach zero and underfit the data. 
  - If alpha is zero, the overfitting is evident.

<img src="res/ridge-choose-alpha.png" width="500"> 

### Grid Search

- The term alpha in Ridge regression is called a **hyperparameter**.
- Scikit-learn has a means of automatically iterating over these hyperparameters using cross-validation called **Grid Search**. 

[Grid Search](https://scikit-learn.org/stable/modules/grid_search.html) takes the model or objects you would like to train and different values of the hyperparameters. It then calculates the mean square error or R-squared for various hyperparameter values, allowing you to choose the best values. 

<img src="res/grid-search.png" width="500"> 

Use the validation dataset to pick the best hyperparameters.

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

parameters1 = [{'alpha': [0.001, 0.1, 1, 10, 100, 1000, 10000, 100000]}]

RR = Ridge()
Grid1 = GridSearchCV(RR, parameters1, cv=4)
Grid1.fit(x_data[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_data)
Grid1.best_estimator_

scores = Grid1.cv_results_
scores['mean_test_score']
```

What are the advantages of Grid Search is how quickly we can test **multiple parameters**. 

<img src="res/grid-search-2-parameters.png" width="500"> 

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

parameters2 = [{'alpha': [0.001, 0.1, 1, 10, 100], 'normalize': [True, False]}]

RR = Ridge()
Grid1 = GridSearchCV(RR, parameters2, cv=4)
Grid1.fit(x_data[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_data)
Grid1.best_estimator_

scores = Grid1.cv_results_

for param, mean_val, mean_test in zip(scores['params'], scores['mean_test_score'], scores['mean_train_score']):
  print(param, "R^2 on test data:", mean_val, "R^2 on train data:", mean_test)
```

<img src="res/grid-search-scores.png" width="500"> 


### [Jupyter Notebook: Model Evaluation and Refinement](res/5-Review-Model-Evaluation-and-Refinement.ipynb)


### [Jupyter Notebook: House Sales in King Count USA](res/House_Sales_in_King_Count_USA.ipynb)



<br/>
<div align="right">
    <b><a href="#top">↥ back to top</a></b>
</div>
<br/>

