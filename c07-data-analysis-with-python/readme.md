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


## Datasets

Understanding Datasets

<img src="res/data-attr.png" width="500"></img>

Data source: https://archive.ics.uci.edu/ml/machine-learning-databases/autos/


<img src="res/import-data.png" width="500"></img>


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

<img src="res/db-api.png" width="500"></img>


### [Jupyter Notebook: Import data](res/DA0101EN-Review-Introduction.jupyterlite.ipynb)

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

<img src="res/binning.png" width="450"></img>

```python
bins = np.linspace(min(df["price"]), max(df["price"]), 4)
group_names = ["Low", "Medium", "High"]
df["price-binned"] = pd.cut(df["price"], bins, labels=group_names, include_lowest=True)
```

### Turning categorical variables into quantitative variables in Python

<img src="res/onehot1.png" width="450"></img>

<img src="res/onehot2.png" width="450"></img>

### [Jupyter Notebook: Preprocessing data](res/DA0101EN-2-Review-Data-Wrangling.ipynb)


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

<img src="res/groupby.png" width="400"></img>

A table of this form isn't the easiest to read and also not very easy to visualize.

To make it easier to understand, we can transform this table to a pivot table by using the `pivot` method.

#### `pivot`

<img src="res/pivot.png" width="400"></img>

The price data now becomes a rectangular grid, which is easier to visualize. This is similar to what is usually done in Excel **spreadsheets**. Another way to represent the pivot table is using a **heat map** plot.

#### Heatmap

<img src="res/heatmap.png" width="400"></img>

### Correlation

<img src="res/corr-positive.png" width="400"></img>

<img src="res/corr-negative.png" width="400"></img>

<img src="res/corr-weak.png" width="400"></img>

### Correlation - Statistics
#### Pearson Correlation

<img src="res/pearson.png" width="400"></img>

<img src="res/corr-dependence.png" width="400"></img>


The correlation reflects the noisiness and direction of a linear relationship (top row), but not the slope of that relationship (middle), nor many aspects of nonlinear relationships (bottom). N.B.: the figure in the center has a slope of 0 but in that case the correlation coefficient is undefined because the variance of Y is zero.

#### Correlation Heatmap

<img src="res/corr-heatmap.png" width="400"></img>


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

<img src="res/chi-square.png" width="400"></img>

<img src="res/chi-square2.png" width="400"></img>

<img src="res/chi-square3.png" width="400"></img>

<img src="res/chi-square4.png" width="400"></img>

### [Jupyter Notebook: Exploratory Data Analysis (EDA)](res/DA0101EN-3-Review-Exploratory-Data-Analysis.ipynb)


<br/>
<div align="right">
    <b><a href="#top">↥ back to top</a></b>
</div>
<br/>




