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




