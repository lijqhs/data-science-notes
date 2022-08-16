# Data Visualization with Python  <!-- omit in toc -->

- [Introduction to Data Visualization](#introduction-to-data-visualization)
  - [Introduction to Matplotlib](#introduction-to-matplotlib)
  - [Ploting with Matplotlib](#ploting-with-matplotlib)
  - [Jupyter Notebook: Introduction to Matplotlib and Line Plot](#jupyter-notebook-introduction-to-matplotlib-and-line-plot)
- [Basic Visualization Tools](#basic-visualization-tools)
  - [Area Plots](#area-plots)
  - [Histogram](#histogram)
  - [Bar Charts](#bar-charts)
  - [Jupyter Notebook: Area Plots, Histograms, Bar Charts](#jupyter-notebook-area-plots-histograms-bar-charts)

## Introduction to Data Visualization

Benefits of visualization:
- for exploratory data analysis
- communicate data clearly
- share unbiased representation of data
- use them to support recommendations to different stakeholders

### Introduction to Matplotlib

Read: [Matplotlib by John Hunter](https://www.aosabook.org/en/matplotlib.html)

The **Matplotlib architecture** is composed of three main layers:
- **Backend Layer** — Handles all the heavy works via communicating to the drawing toolkits in your machine. It is the most complex layer.
- **Artist Layer** — Allows full control and fine-tuning of the Matplotlib figure — the top-level container for all plot elements.
- **Scripting Layer** — The lightest scripting interface among the three layers, designed to make Matplotlib work like MATLAB script.

**Using Artist Layer to generate a histogram:**

```python
# Import the FigureCanvas from the backend of your choice
#  and attach the Figure artist to it.
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
fig = Figure()
canvas = FigureCanvas(fig)

# Import the numpy library to generate the random numbers.
import numpy as np
x = np.random.randn(10000)

# Now use a figure method to create an Axes artist; the Axes artist is
#  added automatically to the figure container fig.axes.
# Here "111" is from the MATLAB convention: create a grid with 1 row and 1
#  column, and use the first cell in that grid for the location of the new
#  Axes.
ax = fig.add_subplot(111)

# Call the Axes method hist to generate the histogram; hist creates a
#  sequence of Rectangle artists for each histogram bar and adds them
#  to the Axes container.  Here "100" means create 100 bins.
ax.hist(x, 100)

# Decorate the figure with a title and save it.
ax.set_title('Normal distribution with $\mu=0, \sigma=1$')
fig.savefig('matplotlib_histogram.png')
```

<img src="res/matplotlib_histogram.png" width="300">

**Using Scripting Layer to do the same:**

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.random.randn(10000)
plt.hist(x, 100)
plt.title(r'Normal distribution with $\mu=0, \sigma=1$')
plt.savefig('matplotlib_histogram.png')
plt.show()
```

### Ploting with Matplotlib

```python
%matplotlib notebook
import matplotlib.pyplot as plt

plt.plot(5, 5, 'o')
```

- A magic function starts with `%matplotlib`, and to enforce plots to be rendered within the browser, you pass in inline as the backend. 
- Matplotlib has a number of different backends available. One limitation of this backend is that you cannot modify a figure once it's rendered. 
- So after rendering the above figure, there is no way for us to add, for example, a figure title or label its axes. You will need to generate a new plot and add a title and the axes labels before calling the show function. 
- A backend that overcomes this limitation is the notebook backend. With the `notebook` backend in place, if a plt function is called, it checks if an active figure exists, and any functions you call will be applied to this active figure. If a figure does not exist, it renders a new figure. So when we call the `plt.plot` function to plot a circular mark at position (5, 5), the backend checks if an active figure exists.

**Matplotlib - Pandas**

Another thing that is great about Matplotlib is that pandas also has a built-in implementation of it.

### [Jupyter Notebook: Introduction to Matplotlib and Line Plot](res/NB1-Introduction-to-Matplotlib-and-Line-Plots.ipynb)



<br/>
<div align="right">
    <b><a href="#top">↥ back to top</a></b>
</div>
<br/>

## Basic Visualization Tools

### Area Plots

```python
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.style.use(['ggplot'])  # for ggplot-like style

df_can.sort_values(by='Total', ascending=False, axis=0, inplace=True)
df_top7 = df_can.head(7)
df_top7 = df_top7[years].transpose()
df_top7.index = df_top7.index.map(int)

df_top7.plot(kind='area', alpha=0.45, figsize=(14, 8)) # pass a tuple (x, y) size

plt.title('Immigration Trend of Top 7 Countries')
plt.ylabel('Number of Immigrants')
plt.xlabel('Years')
plt.show()
```

<img src="res/area-top7.png" width="500">


### Histogram

A **histogram** is a graph that shows the **frequency** of numerical data using rectangles. The height of a rectangle (the vertical axis) represents the distribution frequency of a variable (the amount, or how often that variable appears). The width of the rectangle (horizontal axis) represents the value of the variable (for instance, minutes, years, or ages).

```python
import matplotlib.pyplot as plt

df_can['2013'].plot(kind='hist', figsize=(14, 8))
plt.title('Histogram of Immigration from 195 countries in 2013')
plt.ylabel('Number of Countries')
plt.xlabel('Number of Immigrants')
plt.show()
```

<img src="res/hist1.png" width="500">


A histogram that depicts the distribution of immigration to Canada in 2013, but notice how the bins are **not aligned with the tick marks** on the horizontal axis. This can make the histogram hard to read.

One way to solve this issue is to borrow the `histogram` function from the `Numpy` library. What `histogram` does:
- partitions the spread of the data in column 2013 into 10 bins of equal width,
- computes the number of datapoints that fall in each bin,
- returns this frequency (`count`) and the bin edges (`bin_edges`).


```python
import matplotlib.pyplot as plt
import numpy as np

count, bin_edges = np.histogram(df_can['2013'])

df_can['2013'].plot(kind='hist', xticks = bin_edges, figsize=(14, 8))
plt.title('Histogram of Immigration from 195 countries in 2013')
plt.ylabel('Number of Countries')
plt.xlabel('Number of Immigrants')
plt.show()
```

<img src="res/hist2.png" width="500">

### Bar Charts

A bar chart is a very popular visualization tool. Unlike a histogram, a bar chart also known as a bar graph is a type of plot where the length of each bar is proportional to the value of the item that it represents. It is commonly used to compare the values of a variable at a given point in time.

```python
import matplotlib.pyplot as plt
years = list(map(str, range(1980, 2014)))

df_china = df_can.loc['China', years]

df_china.plot(kind='bar', figsize=(14, 8))
plt.title('Chinese Immigrants to Canada from 1980 to 2013')
plt.ylabel('Number of Immigrants')
plt.xlabel('Years')
plt.show()
```

<img src="res/bar1.png" width="500">


### [Jupyter Notebook: Area Plots, Histograms, Bar Charts](res/NB2-Area-Plots-Histograms-and-Bar-Charts.ipynb)


<br/>
<div align="right">
    <b><a href="#top">↥ back to top</a></b>
</div>
<br/>




---
Notes by Aaron © 2022

