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
- [Specialized Visualization Tools](#specialized-visualization-tools)
  - [Pie Charts](#pie-charts)
  - [Box Plots](#box-plots)
  - [Scatter Plots](#scatter-plots)
  - [Bubble Plots](#bubble-plots)
  - [Jupyter Notebook: Pie, Box, Scatter and Bubble Plots](#jupyter-notebook-pie-box-scatter-and-bubble-plots)
- [Advanced Visualization Tools](#advanced-visualization-tools)
  - [Waffle Charts](#waffle-charts)
  - [Word Clouds](#word-clouds)
  - [Seaborn and Regression Plots](#seaborn-and-regression-plots)
  - [Jupyter Notebook: Waffle Charts, Word Clouds and Regression Plots](#jupyter-notebook-waffle-charts-word-clouds-and-regression-plots)

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

[view on GitHub](https://github.com/lijqhs/data-science-ibm/blob/main/c08-data-visualization-with-python/res/NB1-Introduction-to-Matplotlib-and-Line-Plots.ipynb)


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

[view on GitHub](https://github.com/lijqhs/data-science-ibm/blob/main/c08-data-visualization-with-python/res/NB2-Area-Plots-Histograms-and-Bar-Charts.ipynb)

<br/>
<div align="right">
    <b><a href="#top">↥ back to top</a></b>
</div>
<br/>


## Specialized Visualization Tools

### Pie Charts

```python
df_continents = df_can.groupby('Continent', axis=0).sum()
# print(df_continents.head(6))

colors_list = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'lightgreen', 'pink']
explode_list = [0.1, 0, 0, 0, 0.1, 0.1] # ratio for each continent with which to offset each wedge.

df_continents['Total'].plot(kind='pie',
                            figsize=(15, 8),
                            autopct='%1.1f%%', 
                            startangle=90,    
                            shadow=True,       
                            labels=None,         # turn off labels on pie chart
                            pctdistance=1.12,    # the ratio between the center of each pie slice and the start of the text generated by autopct 
                            colors=colors_list,  # add custom colors
                            explode=explode_list # 'explode' lowest 3 continents
                            )

# scale the title up by 12% to match pctdistance
plt.title('Immigration to Canada by Continent [1980 - 2013]', y=1.12) 

plt.axis('equal') 

# add legend
plt.legend(labels=df_continents.index, loc='upper left') 

plt.show()
```

<img src="res/pie1.png" width="500">


### Box Plots

In descriptive statistics, a [box plot](https://en.wikipedia.org/wiki/Box_plot) or boxplot is a method for graphically demonstrating the locality, spread and skewness groups of numerical data through their quartiles.

The spacings in each subsection of the box-plot indicate the degree of dispersion (spread) and skewness of the data, which are usually described using the five-number summary.

In the most straight-forward method, the boundary of the lower whisker is the minimum value of the data set, and the boundary of the upper whisker is the maximum value of the data set.

Another popular choice for the boundaries of the whiskers is based on the 1.5 **IQR** value. From above the upper quartile (Q3), a distance of 1.5 times the IQR is measured out and a whisker is drawn up to the largest observed data point from the dataset that falls within this distance. 

Similarly, a distance of 1.5 times the IQR is measured out below the lower quartile (Q1) and a whisker is drawn down to the lowest observed data point from the dataset that falls within this distance. **Because the whiskers must end at an observed data point, the whisker lengths can look unequal, even though 1.5 IQR is the same for both sides.** All other observed data points outside the boundary of the whiskers are plotted as outliers. The outliers can be plotted on the box-plot as a dot, a small circle, a star, etc..

<img src="res/boxplot.png" width="300">

```python
years = list(map(str, range(1980, 2014)))
df_china = df_can.loc[['China'], years].transpose()

df_china.plot(kind='box', figsize=(14, 8))

plt.title('Box Plot of Chinese Immigrants from 1980-2013')
plt.ylabel('Number of Immigrants')
plt.show()
```

<img src="res/box1.png" width="500">


```python
df_china.describe()
```

| Country	| China |
| :--: | :--: |
| count	| 34.000000 |
| mean	| 19410.647059 |
| std	| 13568.230790 |
| min	| 1527.000000 |
| 25%	| 5512.750000 |
| 50%	| 19945.000000 |
| 75%	| 31568.500000 |
| max	| 42584.000000 |

### Scatter Plots

We can mathematically analyze the trend using a regression line (line of best fit).

Get the equation of line of best fit. We will use **Numpy**'s `polyfit()` method by passing in the following:
- `x`: x-coordinates of the data.
- `y`: y-coordinates of the data.
- `deg`: Degree of fitting polynomial. 1 = linear, 2 = quadratic, and so on.


```python
# we can use the sum() method to get the total population per year
df_tot = pd.DataFrame(df_can[years].sum(axis=0))

# change the years to type int (useful for regression later on)
df_tot.index = map(int, df_tot.index)

# reset the index to put in back in as a column in the df_tot dataframe
df_tot.reset_index(inplace = True)

# rename columns
df_tot.columns = ['year', 'total']

# view the final dataframe
df_tot.head()

df_tot.plot(kind='scatter', x='year', y='total', figsize=(15, 8), color='darkblue')

plt.title('Total Immigration to Canada from 1980 - 2013')
plt.xlabel('Year')
plt.ylabel('Number of Immigrants')


x = df_tot['year']      # year on x-axis
y = df_tot['total']     # total on y-axis
fit = np.polyfit(x, y, deg=1)


# plot line of best fit
plt.plot(x, fit[0] * x + fit[1], color='red') # recall that x is the Years
plt.annotate('y={0:.0f} x + {1:.0f}'.format(fit[0], fit[1]), xy=(2000, 150000))

plt.show()

# print out the line of best fit
'No. Immigrants = {0:.0f} * Year + {1:.0f}'.format(fit[0], fit[1]) 
```

<img src="res/scatter1.png" width="500">

    No. Immigrants = 5567 * Year + -10926195

### Bubble Plots

- To plot two different scatter plots in one plot, we can include the axes one plot into the other by passing it via the `ax` parameter.
- We will also pass in the weights using the `s` parameter. Given that the normalized weights are between 0-1, they won't be visible on the plot. Therefore, we will:
  - multiply weights by 2000 to scale it up on the graph, and,
  - add 10 to compensate for the min value (which has a 0 weight and therefore scale with $\times 2000$).


```python
# transposed dataframe
df_can_t = df_can[years].transpose()

# cast the Years (the index) to type int
df_can_t.index = map(int, df_can_t.index)

# let's label the index. This will automatically be the column name when we reset the index
df_can_t.index.name = 'Year'

# reset index to bring the Year in as a column
df_can_t.reset_index(inplace=True)

# view the changes
df_can_t.head()

# normalized Chinese data
norm_china = (df_can_t['China'] - df_can_t['China'].min()) / (df_can_t['China'].max() - df_can_t['China'].min())

# normalized Indian data
norm_india = (df_can_t['India'] - df_can_t['India'].min()) / (df_can_t['India'].max() - df_can_t['India'].min())


# China
ax0 = df_can_t.plot(kind='scatter',
                    x='Year',
                    y='China',
                    figsize=(15, 8),
                    alpha=0.5,  # transparency
                    color='green',
                    s=norm_china * 2000 + 10,  # pass in weights 
                    xlim=(1975, 2015)
                    )

# India
ax1 = df_can_t.plot(kind='scatter',
                    x='Year',
                    y='India',
                    alpha=0.5,
                    color="blue",
                    s=norm_india * 2000 + 10,
                    ax=ax0
                    )

ax0.set_ylabel('Number of Immigrants')
ax0.set_title('Immigration from China and India from 1980 to 2013')
ax0.legend(['China', 'India'], loc='upper left', fontsize='x-large')
```

<img src="res/bubble1.png" width="500">


### [Jupyter Notebook: Pie, Box, Scatter and Bubble Plots](res/NB3-Pie-Charts-Box-Plots-Scatter-Plots-and-Bubble-Plots.ipynb)

[view on GitHub](https://github.com/lijqhs/data-science-ibm/blob/main/c08-data-visualization-with-python/res/NB3-Pie-Charts-Box-Plots-Scatter-Plots-and-Bubble-Plots.ipynb)

<br/>
<div align="right">
    <b><a href="#top">↥ back to top</a></b>
</div>
<br/>


## Advanced Visualization Tools

### Waffle Charts

To create a `waffle` chart, use function `create_waffle_chart` which takes the following parameters as input:

> 1.  **categories**: Unique categories or classes in dataframe.
> 2.  **values**: Values corresponding to categories or classes.
> 3.  **height**: Defined height of waffle chart.
> 4.  **width**: Defined width of waffle chart.
> 5.  **colormap**: Colormap class
> 6.  **value_sign**: In order to make our function more generalizable, we will add this parameter to address signs that could be associated with a value such as %, $, and so on. **value_sign** has a default value of empty string.


```python
def create_waffle_chart(categories, values, height, width, colormap, value_sign=''):

    # compute the proportion of each category with respect to the total
    total_values = sum(values)
    category_proportions = [(float(value) / total_values) for value in values]

    # compute the total number of tiles
    total_num_tiles = width * height # total number of tiles
    print ('Total number of tiles is', total_num_tiles)
    
    # compute the number of tiles for each catagory
    tiles_per_category = [round(proportion * total_num_tiles) for proportion in category_proportions]

    # print out number of tiles per category
    for i, tiles in enumerate(tiles_per_category):
        print (df_dsn.index.values[i] + ': ' + str(tiles))
    
    # initialize the waffle chart as an empty matrix
    waffle_chart = np.zeros((height, width))

    # define indices to loop through waffle chart
    category_index = 0
    tile_index = 0

    # populate the waffle chart
    for col in range(width):
        for row in range(height):
            tile_index += 1

            # if the number of tiles populated for the current category 
            # is equal to its corresponding allocated tiles...
            if tile_index > sum(tiles_per_category[0:category_index]):
                # ...proceed to the next category
                category_index += 1       
            
            # set the class value to an integer, which increases with class
            waffle_chart[row, col] = category_index
    
    # instantiate a new figure object
    fig = plt.figure()

    # use matshow to display the waffle chart
    colormap = plt.cm.coolwarm
    plt.matshow(waffle_chart, cmap=colormap)
    plt.colorbar()

    # get the axis
    ax = plt.gca()

    # set minor ticks
    ax.set_xticks(np.arange(-.5, (width), 1), minor=True)
    ax.set_yticks(np.arange(-.5, (height), 1), minor=True)
    
    # add dridlines based on minor ticks
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)

    plt.xticks([])
    plt.yticks([])

    # compute cumulative sum of individual categories to match color schemes between chart and legend
    values_cumsum = np.cumsum(values)
    total_values = values_cumsum[len(values_cumsum) - 1]

    # create legend
    legend_handles = []
    for i, category in enumerate(categories):
        if value_sign == '%':
            label_str = category + ' (' + str(values[i]) + value_sign + ')'
        else:
            label_str = category + ' (' + value_sign + str(values[i]) + ')'
            
        color_val = colormap(float(values_cumsum[i])/total_values)
        legend_handles.append(mpatches.Patch(color=color_val, label=label_str))

    # add legend to chart
    plt.legend(
        handles=legend_handles,
        loc='lower center', 
        ncol=len(categories),
        bbox_to_anchor=(0., -0.2, 0.95, .1)
    )
    plt.show()
```

```python
width = 40 # width of chart
height = 10 # height of chart

categories = df_dsn.index.values # categories
values = df_dsn['Total'] # correponding values of categories

colormap = plt.cm.coolwarm # color map class

create_waffle_chart(categories, values, height, width, colormap)
```

<img src="res/waffle1.png" width="500">




### Word Clouds

```python
import numpy as np  # useful for many scientific computing in Python
import pandas as pd # primary data structure library

%matplotlib inline

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches # needed for waffle Charts

mpl.style.use('ggplot') # optional: for ggplot-like style

# check for latest version of Matplotlib
print ('Matplotlib version: ', mpl.__version__) # >= 2.0.0


df_can = pd.read_excel(
    'Canada.xlsx',
    sheet_name='Canada by Citizenship',
    skiprows=range(20),
    skipfooter=2)

# clean up the dataset to remove unnecessary columns (eg. REG) 
df_can.drop(['AREA','REG','DEV','Type','Coverage'], axis = 1, inplace = True)

# let's rename the columns so that they make sense
df_can.rename (columns = {'OdName':'Country', 'AreaName':'Continent','RegName':'Region'}, inplace = True)

# for sake of consistency, let's also make all column labels of type string
df_can.columns = list(map(str, df_can.columns))

# set the country name as index - useful for quickly looking up countries using .loc method
df_can.set_index('Country', inplace = True)

# add total column
df_can['Total'] =  df_can.sum (axis = 1)

# years that we will be using in this lesson - useful for plotting later on
years = list(map(str, range(1980, 2014)))
print ('data dimensions:', df_can.shape)


total_immigration = df_can['Total'].sum()
# total_immigration

max_words = 90
word_string = ''
for country in df_can.index.values:
    # check if country's name is a single-word name
    if country.count(" ") == 0:
        repeat_num_times = int(df_can.loc[country, 'Total'] / total_immigration * max_words)
        word_string = word_string + ((country + ' ') * repeat_num_times)

# display the generated text
# word_string

# create the word cloud
wordcloud = WordCloud(background_color='white').generate(word_string)

# print('Word cloud created!')

# display the cloud
plt.figure(figsize=(14, 18))

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
```

<img src="res/wordcloud1.png" width="500">


### Seaborn and Regression Plots

```python
df_dsn_tot = pd.DataFrame(df_dsn[years].sum(axis=0))

# change the years to type float (useful for regression later on)
df_dsn_tot.index = map(float, df_dsn_tot.index)

# reset the index to put in back in as a column in the df_tot dataframe
df_dsn_tot.reset_index(inplace=True)

# rename columns
df_dsn_tot.columns = ['year', 'total']

# view the final dataframe
df_dsn_tot.head()


plt.figure(figsize=(15, 10))

sns.set(font_scale=1.5)
sns.set_style('whitegrid')

ax = sns.regplot(x='year', y='total', data=df_dsn_tot, color='green', marker='+', scatter_kws={'s': 200})
ax.set(xlabel='Year', ylabel='Total Immigration')
ax.set_title('Total Immigration from Denmark, Sweden, and Norway to Canada from 1980 - 2013')
plt.show()
```

<img src="res/regplot1.png" width="500">

### [Jupyter Notebook: Waffle Charts, Word Clouds and Regression Plots](res/NB4_Waffle_Charts_Word_Clouds_and_Regression_Plots.ipynb)


<br/>
<div align="right">
    <b><a href="#top">↥ back to top</a></b>
</div>
<br/>



<br/>
<div align="right">
    <b><a href="#top">↥ back to top</a></b>
</div>
<br/>


---
Notes by Aaron © 2022

