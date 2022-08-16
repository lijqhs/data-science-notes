# Data Visualization with Python  <!-- omit in toc -->

- [Introduction to Data Visualization](#introduction-to-data-visualization)
  - [Introduction to Matplotlib](#introduction-to-matplotlib)
  - [Ploting with Matplotlib](#ploting-with-matplotlib)
  - [Jupyter Notebook: Introduction to Matplotlib and Line Plot](#jupyter-notebook-introduction-to-matplotlib-and-line-plot)

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

### [Jupyter Notebook: Introduction to Matplotlib and Line Plot](res/NB-Introduction-to-Matplotlib-and-Line-Plots.ipynb)



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

