# Data Visualization with Python  <!-- omit in toc -->

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