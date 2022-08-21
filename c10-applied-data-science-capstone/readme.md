# Applied Data Science Capstone <!-- omit in toc -->

<img src="res/Falcon9_rocket_family.svg" width="800">

- [Data Collection](#data-collection)
  - [Data Collection API](#data-collection-api)
    - [Jupyter Notebook: Data Collection API](#jupyter-notebook-data-collection-api)
  - [Web Scraping](#web-scraping)

## Data Collection

### Data Collection API

The [SpaceX REST API](https://github.com/r-spacex/SpaceX-API) endpoints:

- https://api.spacexdata.com/v4/cores
- https://api.spacexdata.com/v4/capsules
- https://api.spacexdata.com/v4/launches/past

```python
import requests
import pandas as pd

spacex_url = "https://api.spacexdata.com/v4/launches/past"
response = requests.get(spacex_url)
data = pd.json_normalize(response.json())
```

#### [Jupyter Notebook: Data Collection API](res/jupyter-labs-spacex-data-collection-api.ipynb)

### Web Scraping

Wikipedia: [List of Falcon 9 and Falcon Heavy launches](https://en.wikipedia.org/wiki/List_of_Falcon_9_and_Falcon_Heavy_launches)