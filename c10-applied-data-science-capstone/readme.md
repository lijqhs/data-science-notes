# Applied Data Science Capstone <!-- omit in toc -->

<img src="res/Falcon9_rocket_family.svg" width="800">

Image Source: [Wikipedia: List of Falcon 9 and Falcon Heavy launches](https://en.wikipedia.org/wiki/List_of_Falcon_9_and_Falcon_Heavy_launches)

- [Data Collection](#data-collection)
  - [Data Collection API](#data-collection-api)
    - [Jupyter Notebook: Data Collection API](#jupyter-notebook-data-collection-api)
  - [Web Scraping](#web-scraping)
    - [Jupyter Notebook: Web Scraping](#jupyter-notebook-web-scraping)
  - [Data Wrangling](#data-wrangling)
    - [Jupyter Notebook: Data Wrangling](#jupyter-notebook-data-wrangling)
- [Exploratory Data Analysis](#exploratory-data-analysis)
  - [Exploratory Analysis Using SQL](#exploratory-analysis-using-sql)
    - [Jupyter Notebook: Using SQL](#jupyter-notebook-using-sql)
    - [Reference Links](#reference-links)
  - [Exploratory Analysis Using Pandas and Matplotlib](#exploratory-analysis-using-pandas-and-matplotlib)

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

Data Source: [Wikipedia: List of Falcon 9 and Falcon Heavy launches](https://en.wikipedia.org/wiki/List_of_Falcon_9_and_Falcon_Heavy_launches)

#### [Jupyter Notebook: Web Scraping](res/jupyter-labs-webscraping.ipynb)

### Data Wrangling

#### [Jupyter Notebook: Data Wrangling](res/jupyter-spacex-data-wrangling.ipynb)


<br/>
<div align="right">
    <b><a href="#top">↥ back to top</a></b>
</div>
<br/>



## Exploratory Data Analysis

### Exploratory Analysis Using SQL

#### [Jupyter Notebook: Using SQL](res/jupyter-labs-eda-sql-coursera_sqllite.ipynb)


- [Accessing SQLite Databases Using Python and Pandas](https://datacarpentry.org/python-ecology-lesson/09-working-with-sql/index.html)

#### Reference Links

- [Hands-on Lab : String Patterns, Sorting and Grouping](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DB0201EN-SkillsNetwork/labs/Labs_Coursera_V5/labs/Lab%20-%20String%20Patterns%20-%20Sorting%20-%20Grouping/instructional-labs.md.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDS0321ENSkillsNetwork26802033-2022-01-01&origin=www.coursera.org)
- [Hands-on Lab: Built-in functions](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DB0201EN-SkillsNetwork/labs/Labs_Coursera_V5/labs/Lab%20-%20Built-in%20functions%20/Hands-on_Lab__Built-in_Functions.md.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDS0321ENSkillsNetwork26802033-2022-01-01&origin=www.coursera.org)
- [Hands-on Lab : Sub-queries and Nested SELECT Statements](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DB0201EN-SkillsNetwork/labs/Labs_Coursera_V5/labs/Lab%20-%20Sub-queries%20and%20Nested%20SELECTs%20/instructional-labs.md.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDS0321ENSkillsNetwork26802033-2022-01-01&origin=www.coursera.org)
- [Hands-on Tutorial: Accessing Databases with SQL magic](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DB0201EN-SkillsNetwork/labs/Module%205/DB0201EN-Week3-1-3-SQLmagic.ipynb?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDS0321ENSkillsNetwork26802033-2022-01-01)
- [Hands-on Lab: Analyzing a real World Data Set](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DB0201EN-SkillsNetwork/labs/Module%205/DB0201EN-Week3-1-4-Analyzing.ipynb?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDS0321ENSkillsNetwork26802033-2022-01-01)


### Exploratory Analysis Using Pandas and Matplotlib



<br/>
<div align="right">
    <b><a href="#top">↥ back to top</a></b>
</div>
<br/>

