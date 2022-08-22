# Applied Data Science Capstone <!-- omit in toc -->

<img src="res/Falcon9_rocket_family.svg" width="800">

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
    - [Lab Reference Links](#lab-reference-links)
  - [Exploratory Analysis Using Pandas and Matplotlib](#exploratory-analysis-using-pandas-and-matplotlib)
    - [Jupyter Notebook: Data Visualization](#jupyter-notebook-data-visualization)
  - [Interactive Visual Analytics and Dashboards](#interactive-visual-analytics-and-dashboards)
    - [Jupyter Notebook: Folium](#jupyter-notebook-folium)
    - [SpaceX Dash App](#spacex-dash-app)

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

#### Lab Reference Links

- [Hands-on Lab : String Patterns, Sorting and Grouping](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DB0201EN-SkillsNetwork/labs/Labs_Coursera_V5/labs/Lab%20-%20String%20Patterns%20-%20Sorting%20-%20Grouping/instructional-labs.md.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDS0321ENSkillsNetwork26802033-2022-01-01&origin=www.coursera.org)
- [Hands-on Lab: Built-in functions](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DB0201EN-SkillsNetwork/labs/Labs_Coursera_V5/labs/Lab%20-%20Built-in%20functions%20/Hands-on_Lab__Built-in_Functions.md.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDS0321ENSkillsNetwork26802033-2022-01-01&origin=www.coursera.org)
- [Hands-on Lab : Sub-queries and Nested SELECT Statements](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DB0201EN-SkillsNetwork/labs/Labs_Coursera_V5/labs/Lab%20-%20Sub-queries%20and%20Nested%20SELECTs%20/instructional-labs.md.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDS0321ENSkillsNetwork26802033-2022-01-01&origin=www.coursera.org)
- [Hands-on Tutorial: Accessing Databases with SQL magic](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DB0201EN-SkillsNetwork/labs/Module%205/DB0201EN-Week3-1-3-SQLmagic.ipynb?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDS0321ENSkillsNetwork26802033-2022-01-01)
- [Hands-on Lab: Analyzing a real World Data Set](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DB0201EN-SkillsNetwork/labs/Module%205/DB0201EN-Week3-1-4-Analyzing.ipynb?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDS0321ENSkillsNetwork26802033-2022-01-01)


### Exploratory Analysis Using Pandas and Matplotlib

#### [Jupyter Notebook: Data Visualization](res/jupyter-labs-eda-dataviz.ipynb)


### Interactive Visual Analytics and Dashboards

#### [Jupyter Notebook: Folium](res/jupyter_launch_site_location_folium.ipynb)

```python
# Function to assign color to launch outcome
def assign_marker_color(launch_outcome):
    if launch_outcome == 1:
        return 'green'
    else:
        return 'red'
    
spacex_df['marker_color'] = spacex_df['class'].apply(assign_marker_color)
```

```python
marker_cluster = MarkerCluster()
# Add marker_cluster to current site_map
site_map.add_child(marker_cluster)

# for each row in spacex_df data frame
# create a Marker object with its coordinate
# and customize the Marker's icon property to indicate if this launch was successed or failed, 
# e.g., icon=folium.Icon(color='white', icon_color=row['marker_color']
for index, record in spacex_df.iterrows():
    # TODO: Create and add a Marker cluster to the site map
    # marker = folium.Marker(...)
    
    marker = folium.map.Marker(
        [record['Lat'],record['Long']],
        # Create an icon as a text label
        icon=folium.Icon(color='white', icon_color=record['marker_color'])
    )
    marker_cluster.add_child(marker)
```

#### SpaceX Dash App


[SpaceX Dash App Assignment Guide](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/labs/module_3/lab_theia_plotly_dash.md.html)

Plotly Dash Reference links:
- [dcc.Dropdown (input)](https://dash.plotly.com/dash-core-components/dropdown)
- [dcc.RangeSlider (input)](https://dash.plotly.com/dash-core-components/rangeslider)
- [Pie Charts (output)](https://plotly.com/python/pie-charts/)
- [Scatter Plots (output)](https://plotly.com/python/line-and-scatter/)


<br/>
<div align="right">
    <b><a href="#top">↥ back to top</a></b>
</div>
<br/>

