# Import required libraries
import pandas as pd
import plotly.graph_objects as go
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output

# Read the airline data into pandas dataframe
airline_data =  pd.read_csv('airline_data.csv', 
                            encoding = "ISO-8859-1",
                            dtype={'Div1Airport': str, 'Div1TailNum': str, 
                                   'Div2Airport': str, 'Div2TailNum': str})


# Create a dash application
app = dash.Dash(__name__)

# Get the layout of the application and adjust it.
# Create an outer division using html.Div and add title to the dashboard using html.H1 component
# Add a html.Div and core input text component
# Finally, add graph component.
app.layout = html.Div(children=[
        html.H1("Airline Performance Dashboard",
                style={'textAlign': 'center', 
                        'color': '#503D36', 
                        'font-size': 40}),
        html.Div(["Input Year", 
                  dcc.Input(id='input-year', 
                            type='number', 
                            value='2010', 
                            style={'height': '50px', 
                                'font-size': 35}),], 
        style={'font-size': 40}),
        html.Br(),
        html.Br(),
        html.Div(dcc.Graph(id='line-plot')),
    ])


# add callback decorator
@app.callback(Output(component_id='line-plot', component_property='figure'),
               Input(component_id='input-year', component_property='value'))

# Add computation to callback function and return graph
def get_graph(entered_year):
    # Select data based on the entered year
    df =  airline_data[airline_data['Year']==int(entered_year)]

    # Group the data by Month and compute average over arrival delay time.
    line_data = df.groupby('Month')['ArrDelay'].mean().reset_index()

    # 
    fig = go.Figure(data=go.Scatter(x=line_data['Month'],
                                    y=line_data['ArrDelay'],
                                    mode='lines',
                                    marker=dict(color='green')))
    fig.update_layout(title='Month vs Average Flight Delay Time',
                      xaxis_title="Month",
                      yaxis_title='ArrDelay')
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server()