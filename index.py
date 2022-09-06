from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from app import app
from pages import (
    model_description_page,
    simulate_model_page
)

from utils import (
    make_dash_table,
    Header,
    NamedInput
)

import numpy as np
import pandas as pd

import pathlib


import data.model1.simulateModelContent
import data.model2.simulateModelContent

# layout of app
app.layout = html.Div(
    [dcc.Location(id = 'url', refresh = False), html.Div(id='page-content')]
)

# layout for validation
app.validation_layout = html.Div(
    [
        model_description_page.layout,
        simulate_model_page.layout,
        data.model1.simulateModelContent.set_parameter_values_content,
        data.model1.simulateModelContent.simulation_results_content,
        data.model2.simulateModelContent.set_parameter_values_content,
        data.model2.simulateModelContent.simulation_results_content,
    ]
)

# update page
@app.callback(Output('page-content', 'children'), [Input('url', 'pathname')])
def display_page(pathname):
    print('CallBack: display_page')
    if pathname == '/bacteriophage/model-description':
        return model_description_page.layout
    elif pathname == '/bacteriophage/simulate-model':
        return simulate_model_page.layout
    else:
        return model_description_page.layout

# other callbacks
@app.callback(
    Output('df-variables', 'children'),
    Output('df-parameters', 'children'),
    Output('df-equation', 'children'),
    Input('model-kind-input', 'value')
)
def model_description_content(model_kind):
    print('Callback: model_description_content')
    PATH=pathlib.Path(__file__).parent
    print(PATH)
    DATA_PATH = PATH.joinpath('./data').resolve()
    df_variables = pd.read_csv(DATA_PATH.joinpath(f"./{model_kind}/df_variables.csv"))
    df_parameters = pd.read_csv(DATA_PATH.joinpath(f'./{model_kind}/df_parameters.csv'))
    df_equation = pd.read_csv(DATA_PATH.joinpath(f'./{model_kind}/df_equation.csv'))
    return make_dash_table(df_variables), make_dash_table(df_parameters), make_dash_table(df_equation)

@app.callback(
    Output(f'set-parameter-values-content', 'children'),
    Output(f'simulation-results-content', 'children'),
    Input('model-kind-input', 'value')
)
def model1_simulate_model_content(model_kind):
    print('Callback: model1_simulate_model_content')
    if model_kind == 'model1':
        return data.model1.simulateModelContent.set_parameter_values_content, data.model1.simulateModelContent.simulation_results_content
    elif model_kind == 'model2':
        return data.model2.simulateModelContent.set_parameter_values_content, data.model2.simulateModelContent.simulation_results_content

# model1
model_name = 'model1'
        
parameter_names = ['B0_123', 'B0_12', 'B0_13', 'B0_23', 'B0_1', 'B0_2', 'B0_3', 'B0_sus', 'lam0', 'lam1', 'Bmax', 'k1', 'b1', 'ktr1', 'frac1', 'pr1', 'pr21', 'k2', 'b2', 'ktr2', 'frac2', 'pr2', 'pr22', 'k3', 'b3', 'ktr3', 'frac3', 'pr3', 'pr23', 'moi1', 'moi2', 'moi3']
ids = [f'{model_name}-{i}-input' for i in parameter_names]
list1 = [Input(i, 'value') for i in ids]
@app.callback(
    Output(f'{model_name}-simulation-plot', 'figure'),
    Input(f'{model_name}-variable-kind', 'value'),
    Input('model-kind-input', 'value'),
    *list1
)
def model1_simulate_and_plot(variable_kind, model_kind, *args):
    print('Callback: model1_simulate_and_plot')
    if model_kind == 'model1':
        times = np.arange(0, 24, 0.1)

        res = data.model1.simulateModelContent.simulate(times,*args)

        fig = data.model1.simulateModelContent.plot(res, variable_kind)

        return fig
    else:
        raise PreventUpdate

# model2
model_name = 'model2'

parameter_names = ['B0_123', 'B0_12', 'B0_13', 'B0_23', 'B0_1', 'B0_2', 'B0_3', 'B0_sus', 'lam0', 'lam1', 'Bmax', 'k1', 'b1', 'ktr1', 'frac1', 'pr1', 'pr21', 'k2', 'b2', 'ktr2', 'frac2', 'pr2', 'pr22', 'k3', 'b3', 'ktr3', 'frac3', 'pr3', 'pr23', 'moi1', 'moi2', 'moi3']
ids = [f'{model_name}-{i}-input' for i in parameter_names]
list1 = [Input(i, 'value') for i in ids]
@app.callback(
    Output(f'{model_name}-simulation-plot', 'figure'),
    Input(f'{model_name}-variable-kind', 'value'),
    Input('model-kind-input', 'value'),
    *list1
)
def model2_simulate_and_plot(variable_kind, model_kind, *args):
    print('Callback: model2_simulate_and_plot')
    if model_kind == 'model2':
        times = np.arange(0, 24, 0.1)

        res = data.model2.simulateModelContent.simulate(times,*args)

        fig = data.model2.simulateModelContent.plot(res, variable_kind)

        return fig
    else:
        raise PreventUpdate