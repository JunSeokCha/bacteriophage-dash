from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import numpy as np
import pandas as pd

import pathlib

from app import app

from utils import (
    make_dash_table,
    Header,
    NamedInput
)

# model description page layout
PATH=pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath('../data').resolve()
df_variables = pd.read_csv(DATA_PATH.joinpath(f"./model1/df_variables.csv"))
df_parameters = pd.read_csv(DATA_PATH.joinpath(f'./model1/df_parameters.csv'))
df_equation = pd.read_csv(DATA_PATH.joinpath(f'./model1/df_equation.csv'))

layout = html.Div(
        [
            #header
            Header(app),

            # page 1
            html.Div(
                [
                    # Row
                    html.Div(
                        [
                            html.H6(
                                ["Variables"], className="subtitle padded"
                            ),
                            html.Table(
                                id='df-variables',
                                children = make_dash_table(df_variables)
                            ),
                        ],
                        className="row",
                        style={"margin-bottom": "35px"},
                    ),
                    # Row
                    html.Div(
                        [
                            html.H6(
                                ["Parameters"], className="subtitle padded"
                            ),
                            html.Table(
                                id = 'df-parameters',
                                children = make_dash_table(df_parameters)
                            ),
                        ],
                        className="row",
                        style={"margin-bottom": "35px"},
                    ),
                    # Row
                    html.Div(
                        [
                            html.H6(
                                ["Model"], className="subtitle padded"
                            ),
                            html.Table(
                                id = 'df-equation',
                                children = make_dash_table(df_equation)
                            )
                        ],
                        className="row",
                        style={"margin-bottom": "35px"},
                    ),
                ],
                className="sub_page",
            ),
        ],
        className="page"
    )