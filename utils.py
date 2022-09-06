from dash import dcc, html

import pandas as pd
import numpy as np
import scipy as sp
import scipy.integrate

import plotly.express as px

def Header(app):
    return html.Div([get_header(app), get_menu(), get_model_kind()])

def get_header(app):
    header = html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.A(
                                html.Img(
                                    src = app.get_asset_url('yonsei_logo.png'),
                                    className = 'logo',
                                ),
                                href='http://sevcpt.yuhs.ac/',
                                style = {'float':'left'}
                            ),
                            html.A(
                                html.H1(
                                    children = 'SEVCPT',
                                    style = {
                                        'font-size':'30px',
                                        'margin-top': '20px'
                                    }
                                ),
                                href='http://sevcpt.yuhs.ac/',
                                style = {
                                    'float':'left',
                                }
                            ),
                        ],
                        className="twelve columns",
                        style={"padding-left": "0"},
                    ),
                ],
                className="row",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [html.H5("Bacteriophage Cocktail Simulator")],
                                className="seven columns main-title",
                            )
                        ],
                        className="twelve columns",
                        style={"padding-left": "0"},
                    ),
                ],
                className="row",
            )
        ]
    )
    return header

def get_menu():
    menu = html.Div(
        [
            dcc.Link(
                'Model Description',
                href = '/bacteriophage/model-description',
                className='tab',
            ),
            dcc.Link(
                'Model Simulation',
                href = '/bacteriophage/simulate-model',
                className = 'tab'
            ),
        ],
        className = 'row all-tabs'
    )
    return menu

def get_model_kind():
    return html.Div(
        [
            html.Div(
                [
                    dcc.Dropdown(
                        ['model1', 'model2'],
                        'model1',
                        id = 'model-kind-input'
                    )
                ],
                className = 'twelve columns'
            )
        ],
        className = 'row',
        style = {
            'margin-bottom': '30px',
            'margin-top': '30px',
            'margin-left': '30px',
            'margin-right': '30px'
        }
    )

def make_dash_table(df):
    '''
    This function was taken from https://github.com/plotly/dash-sample-apps/blob/main/apps/dash-financial-report/utils.py
    '''
    table = []
    for index, row in df.iterrows():
        html_row = []
        for i in range(len(row)):
            html_row.append(html.Td([row[i]]))
        table.append(html.Tr(html_row))
    return table

def NamedInput(name, **kwargs):
    return html.Div(
        style={
            "margin": "0px 0px",
            'float': 'left'
        },
        children=[
            html.P(children=f"{name}:", style={"margin-left": "3px"}),
            dcc.Input(**kwargs,
                style = {'width': '150px'}
            ),
        ],
    )