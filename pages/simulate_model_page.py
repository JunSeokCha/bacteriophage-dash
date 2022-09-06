from dash import dcc, html

import numpy as np
import scipy as sp
import pandas as pd
import scipy.integrate

import pathlib
import plotly.express as px

from utils import NamedInput, Header

from app import app

model_name = 'model1'

# make content for setting parameter values
set_parameter_values_content = []

#Bacterial Growth Parameters
bacterial_growth_parameters_content = []

bacterial_growth_parameters_content.append(
    html.H6(
        ['Bacterial Growth Parameters'],
        className = 'subtitle padded'
    )
)

names = ['\(B_{123}^{0}\)', '\(B_{12}^{0}\)', '\(B_{13}^{0}\)', '\(B_{23}^{0}\)', '\(B_{1}^{0}\)', '\(B_{2}^{0}\)', '\(B_{3}^{0}\)', '\(B_{sus}^{0}\)', '\(\lambda_{0}\)', '\(\lambda_{1}\)', '\(B_{max}\)', '\(frac_{1}\)', '\(frac_{2}\)', '\(frac_{3}\)']
ids = ['B0_123-input', 'B0_12-input', 'B0_13-input', 'B0_23-input', 'B0_1-input', 'B0_2-input', 'B0_3-input', 'B0_sus-input', 'lam0-input', 'lam1-input', 'Bmax-input', 'frac1-input', 'frac2-input', 'frac3-input']
ids = [f'{model_name}-{i}' for i in ids]
values = [0, 0, 0, 0, 1.5e7/4, 1.5e7/4, 1.5e7/4, 1.5e7/4, 2.45, 10**7.71, 10**8.16, 0.465, 0.355, 0.158]
for i, j, k in zip(names, ids, values):
    bacterial_growth_parameters_content.append(
        NamedInput(
            name = i,
            id = j,
            type = 'number',
            value = k
        )
    )

set_parameter_values_content.append(
    html.Div(
        html.Div(
            bacterial_growth_parameters_content,
            className = 'twelve columns'
        ),
        className = 'row'
    )       
)

# Phage 1 Parameters
phage1_parameters_content = []

phage1_parameters_content.append(
    html.H6(
        ['Phage 1 Parameters'],
        className = 'subtitle padded'
    )
)

names = ['\(k_{1}\)', '\(b_{1}\)', '\(k_{tr1}\)', '\(pr_{1}\)', '\(pr_{21}\)']
ids = ['k1-input', 'b1-input', 'ktr1-input','pr1-input', 'pr21-input']
ids = [f'{model_name}-{i}' for i in ids]
values = [10**(-5.88), 546, 1.47, 0.115, 0.796]

for i, j, k in zip(names, ids, values):
    phage1_parameters_content.append(
        NamedInput(
            name = i,
            id = j,
            type = 'number',
            value = k,
        )
    )

set_parameter_values_content.append(
    html.Div(
        html.Div(
            phage1_parameters_content,
            className = 'twelve columns'
        ),
        className = 'row'
    )       
)

# Phage 2 Parameters
phage2_parameters_content = []

phage2_parameters_content.append(
    html.H6(
        ['Phage 2 Parameters'],
        className = 'subtitle padded'
    )
)

names = ['\(k_{2}\)', '\(b_{2}\)', '\(k_{tr2}\)', '\(pr_{2}\)', '\(pr_{22}\)']
ids = ['k2-input', 'b2-input', 'ktr2-input','pr2-input', 'pr22-input']
ids = [f'{model_name}-{i}' for i in ids]
values = [10**(-6.62), 123, 1.28, 0.373, 0.363]

for i, j, k in zip(names, ids, values):
    phage2_parameters_content.append(
        NamedInput(
            name = i,
            id = j,
            type = 'number',
            value = k
        )
    )

set_parameter_values_content.append(
    html.Div(
        html.Div(
            phage2_parameters_content,
            className = 'twelve columns'
        ),
        className = 'row'
    )       
)

# Phage 3 Parameters
phage3_parameters_content = []

phage3_parameters_content.append(
    html.H6(
        ['Phage 3 Parameters'],
        className = 'subtitle padded'
    )
)

names = ['\(k_{3}\)', '\(b_{3}\)', '\(k_{tr3}\)', '\(pr_{3}\)', '\(pr_{23}\)']
ids = ['k3-input', 'b3-input', 'ktr3-input','pr3-input', 'pr23-input']
ids = [f'{model_name}-{i}' for i in ids]
values = [10**(-7.413), 327, 1.05, 0.339, 0.461]

for i, j, k in zip(names, ids, values):
    phage3_parameters_content.append(
        NamedInput(
            name = i,
            id = j,
            type = 'number',
            value = k
        )
    )

set_parameter_values_content.append(
    html.Div(
        html.Div(
            phage3_parameters_content,
            className = 'twelve columns'
        ),
        className = 'row'
    )       
)

# phage dose
phage_dose_parameters_content = []

phage_dose_parameters_content.append(
    html.H6(
        ['Phage Dose Parameters'],
        className = 'subtitle padded'
    )
)

names = ['\(moi1\)', '\(moi2\)', '\(moi3\)']
ids = ['moi1-input', 'moi2-input', 'moi3-input']
ids = [f'{model_name}-{i}' for i in ids]
values = [1e-1, 0, 0]

for i, j, k in zip(names, ids, values):
    phage_dose_parameters_content.append(
        NamedInput(
            name = i,
            id = j,
            type = 'number',
            value = k
        )
    )

set_parameter_values_content.append(
    html.Div(
        html.Div(
            phage_dose_parameters_content,
            className = 'twelve columns'
        ),
        className = 'row'
    )       
)

# make content for plotting the simulation results
simulation_results_content = []

simulation_results_content.append(
    html.Div(
        html.Div(
            dcc.Dropdown(
                options = {
                    'B_tot':'\(B_{tot}\)',
                    'B_123':'\(B_{123}\)', 
                    'B_12':'\(B_{12}\)',
                    'B_13':'\(B_{13}\)',
                    'B_23':'\(B_{23}\)',
                    'B_1':'\(B_{1}\)',
                    'B_2':'\(B_{2}\)',
                    'B_3':'\(B_{3}\)',
                    'B_sus':'\(B_{sus}\)',
                    'P_1':'\(P_{1}\)',
                    'P_2':'\(P_{2}\)',
                    'P_3':'\(P_{3}\)',
                    'I_11':'\(I_{11}\)',
                    'I_21':'\(I_{21}\)',
                    'I_31':'\(I_{31}\)',
                    'I_12':'\(I_{12}\)',
                    'I_22':'\(I_{22}\)',
                    'I_32':'\(I_{32}\)',
                    'I_13':'\(I_{13}\)',
                    'I_23':'\(I_{23}\)',
                    'I_33':'\(I_{33}\)'
                },
                value = 'B_tot',
                id = f'{model_name}-variable-kind'
            ),
            className = 'twelve columns'
        ),
        className = 'row'
    )
)

simulation_results_content.append(
    dcc.Graph(id = f'{model_name}-simulation-plot')
)

layout = html.Div(
        [
            #header
            Header(app),
            # page 1
            html.Div(
                [
                            
                    html.H5(
                        ["Set Parmeter Values"], className="subtitle padded"
                    ),
                    html.Div(
                        set_parameter_values_content,
                        id = 'set-parameter-values-content'    
                    ),

                    html.Div(
                        [
                            html.H6(
                                ["Simulation Result"], className="subtitle padded"
                            ),
                            html.Div(
                                simulation_results_content,
                                id = 'simulation-results-content'    
                            )
                        ],
                        className='row',
                        style={"margin-bottom": "35px"},
                    )
                ],
                className="sub_page",
            ),
        ],
        className="page",
    )