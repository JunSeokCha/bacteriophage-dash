from dash import dcc, html

import numpy as np
import scipy as sp
import pandas as pd
import scipy.integrate

import pathlib
import plotly.express as px

from utils import NamedInput, Header


model_name = 'model2'

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
values = [0, 0, 0, 0, 1.5e7/4, 1.5e7/4, 1.5e7/4, 1.5e7/4, 2, 2, 10**8.16, 0.465, 0.355, 0.158]
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

# the functions

def model(time, state, 
    B0_123, B0_12, B0_13, B0_23, B0_1, B0_2, B0_3, B0_sus,
    lam0, lam1, Bmax, 
    k1, b1, ktr1, frac1, pr1, pr21, 
    k2, b2, ktr2, frac2, pr2, pr22,
    k3, b3, ktr3, frac3, pr3, pr23,
    moi1, moi2, moi3
    ):

    R123, R12, R13, R23, R1, R2, R3, B, P1, P2, P3, I11, I21, I31, I12, I22, I32, I13, I23, I33 = state


    # fraction of lowering virulence for double resistant subpopulations (let the lowering of virulence for resistance for different phages be multiplicative)
    frac12 = frac1*frac2
    frac13 = frac1*frac3
    frac23 = frac2*frac3
    frac123 = frac1*frac2*frac3

    # amount of total bacteria
    Itot = I11 + I21 + I31 + I12 + I22 + I32 + I13 + I23 + I33
    Btot = B + R1 + R2 + R3 + R12 + R13 + R23 + R123 + Itot

    # amount of susceptible supbpopulations
    S1 = Btot - R1 - R12 - R13 - R123
    S2 = Btot - R2 - R12 - R23 - R123
    S3 = Btot - R3 - R13 - R23 - R123

    # equations for phage dynamics
    ddt_P1 = -k1*P1*S1 + b1*ktr1*(1 - pr21)*(I11 + I21 + I31)
    ddt_P2 = -k2*P2*S2 + b2*ktr2*(1 - pr22)*(I12 + I22 + I32)
    ddt_P3 = -k3*P3*S3 + b3*ktr3*(1 - pr23)*(I13 + I23 + I33)

    # equations for non-resistant bacteria dynamics
    ddt_B = lam0 * B *(1 - (Btot/(Bmax))**lam1) - (k1*P1 + k2*P2 + k3*P3)*B

    # equations for mono resistant bacteria dynamics
    ddt_R1 = lam0 * R1 *(1 - (Btot/(frac1 * Bmax))**lam1) - (k2*P2 + k3*P3)*R1
    ddt_R2 = lam0 * R2 *(1 - (Btot/(frac2 * Bmax))**lam1) - (k1*P1 + k3*P3)*R2
    ddt_R3 = lam0 * R3 *(1 - (Btot/(frac3 * Bmax))**lam1) - (k1*P1 + k2*P2)*R3

    # equations for double resistant bacteria dynamics
    ddt_R12 = lam0 * R12 *(1 - (Btot/(frac12 * Bmax))**lam1) - k3*P3*R12
    ddt_R13 = lam0 * R13 *(1 - (Btot/(frac13 * Bmax))**lam1) - k2*P2*R13
    ddt_R23 = lam0 * R23 *(1 - (Btot/(frac23 * Bmax))**lam1) - k1*P1*R23

    # equations for triple resistant bacteria dynamics
    ddt_R123 = lam0 * R123 *(1 - (Btot/(frac123 * Bmax))**lam1)

    # equations for infected bacteria
    ddt_I11 = lam0 * I11 *(1 - (Btot/(Bmax))**lam1) + pr1 * k1*P1*S1 - ktr1*I11
    ddt_I21 = lam0 * I21 *(1 - (Btot/(Bmax))**lam1) + pr21 * ktr1*I11 - ktr1*I21
    ddt_I31 = lam0 * I31 *(1 - (Btot/(Bmax))**lam1) + pr21 * ktr1*I21 - ktr1*I31

    ddt_I12 = lam0 * I12  *(1 - (Btot/(Bmax))**lam1) + pr2 * k2*P2*S2 - ktr2*I12
    ddt_I22 = lam0 * I22  *(1 - (Btot/(Bmax))**lam1) + pr22 * ktr2*I12 - ktr2*I22
    ddt_I32 = lam0 * I32  *(1 - (Btot/(Bmax))**lam1) + pr22 * ktr2*I22 - ktr2*I32

    ddt_I13 = lam0 * I13  *(1 - (Btot/(Bmax))**lam1) + pr3 * k3*P3*S3 - ktr3*I13
    ddt_I23 = lam0 * I23  *(1 - (Btot/(Bmax))**lam1) + pr23 * ktr3*I13 - ktr3*I23
    ddt_I33 = lam0 * I33  *(1 - (Btot/(Bmax))**lam1) + pr23 * ktr3*I23 - ktr3*I33

    return [ddt_R123, ddt_R12, ddt_R13, ddt_R23, ddt_R1, ddt_R2, ddt_R3, ddt_B, ddt_P1, ddt_P2, ddt_P3, ddt_I11, ddt_I21, ddt_I31, ddt_I12, ddt_I22, ddt_I32, ddt_I13, ddt_I23, ddt_I33]


def simulate(times, 
    B0_123, B0_12, B0_13, B0_23, B0_1, B0_2, B0_3, B0_sus,
    lam0, lam1, Bmax, 
    k1, b1, ktr1, frac1, pr1, pr21, 
    k2, b2, ktr2, frac2, pr2, pr22,
    k3, b3, ktr3, frac3, pr3, pr23,
    moi1, moi2, moi3):
    params = tuple([
        B0_123, B0_12, B0_13, B0_23, B0_1, B0_2, B0_3, B0_sus,
        lam0, lam1, Bmax, 
        k1, b1, ktr1, frac1, pr1, pr21, 
        k2, b2, ktr2, frac2, pr2, pr22,
        k3, b3, ktr3, frac3, pr3, pr23,
        moi1, moi2, moi3
    ])
    state0 = [
        B0_123,
        B0_12, B0_13, B0_23, 
        B0_1, B0_2, B0_3,
        B0_sus,
        1.5e7*moi1, 1.5e7*moi2, 1.5e7*moi3,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
    ]
    try:
        res = sp.integrate.solve_ivp(
            fun = model,
            t_span = [0, 24],
            method = 'BDF',
            args = params,
            y0 = state0,
            t_eval = times
        )
    except Exception:
        print('failed')
        return 'failed'
    else:
        return res

def plot(res, variable_kind):
    if res == 'failed':
        df = pd.DataFrame({
            'time':[0],
            'y':[0]
        })
        fig = px.line(
            df,
            x = 'time',
            y= 'y'
        )
        fig.add_trace(
            x = [0],
            y = [0],
            mode = 'lines+text',
            text = 'failed'
        )
    else:
        df = pd.DataFrame({
            'time':res.t,
            'B_tot':(res.y[0] + res.y[1] + res.y[2] + res.y[3] + res.y[4] + res.y[5] + res.y[6] + res.y[7] + res.y[11] + res.y[12] + res.y[13] + res.y[14] + res.y[15] + res.y[16] + res.y[17] + res.y[18] + res.y[19])/2e8,
            'B_123': res.y[0]/2e8,
            'B_12': res.y[1]/2e8,
            'B_13': res.y[2]/2e8,
            'B_23': res.y[3]/2e8,
            'B_1': res.y[4]/2e8,
            'B_2': res.y[5]/2e8,
            'B_3': res.y[6]/2e8,
            'B_sus': res.y[7]/2e8,
            'P_1': res.y[8]/2e8,
            'P_2': res.y[9]/2e8,
            'P_3': res.y[10]/2e8,
            'I_11': res.y[11]/2e8,
            'I_21': res.y[12]/2e8,
            'I_31': res.y[13]/2e8,
            'I_12': res.y[14]/2e8,
            'I_22': res.y[15]/2e8,
            'I_32': res.y[16]/2e8,
            'I_13': res.y[17]/2e8,
            'I_23': res.y[18]/2e8,
            'I_33': res.y[19]/2e8
        })
        fig = px.line(
            df,
            x = 'time',
            y = variable_kind
        )
    return fig