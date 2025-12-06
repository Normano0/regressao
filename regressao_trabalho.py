#Aluno Kauan Cruvinel Wehbe RA: 5173989
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd

#para rodar o código digite no terminal: streamlit run regressao_trabalho.py
def calcular_r2(X, y, theta):
    m = X.shape[0]
    ones_col = np.ones((m, 1))
    Xb = np.hstack((ones_col, X))
    
    y_pred = Xb @ theta
    y_mean = np.mean(y)
    
    SST = np.sum((y - y_mean)**2)
    SSE = np.sum((y - y_pred)**2)
    
    R2 = 1 - (SSE / SST)
    return R2

def regressao_linear_matriz(X, y):

    m, n = X.shape

    ones_col = np.ones((m, 1))
    Xb = np.hstack((ones_col, X))  
    Xt = Xb.T
    XtX = Xt @ Xb

    XtX_inv = np.linalg.inv(XtX)

    Xty = Xt @ y
    theta = XtX_inv @ Xty

    return theta

def plot_3d(X, y, theta):
    theta_0 = theta[0][0]
    theta_1 = theta[1][0]
    theta_2 = theta[2][0]

    fig = px.scatter_3d(
        x=X[:, 0], y=X[:, 1], z=y.flatten(),
        labels={'x': 'Horas de Estudo', 'y': 'Horas de Sono', 'z': 'Notas'},
        title='Regressão Múltipla (ambos os dados): Plano de Ajuste (Coeficiente de Determinação={:.4f})'.format(calcular_r2(X, y, theta)),
        color=y.flatten(),
        opacity=0.8
    )

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    x1_surf = np.linspace(x1_min, x1_max, 30)
    x2_surf = np.linspace(x2_min, x2_max, 30)
    x1_surf, x2_surf = np.meshgrid(x1_surf, x2_surf)

    y_surf = theta_0 + theta_1 * x1_surf + theta_2 * x2_surf

    fig.add_trace(go.Surface(
        x=x1_surf, y=x2_surf, z=y_surf,
        name='Plano de Regressão',
        opacity=0.5,
        colorscale=px.colors.sequential.Plasma,
        showscale=False
    ))
    
    fig.update_layout(scene=dict(zaxis_title='Notas'))
    return fig

def plot_2d(X_data, y_data, theta, title_prefix, xlabel):

    theta_0 = theta[0][0]
    theta_1 = theta[1][0]

    x_line = np.linspace(X_data.min(), X_data.max(), 50).reshape(-1, 1)
    y_line = theta_0 + theta_1 * x_line

    df = pd.DataFrame({
        xlabel: X_data.flatten(), 
        'Notas': y_data.flatten()
    })

    fig = px.scatter(
        df, x=xlabel, y='Notas', 
        title=f'{title_prefix} (Coeficiente de Determinação={calcular_r2(X_data, y_data, theta):.4f})',
        opacity=0.8
    )

    fig.add_trace(go.Scatter(
        x=x_line.flatten(), y=y_line.flatten(), 
        mode='lines', name='Linha de Regressão',
        line=dict(color='red', width=3)
    ))
    return fig

X = np.array([
    [2, 5],
    [3, 6],
    [4, 5],
    [5, 7],
    [6, 6],
    [7, 8],
    [8, 7],
    [9, 6],
    [10, 8],
    [11, 7]
])

y = np.array([55, 63, 66, 74, 72, 85, 88, 90, 95, 96]).reshape(-1, 1) #notas

theta = regressao_linear_matriz(X, y)

estudo = X[:, 0].reshape(-1, 1)
theta_simples1 = regressao_linear_matriz(estudo, y)

sono = X[:, 1].reshape(-1, 1)
theta_simples2 = regressao_linear_matriz(sono, y)


st.title("Dashboard de Análise sobre desempenho de Alunos")

st.write("Neste gráfico é possível ver a relação entre as 3 variáveis em um plano 3D")
g1 = plot_3d(X, y, theta)
st.plotly_chart(g1)

st.write("Perceba que a quantidade de horas de estudo tem uma correlação forte com as notas, " \
"O coeficiente de determinação indica que a cada hora estudada a nota aumenta em aproximadamente 4.88 pontos")

g2 = plot_2d(estudo, y, theta_simples1, "Relação entre horas de estudo e notas", "Horas de Estudo")
st.plotly_chart(g2)

st.write("Por outro lado o número de horas de sono tem um impacto muito menor nas notas. Contudo ainda contribuem para o desempenho " \
"de maneira positiva")

g3 = plot_2d(sono, y, theta_simples2, "Relação entre  Horas de Sono e Notas", "Horas de Sono")
st.plotly_chart(g3)