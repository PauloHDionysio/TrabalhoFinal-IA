# Importações necessárias
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MaxAbsScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
import plotly.express as px
# from tensorflow.keras.models import load_model

# Configuração inicial
st.title("Previsão de Preços de Casas")
st.subheader("Trabalho Final de Inteligência Artificial")
st.write("Aluno: Paulo Henrique Dionysio | RA: 221026169")

# Navegação
menu = st.sidebar.radio(
    "Selecione a função de ativação para o modelo:",
    ("ReLU", "Sigmoid", "Tanh")
)

# Função para carregar e preparar os dados
@st.cache
def carregar_dados(treino_file, teste_file):
    dados_treino = pd.read_csv(treino_file)
    dados_teste = pd.read_csv(teste_file)
    dados = pd.concat([dados_treino, dados_teste], sort=False)
    return dados_treino, dados_teste, dados

# Função para pré-processar os dados
def preprocessar_dados(dados, dados_treino, scaler, imputer):
    # Transformando categóricas em dummies
    cat_cols = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
                'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
                'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
                'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
                'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
                'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
                'GarageCond', 'PavedDrive', 'PoolQC', 'MiscFeature', 'SaleType',
                'SaleCondition', 'Fence']
    juntos_num = pd.get_dummies(dados, columns=cat_cols)
    
    # Separando treino e teste
    treino_num = juntos_num.iloc[:len(dados_treino), :]
    teste_num = juntos_num.iloc[len(dados_treino):, :]

    X = treino_num.drop(["Id", "SalePrice"], axis=1)
    y = treino_num["SalePrice"]
    Xtr, Xval, ytr, yval = train_test_split(X, y, train_size=0.5, random_state=0)

    Xtr = scaler.fit_transform(imputer.fit_transform(Xtr))
    Xval = scaler.transform(imputer.transform(Xval))
    ytr = ytr.values
    yval = yval.values
    return Xtr, Xval, ytr, yval, treino_num, teste_num

# Configuração da página
if "model" not in st.session_state:
    st.session_state["model"] = None

# Upload dos arquivos de treino e teste
treino_file = st.file_uploader("Carregue o arquivo de treino (train.csv)", type="csv")
teste_file = st.file_uploader("Carregue o arquivo de teste (test.csv)", type="csv")

if treino_file and teste_file:
    dados_treino, dados_teste, dados = carregar_dados(treino_file, teste_file)

    scaler = MaxAbsScaler()
    imputer = SimpleImputer(strategy="median")
    Xtr, Xval, ytr, yval, treino_num, teste_num = preprocessar_dados(
        dados, dados_treino, scaler, imputer
    )

    st.write("Amostra dos Dados de Treino:")
    st.write(dados_treino.head())

    # Configuração do modelo
    if menu == "ReLU":
        ativacao = "relu"
    elif menu == "Sigmoid":
        ativacao = "sigmoid"
    else:  # Tanh
        ativacao = "tanh"

    if st.button(f"Treinar Modelo com {ativacao}"):
        tf.random.set_seed(2)

        # Construindo a rede neural
        inp = tf.keras.layers.Input((Xtr.shape[1],))
        hid1 = tf.keras.layers.Dense(100, activation=ativacao)(inp)
        drop = tf.keras.layers.Dropout(0.5)(hid1)
        hid2 = tf.keras.layers.Dense(50, activation=ativacao)(drop)
        hid3 = tf.keras.layers.Dense(25, activation=ativacao)(hid2)
        out = tf.keras.layers.Dense(1, activation="relu")(hid3)

        mdl = tf.keras.Model(inp, out)
        mdl.compile(loss="mean_squared_logarithmic_error", optimizer="adam")
        es = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", min_delta=0.1, patience=10, mode="min", restore_best_weights=True
        )

        # Treinando o modelo
        with st.spinner(f"Treinando o modelo com {ativacao}, aguarde..."):
            history = mdl.fit(Xtr, ytr, validation_data=(Xval, yval), epochs=100, shuffle=True, batch_size=1, callbacks=[es])

        st.session_state["model"] = mdl
        st.success(f"Modelo treinado com sucesso usando {ativacao}!")

        # Gerando gráfico de treino e validação
        st.write("Gráfico de Perda durante o Treinamento:")
        fig = px.line(
            x=range(len(history.history["loss"])),
            y=[history.history["loss"], history.history["val_loss"]],
            labels={"x": "Épocas", "y": "Perda"},
            title="Perda de Treinamento vs Validação"
        )
        fig.update_traces(mode="lines+markers")
        fig.update_layout(
            legend=dict(
                title=dict(text="Tipo de Perda"),
                orientation="h",  # Alinha a legenda horizontalmente
                font=dict(size=12, color="black"),
                x=0.5,  # Centraliza a legenda horizontalmente
                xanchor="center",
                y=-0.2  # Move a legenda para fora da área do gráfico
            )
        ) # Atualiza o layout do gráfico


    # Prevendo com os dados de teste
    if st.button("Prever Preços das Casas"):
        if st.session_state["model"] is None:
            st.error("Por favor, treine o modelo primeiro!")
        else:
            teste1 = teste_num.drop(["Id", "SalePrice"], axis=1)
            teste = scaler.transform(imputer.transform(teste1))
            p_teste = st.session_state["model"].predict(teste).squeeze()
            resultados = pd.DataFrame({"Id": teste_num["Id"], "SalePrice": p_teste})

            st.write("Distribuição das Previsões de Preços:")
            hist_fig = px.histogram(resultados, x="SalePrice", title="Distribuição de Previsões de Preços")
            st.plotly_chart(hist_fig)

            csv = resultados.to_csv(index=False)
            st.download_button("Baixar Previsões", csv, "previsoes.csv", "text/csv")
