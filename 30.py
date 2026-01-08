import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# Configuração da página
st.set_page_config(page_title="Compradores Decididos", page_icon="🛍️", layout="wide")
st.title("Compradores Decididos")
# Carregar dados
@st.cache_data
def load_data():
    df = pd.read_csv("online_shoppers_intention.csv")
    return df

df = load_data()

# Traduzir colunas
traducao_colunas = {
    "Administrative": "Administrativo",
    "Administrative_Duration": "Duração_Administrativo",
    "Informational": "Informativo",
    "Informational_Duration": "Duração_Informativo",
    "ProductRelated": "Relacionado_Produto",
    "ProductRelated_Duration": "Duração_Relacionado_Produto",
    "BounceRates": "Taxa_Rejeição",
    "ExitRates": "Taxa_Saída",
    "PageValues": "Valor_Página",
    "SpecialDay": "Dia_Especial",
    "Month": "Mês",
    "OperatingSystems": "Sistemas_Operacionais",
    "Browser": "Navegador",
    "Region": "Região",
    "TrafficType": "Tipo_Tráfego",
    "VisitorType": "Tipo_Visitante",
    "Weekend": "Fim_de_Semana",
    "Revenue": "Compra"
}
df.rename(columns=traducao_colunas, inplace=True)

st.success("Dados carregados com sucesso!")
st.dataframe(df.head())

# Sidebar com filtros
st.sidebar.header("Filtros")
mes = st.sidebar.selectbox("Mês", sorted(df["Mês"].unique()))
tipo_visitante = st.sidebar.selectbox("Tipo de visitante", df["Tipo_Visitante"].unique())
fim_semana = st.sidebar.radio("Fim de semana?", ["True", "False"])
fim_semana_bool = True if fim_semana == "True" else False

df_filtrado = df[
    (df["Mês"] == mes) &
    (df["Tipo_Visitante"] == tipo_visitante) &
    (df["Fim_de_Semana"] == fim_semana_bool)
]

# Seleção de variáveis para agrupamento
variaveis = [
    "Administrativo", "Duração_Administrativo",
    "Informativo", "Duração_Informativo",
    "Relacionado_Produto", "Duração_Relacionado_Produto",
    "Valor_Página", "Dia_Especial", "Mês", "Fim_de_Semana"
]

X = df_filtrado[variaveis].copy()

# Codificação
le = LabelEncoder()
X["Mês"] = le.fit_transform(X["Mês"])
X["Fim_de_Semana"] = X["Fim_de_Semana"].astype(int)

# Padronização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Amostra para dendrograma
X_amostra = X_scaled[:500]
Z_amostra = linkage(X_amostra, method="ward")

# Dendrograma
st.subheader("Dendrograma (amostra de 500 registros)")
fig, ax = plt.subplots(figsize=(10, 5))
dendrogram(Z_amostra, truncate_mode="level", p=5, ax=ax)
st.pyplot(fig)

# Número de grupos
num_grupos = st.sidebar.radio("Número de grupos", [3, 4])
Z = linkage(X_scaled, method="ward")
clusters = fcluster(Z, num_grupos, criterion="maxclust")
df_filtrado["Grupo"] = clusters

# Taxa de compra por grupo
st.subheader("Taxa de compra por grupo")
taxa_compra = df_filtrado.groupby("Grupo")["Compra"].mean()

fig2, ax2 = plt.subplots()
taxa_compra.plot(kind="bar", color="green", ax=ax2)
ax2.set_ylabel("Taxa média de compra")
ax2.set_xlabel("Grupo")
st.pyplot(fig2)

# Estatísticas adicionais
st.subheader("Estatísticas por grupo")
st.dataframe(df_filtrado.groupby("Grupo")[["Taxa_Rejeição", "Compra"]].mean())