import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# --- DICIONÁRIOS DE TRADUÇÃO (Interface p/ Modelo) ---
map_gender = {'Feminino': 'Female', 'Masculino': 'Male'}
map_yes_no = {'Sim': 'yes', 'Não': 'no'}
map_caec = {'Não': 'no', 'Às vezes': 'Sometimes', 'Frequentemente': 'Frequently', 'Sempre': 'Always'}
map_calc = {'Não': 'no', 'Às vezes': 'Sometimes', 'Frequentemente': 'Frequently', 'Sempre': 'Always'}
map_mtrans = {
    'Transporte Público': 'Public_Transportation',
    'Caminhando': 'Walking',
    'Carro': 'Automobile',
    'Moto': 'Motorbike',
    'Bicicleta': 'Bike'
}
dicionario_traducao = {
    'Insufficient_Weight': 'Abaixo do Peso', 'Normal_Weight': 'Peso Normal',
    'Overweight_Level_I': 'Sobrepeso Nível I', 'Overweight_Level_II': 'Sobrepeso Nível II',
    'Obesity_Type_I': 'Obesidade Tipo I', 'Obesity_Type_II': 'Obesidade Tipo II',
    'Obesity_Type_III': 'Obesidade Tipo III'
}

# Configuração da página
st.set_page_config(page_title="Predição de Obesidade", page_icon="🏥", layout="wide")

@st.cache_resource
def carregar_modelos():
    modelo = joblib.load('modelo_obesidade.joblib')
    tradutor = joblib.load('tradutor_classes.joblib')
    return modelo, tradutor

@st.cache_data
def carregar_dados():
    df = pd.read_csv('Obesity.csv')
    # Traduzindo colunas para o Dashboard ficar em PT
    df['Obesity'] = df['Obesity'].replace(dicionario_traducao)
    df['Gender'] = df['Gender'].replace({'Female': 'Feminino', 'Male': 'Masculino'})
    return df

modelo, tradutor = carregar_modelos()
df_dados = carregar_dados()

st.title("🏥 Sistema de Suporte à Decisão Clínica: Obesidade")
st.markdown("---")

aba_predicao, aba_dashboard = st.tabs(["🔮 Predição Clínica", "📊 Painel Analítico"])

with aba_predicao:
    st.header("Formulário do Paciente")
    with st.form("form_paciente"):
        col1, col2, col3 = st.columns(3)
        with col1:
            gender_br = st.selectbox("Sexo", list(map_gender.keys()))
            age = st.number_input("Idade", min_value=14, max_value=100, value=25)
            height = st.number_input("Altura (m)", min_value=1.0, max_value=2.5, value=1.70, step=0.01)
            weight = st.number_input("Peso (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.1)
            mtrans_br = st.selectbox("Transporte Habitual", list(map_mtrans.keys()))
        with col2:
            family_history_br = st.selectbox("Histórico familiar?", list(map_yes_no.keys()))
            favc_br = st.selectbox("Consome alimentos calóricos?", list(map_yes_no.keys()))
            fcvc = st.slider("Consumo de vegetais (1-3)", 1, 3, 2)
            ncp = st.slider("Refeições principais", 1, 4, 3)
            caec_br = st.selectbox("Lanches entre refeições?", list(map_caec.keys()))
        with col3:
            smoke_br = st.selectbox("Fumante?", list(map_yes_no.keys()))
            ch2o = st.slider("Consumo de água (1-3)", 1, 3, 2)
            scc_br = st.selectbox("Monitora calorias?", list(map_yes_no.keys()))
            faf = st.slider("Atividade física (0-3)", 0, 3, 1)
            tue = st.slider("Tempo em telas (0-2)", 0, 2, 1)
            calc_br = st.selectbox("Consumo de álcool?", list(map_calc.keys()))
            
        if st.form_submit_button("Gerar Diagnóstico"):
            dados_entrada = pd.DataFrame([{
                'Gender': map_gender[gender_br], 'Age': age, 'Height': height, 'Weight': weight,
                'family_history': map_yes_no[family_history_br], 'FAVC': map_yes_no[favc_br],
                'FCVC': fcvc, 'NCP': ncp, 'CAEC': map_caec[caec_br], 'SMOKE': map_yes_no[smoke_br],
                'CH2O': ch2o, 'SCC': map_yes_no[scc_br], 'FAF': faf, 'TUE': tue,
                'CALC': map_calc[calc_br], 'MTRANS': map_mtrans[mtrans_br],
                'BMI': weight / (height ** 2)
            }])
            predicao = modelo.predict(dados_entrada)[0]
            res_ingles = tradutor.inverse_transform([predicao])[0]
            st.success(f"### Diagnóstico: {dicionario_traducao.get(res_ingles, res_ingles)}")
            st.info(f"**IMC:** {weight / (height ** 2):.2f} kg/m²")

with aba_dashboard:
    st.header("📊 Painel Analítico")
    
    # Filtros na Barra Lateral
    st.sidebar.header("Filtros de Análise")
    generos = ["Todos"] + sorted(df_dados['Gender'].unique().tolist())
    filtro_genero = st.sidebar.selectbox("Gênero", generos)
    
    classes = ["Todos"] + sorted(df_dados['Obesity'].unique().tolist())
    filtro_classe = st.sidebar.selectbox("Nível de Obesidade", classes)
    
    filtro_idade = st.sidebar.slider("Faixa Etária", int(df_dados['Age'].min()), int(df_dados['Age'].max()), (14, 60))

    # Aplicando Filtros
    df_filtrado = df_dados.copy()
    if filtro_genero != "Todos": df_filtrado = df_filtrado[df_filtrado['Gender'] == filtro_genero]
    if filtro_classe != "Todos": df_filtrado = df_filtrado[df_filtrado['Obesity'] == filtro_classe]
    df_filtrado = df_filtrado[(df_filtrado['Age'] >= filtro_idade[0]) & (df_filtrado['Age'] <= filtro_idade[1])]

    # Gráficos
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribuição por Nível")
        st.plotly_chart(px.pie(df_filtrado, names='Obesity', hole=0.3), use_container_width=True)
    with col2:
        st.subheader("Risco: Consumo de Calóricos (FAVC)")
        st.plotly_chart(px.histogram(df_filtrado, x='Obesity', color='FAVC', barmode='group'), use_container_width=True)

    st.subheader("Amostra dos Dados Filtrados")
    st.dataframe(df_filtrado.head(10), use_container_width=True)
