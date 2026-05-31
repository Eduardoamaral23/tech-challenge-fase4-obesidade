import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Configuração da página
st.set_page_config(page_title="Predição de Obesidade", page_icon="🏥", layout="wide")

# Carregando os modelos e os dados
@st.cache_resource
def carregar_modelos():
    modelo = joblib.load('modelo_obesidade.joblib')
    tradutor = joblib.load('tradutor_classes.joblib')
    return modelo, tradutor

@st.cache_data
def carregar_dados():
    return pd.read_csv('Obesity.csv')

modelo, tradutor = carregar_modelos()
df_dados = carregar_dados()

st.title("🏥 Sistema de Suporte à Decisão Clínica: Obesidade")
st.markdown("---")

# Criando as abas
aba_predicao, aba_dashboard = st.tabs(["🔮 Predição Clínica", "📊 Painel Analítico"])

# ==========================================
# ABA 1: PREDIÇÃO CLÍNICA
# ==========================================
with aba_predicao:
    st.header("Formulário do Paciente")
    
    with st.form("form_paciente"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            gender = st.selectbox("Sexo", ["Female", "Male"])
            age = st.number_input("Idade", min_value=14, max_value=100, value=25)
            height = st.number_input("Altura (m)", min_value=1.0, max_value=2.5, value=1.70, step=0.01)
            weight = st.number_input("Peso (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.1)
            mtrans = st.selectbox("Transporte Habitual", ["Public_Transportation", "Automobile", "Walking", "Motorbike", "Bike"])
            
        with col2:
            family_history = st.selectbox("Histórico familiar de excesso de peso?", ["yes", "no"])
            favc = st.selectbox("Consome alimentos muito calóricos?", ["yes", "no"])
            fcvc = st.slider("Consumo de vegetais (1-Raro a 3-Sempre)", 1, 3, 2)
            ncp = st.slider("Refeições principais por dia", 1, 4, 3)
            caec = st.selectbox("Come lanches entre as refeições?", ["no", "Sometimes", "Frequently", "Always"])
            
        with col3:
            smoke = st.selectbox("Fumante?", ["no", "yes"])
            ch2o = st.slider("Consumo de água (1-Pouco a 3-Muito)", 1, 3, 2)
            scc = st.selectbox("Monitora as calorias?", ["no", "yes"])
            faf = st.slider("Frequência de atividade física semanal (0 a 3)", 0, 3, 1)
            tue = st.slider("Tempo diário em telas (0 a 2)", 0, 2, 1)
            calc = st.selectbox("Consumo de álcool?", ["no", "Sometimes", "Frequently", "Always"])
            
        botao_prever = st.form_submit_button("Gerar Diagnóstico")
        
        if botao_prever:
            # Calculando o IMC e montando a tabela de entrada EXATAMENTE como o modelo aprendeu
            imc = weight / (height ** 2)
            
            dados_entrada = pd.DataFrame([{
                'Gender': gender, 'Age': age, 'Height': height, 'Weight': weight,
                'family_history': family_history, 'FAVC': favc, 'FCVC': fcvc,
                'NCP': ncp, 'CAEC': caec, 'SMOKE': smoke, 'CH2O': ch2o,
                'SCC': scc, 'FAF': faf, 'TUE': tue, 'CALC': calc, 'MTRANS': mtrans,
                'BMI': imc
            }])
            
            # Dicionário de tradução do inglês para o português
            dicionario_traducao = {
                'Insufficient_Weight': 'Abaixo do Peso',
                'Normal_Weight': 'Peso Normal',
                'Overweight_Level_I': 'Sobrepeso Nível I',
                'Overweight_Level_II': 'Sobrepeso Nível II',
                'Obesity_Type_I': 'Obesidade Tipo I',
                'Obesity_Type_II': 'Obesidade Tipo II',
                'Obesity_Type_III': 'Obesidade Tipo III'
            }
            
            # Fazendo a predição
            predicao = modelo.predict(dados_entrada)[0]
            resultado_ingles = tradutor.inverse_transform([predicao])[0]
            
            # Traduzindo o resultado
            resultado_portugues = dicionario_traducao.get(resultado_ingles, resultado_ingles)
            
            st.success(f"### Diagnóstico Preditivo: {resultado_portugues}")
            st.info(f"**IMC Calculado:** {imc:.2f} kg/m²")

# ==========================================
# ABA 2: PAINEL ANALÍTICO (DASHBOARD)
# ==========================================
with aba_dashboard:
    st.header("Análise de Dados Históricos (Insights)")
    st.markdown("Visão geral da base de dados para auxiliar na tomada de decisão da equipe médica.")
    
    col_graf1, col_graf2 = st.columns(2)
    
    with col_graf1:
        st.subheader("Distribuição dos Níveis de Obesidade")
        fig_pizza = px.pie(df_dados, names='Obesity', hole=0.3)
        st.plotly_chart(fig_pizza, use_container_width=True)
        
    with col_graf2:
        st.subheader("Idade vs Nível de Obesidade")
        fig_box = px.box(df_dados, x='Obesity', y='Age', color='Gender')
        st.plotly_chart(fig_box, use_container_width=True)
        
    st.subheader("Relação entre Histórico Familiar e Obesidade")
    fig_hist = px.histogram(df_dados, x='Obesity', color='family_history', barmode='group')
    st.plotly_chart(fig_hist, use_container_width=True)
