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
    modelo = joblib.load('modelo.pkl')
    tradutor = joblib.load('tradutor_classes.joblib')
    return modelo, tradutor

@st.cache_data
def carregar_dados():
    df = pd.read_csv('Obesity.csv')
    
    # Criando o IMC se não existir
    if 'BMI' not in df.columns:
        df['BMI'] = df['Weight'] / (df['Height'] ** 2)
        
    # Traduzindo colunas para o Dashboard
    df['Obesity'] = df['Obesity'].replace(dicionario_traducao)
    df['Gender'] = df['Gender'].replace({'Female': 'Feminino', 'Male': 'Masculino'})
    df['FAVC'] = df['FAVC'].replace({'yes': 'Sim', 'no': 'Não'})
    df['SMOKE'] = df['SMOKE'].replace({'yes': 'Sim', 'no': 'Não'})
    df['family_history'] = df['family_history'].replace({'yes': 'Sim', 'no': 'Não'})
    
    # Criando coluna de Risco Alto
    classes_risco = ['Obesidade Tipo I', 'Obesidade Tipo II', 'Obesidade Tipo III']
    df['Risco Alto'] = df['Obesity'].apply(lambda x: 'Sim' if x in classes_risco else 'Não')
    
    # Criando Faixas Etárias
    bins = [0, 20, 30, 40, 50, 100]
    labels = ['<20', '20-29', '30-39', '40-49', '50+']
    df['Faixa Etária'] = pd.cut(df['Age'], bins=bins, labels=labels)
    
    return df

modelo, tradutor = carregar_modelos()
df_dados = carregar_dados()

# ==========================================
# MENU DE NAVEGAÇÃO LATERAL
# ==========================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3063/3063206.png", width=100)
st.sidebar.title("Menu")
pagina = st.sidebar.radio("Selecione a página:", ["🔮 Predição Clínica", "📊 Painel Analítico"])

# ==========================================
# PÁGINA 1: PREDIÇÃO CLÍNICA
# ==========================================
if pagina == "🔮 Predição Clínica":
    st.title("🏥 Sistema de Suporte à Decisão Clínica")
    st.markdown("---")
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
            imc = weight / (height ** 2)
            dados_entrada = pd.DataFrame([{
                'Gender': map_gender[gender_br], 'Age': age, 'Height': height, 'Weight': weight,
                'family_history': map_yes_no[family_history_br], 'FAVC': map_yes_no[favc_br],
                'FCVC': fcvc, 'NCP': ncp, 'CAEC': map_caec[caec_br], 'SMOKE': map_yes_no[smoke_br],
                'CH2O': ch2o, 'SCC': map_yes_no[scc_br], 'FAF': faf, 'TUE': tue,
                'CALC': map_calc[calc_br], 'MTRANS': map_mtrans[mtrans_br], 'BMI': imc
            }])
            
            predicao = modelo.predict(dados_entrada)[0]
            res_ingles = tradutor.inverse_transform([predicao])[0]
            
            # Formatação do IMC com 1 casa decimal e vírgula
            imc_formatado = f"{imc:.1f}".replace(".", ",")
            
            st.success(f"### Diagnóstico: {dicionario_traducao.get(res_ingles, res_ingles)}")
            st.info(f"**IMC Calculado:** {imc_formatado} kg/m²")

# ==========================================
# PÁGINA 2: PAINEL ANALÍTICO
# ==========================================
elif pagina == "📊 Painel Analítico":
    st.title("📊 Dashboard de Análise")
    st.markdown("---")
    
    # Filtros Exclusivos desta página
    st.sidebar.markdown("### Filtros de Dados")
    generos = ["Todos"] + sorted(df_dados['Gender'].unique().tolist())
    filtro_genero = st.sidebar.selectbox("Gênero", generos)
    
    classes = ["Todos"] + sorted(df_dados['Obesity'].unique().tolist())
    filtro_classe = st.sidebar.selectbox("Nível de Obesidade", classes)
    
    filtro_idade = st.sidebar.slider("Faixa Etária", int(df_dados['Age'].min()), int(df_dados['Age'].max()), (14, 60))

    # Aplicando os Filtros
    df_filtrado = df_dados.copy()
    if filtro_genero != "Todos": df_filtrado = df_filtrado[df_filtrado['Gender'] == filtro_genero]
    if filtro_classe != "Todos": df_filtrado = df_filtrado[df_filtrado['Obesity'] == filtro_classe]
    df_filtrado = df_filtrado[(df_filtrado['Age'] >= filtro_idade[0]) & (df_filtrado['Age'] <= filtro_idade[1])]

    # 1. KPIs MAcro
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    
    total_pacientes = len(df_filtrado)
    imc_medio = df_filtrado['BMI'].mean()
    
    classes_excesso = ['Sobrepeso Nível I', 'Sobrepeso Nível II', 'Obesidade Tipo I', 'Obesidade Tipo II', 'Obesidade Tipo III']
    excesso_peso_pct = (df_filtrado['Obesity'].isin(classes_excesso).sum() / total_pacientes) * 100 if total_pacientes > 0 else 0
    
    col_kpi1.metric("👥 Pacientes Analisados", f"{total_pacientes:,}".replace(',', '.'))
    col_kpi2.metric("⚖️ IMC Médio", f"{imc_medio:.1f}".replace('.', ','))
    col_kpi3.metric("🚨 Excesso de Peso (%)", f"{excesso_peso_pct:.1f}%".replace('.', ','))
    
    st.markdown("---")

    # 2. Criando as 3 Abas
    aba_visao, aba_risco, aba_habitos = st.tabs(["Visão Geral", "Risco", "Hábitos"])

    # --- ABA: VISÃO GERAL ---
    with aba_visao:
        col1, col2 = st.columns(2)
        with col1:
            # Gráfico de Prevalência
            df_prev = df_filtrado['Obesity'].value_counts(normalize=True).reset_index()
            df_prev.columns = ['Classe', 'Proporção']
            df_prev['Prevalência (%)'] = df_prev['Proporção'] * 100
            fig_prev = px.bar(df_prev, x='Prevalência (%)', y='Classe', orientation='h', title="Prevalência por Classe (%)")
            st.plotly_chart(fig_prev, use_container_width=True)
            
        with col2:
            # Boxplot IMC por Classe
            fig_box = px.box(df_filtrado, x='Obesity', y='BMI', color='Gender', title="Distribuição de IMC por Classe (Boxplot)")
            st.plotly_chart(fig_box, use_container_width=True)
            
        # Scatter Plot Idade x IMC
        fig_scatter = px.scatter(df_filtrado, x='Age', y='BMI', color='Obesity', title="Idade vs IMC (Dispersão)")
        st.plotly_chart(fig_scatter, use_container_width=True)

    # --- ABA: RISCO ---
    with aba_risco:
        col3, col4 = st.columns(2)
        
        with col3:
            # Risco Alto por Gênero (%)
            df_risco_gen = df_filtrado.groupby('Gender')['Risco Alto'].apply(lambda x: (x == 'Sim').mean() * 100).reset_index()
            df_risco_gen.columns = ['Gênero', '% Risco Alto']
            fig_risco_gen = px.bar(df_risco_gen, x='Gênero', y='% Risco Alto', color='Gênero', title="Risco clínico alto por Gênero (%)")
            st.plotly_chart(fig_risco_gen, use_container_width=True)
            
        with col4:
            # Risco Alto por Faixa Etária (%)
            df_risco_idade = df_filtrado.groupby('Faixa Etária')['Risco Alto'].apply(lambda x: (x == 'Sim').mean() * 100).reset_index()
            df_risco_idade.columns = ['Faixa Etária', '% Risco Alto']
            fig_risco_idade = px.bar(df_risco_idade, x='Faixa Etária', y='% Risco Alto', title="Risco clínico alto por Faixa Etária (%)")
            st.plotly_chart(fig_risco_idade, use_container_width=True)
            
        # Tabela Resumo de Risco
        st.subheader("Tabela-resumo de Risco e Médias Clínicas")
        resumo_risco = df_filtrado.groupby('Obesity').agg(
            Pacientes=('Age', 'count'),
            Idade_Média=('Age', 'mean'),
            IMC_Médio=('BMI', 'mean')
        ).reset_index().round(1)
        st.table(resumo_risco)

    # --- ABA: HÁBITOS ---
    with aba_habitos:
        st.subheader("Impacto dos Hábitos no Risco Clínico Alto")
        
        # Gráfico Risco por Hábito (Comparativo FAVC)
        df_habito = df_filtrado.groupby(['FAVC', 'Risco Alto']).size().reset_index(name='Contagem')
        fig_habitos = px.bar(df_habito, x='FAVC', y='Contagem', color='Risco Alto', barmode='group', 
                             title="Risco Alto vs Consumo de Alimentos Calóricos (FAVC)")
        st.plotly_chart(fig_habitos, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Amostra dos Dados Filtrados (100%)")
        st.markdown("Utilize a barra de rolagem para navegar por todos os registros ou ordene clicando no cabeçalho das colunas.")
        # Tabela com barra de rolagem
        st.dataframe(df_filtrado, use_container_width=True, height=400)
