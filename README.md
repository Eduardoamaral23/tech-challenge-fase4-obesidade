# TECH CHALLENGE - PREVISÃO DE OBESIDADE

**Documento de Entrega - Fase 4**

**Data:** Maio/2026
**Projeto:** Modelo Preditivo de Obesidade para Hospital
**Status:** Concluído

---

## 1. RESUMO EXECUTIVO
Foi desenvolvido um modelo de Machine Learning para prever o nível de obesidade de pacientes com base em dados demográficos, hábitos de vida e histórico familiar.

**Resultado principal:**
- Acurácia de teste: 97,87%
- Meta mínima exigida: > 75%

**Requisitos atendidos:**
- Pipeline completo de ML (feature engineering + treino).
- Modelo com desempenho superior ao exigido.
- Aplicação preditiva em Streamlit com interface clínica.
- Painel analítico com insights para suporte à decisão médica.
- Projeto publicado em repositório GitHub e deploy no Streamlit Cloud.

---

## 2. MÉTRICAS DE DESEMPENHO
O modelo apresentou uma performance robusta, com alta precisão em todas as 7 classes de obesidade.

| Classe | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| Insufficient_Weight | 0.98 | 0.98 | 0.98 |
| Normal_Weight | 0.91 | 0.97 | 0.94 |
| Obesity_Type_I | 1.00 | 1.00 | 1.00 |
| Obesity_Type_II | 1.00 | 1.00 | 1.00 |
| Obesity_Type_III | 1.00 | 1.00 | 1.00 |
| Overweight_Level_I | 0.96 | 0.91 | 0.94 |
| Overweight_Level_II | 1.00 | 0.98 | 0.99 |

- **Acurácia Geral:** 98% (97,87%)
- **Base de dados:** 2.111 registros

---

## 3. ARQUITETURA DA SOLUÇÃO
- **Modelo:** RandomForestClassifier (Otimizado via GridSearchCV).
- **Preprocessamento:** StandardScaler para variáveis numéricas e OneHotEncoder para categóricas, integrados em um Pipeline com ColumnTransformer.
- **Feature Engineering:** Cálculo dinâmico de BMI (IMC), arredondamento de escalas (FCVC, NCP, CH2O, FAF, TUE) e codificação de variáveis.
- **Artefatos:** `modelo.pkl` (Pipeline completo de transformação e modelo serializado via Pickle).

---

## 4. ESTRUTURA DE ARQUIVOS
- `app.py`: Aplicação principal (Streamlit).
- `modelo.pkl`: Modelo treinado.
- `Obesity.csv`: Base de dados original.
- `requirements.txt`: Dependências do projeto.
- `README.md`: Documentação.

---

## 5. INSTRUÇÕES DE EXECUÇÃO LOCAL
1. **Instalar dependências:** `pip install -r requirements.txt`
2. **Rodar aplicação:** `streamlit run app.py`
3. **Acessar:** `http://localhost:8501`

---

## 6. FUNCIONALIDADES
- **Aba 1 (Predição Clínica):** Formulário em português com conversão automática de variáveis e exibição de diagnóstico médico.
- **Aba 2 (Painel Analítico):** Visualização epidemiológica com filtros interativos e gráficos dinâmicos para a equipe médica.

---

## 7. FEATURES E TARGET
- **Entradas:** Gender, Age, Height, Weight, family_history, FAVC, FCVC, NCP, CAEC, SMOKE, CH2O, SCC, FAF, TUE, CALC, MTRANS.
- **Engenharia:** O BMI é calculado automaticamente a partir de Height e Weight.
- **Saída:** Nível de obesidade (7 classes).

---

## 8. LINKS DA ENTREGA
- **Aplicação (Streamlit Cloud):** [Inserir aqui o seu link final]
- **Repositório GitHub:** [Inserir aqui o seu link final]

---

## 9. CHECKLIST FINAL
[X] Pipeline de ML completo  
[X] Acurácia > 75% (resultado: 97,87%)  
[X] Deploy em Streamlit  
[X] Painel analítico com insights  
[X] Repositório GitHub com código e artefatos  
[X] Documentação completa
