import streamlit as st 
import pandas as pd 
import plotly.express as px 
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt 
import time

NUM_COLS = 3

st.set_page_config(
    page_title="Projeto Final - Mineração de Dados",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

if "df" not in st.session_state:
    st.session_state.df = None


menu =  st.sidebar.radio(
     "Menu",
     ["Upload", "Visualização", "Tratamento", "Treino e Teste"]
)
    
st.write("# Mineração de Dados - Projeto Final")


check = False

#-------------MENU 1 (UPLOAD) ------------------

if menu == "Upload":

    st.header("Upload do Arquivo")

    arq = st.file_uploader(
        "Faça upload do seu arquivo (CSV ou Excel)",
        type=["csv", "xlsx"]
    )

    # Novo Upload de Arquivo 
    if arq is not None:
        st.session_state.uploaded_file = arq

        try:
            if arq.name.endswith(".csv"):
                df = pd.read_csv(arq)
            else:
                df = pd.read_excel(arq)

            st.session_state.df = df

            st.success(f"Arquivo carregado: {arq.name}")

        except Exception as e:
            st.error(f"Erro ao processar o arquivo: {e}")
            st.stop()

    if st.session_state.df is None:
        st.info("Por favor, carregue um arquivo para começar.")
        st.stop()

    df = st.session_state.df

    df_info = pd.DataFrame({
        "Tipo de Dado": df.dtypes,
        "Contagem Nulos": df.isnull().sum(),
        "Contagem Não-Nulos": df.count(),
        "% Nulos": (df.isnull().sum() / len(df)) * 100
    })

    st.write("### Informações Básicas do Dataset")
    st.write(f"**Dimensões:** {df.shape[0]} linhas × {df.shape[1]} colunas")

    st.dataframe(df_info)
    st.write("### Visualização dos Dados")
    st.dataframe(df.head(7))




elif menu == "Visualização":
    st.write("## Visualização dos Dados")
    colunas = st.columns(3)
    if "df" in st.session_state:
        df = st.session_state.df

        education_map = {1: 'Below College', 2: 'College', 3: 'Bachelor', 4: 'Master', 5: 'Doctor'}
        satisfaction_map = {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'}
        rating_map = {1: 'Low', 2: 'Good', 3: 'Excellent', 4: 'Outstanding'}

        # Criando colunas legíveis para os gráficos (sem substituir os dados originais se não quiser)
        df['Education_Label'] = df['Education'].map(education_map)
        df['JobSat_Label'] = df['JobSatisfaction'].map(satisfaction_map)
            # Prepara os dados (pode usar o crosstab da opção 2)

        n_df = pd.crosstab(df['JobRole'], df['Attrition']).reset_index()

        # Derrete o dataframe para formato longo (necessário para Plotly/Altair)
        n_df_long = n_df.melt(id_vars='JobRole', var_name='Attrition', value_name='Quantidade')
        with colunas[0]:
            # Cria o gráfico
            fig = px.bar(
                n_df_long, 
                x='Quantidade', 
                y='JobRole', 
                color='Attrition', 
                orientation='h', # Barras horizontais
                title='JobRole e Attrition',
                barmode='group' # ou 'stack' se preferir empilhado
            )   
        
            st.plotly_chart(fig)

            st.subheader("3. Média Salarial por Nível Educacional")

            # Agrupando os dados
            avg_income = df.groupby(['Education_Label', 'Attrition'])['MonthlyIncome'].mean().reset_index()

            # Ordenar a categoria Education para não ficar alfabético (opcional, mas recomendado)
            ordem_educacao = ['Below College', 'College', 'Bachelor', 'Master', 'Doctor']

            fig_bar = px.bar(
                avg_income, 
                x='Education_Label', 
                y='MonthlyIncome', 
                color='Attrition', 
                barmode='group',
                category_orders={'Education_Label': ordem_educacao}, # Força a ordem lógica
                title="Renda Mensal Média por Educação e Attrition"
            )
            st.plotly_chart(fig_bar)

        with colunas[1]:
            st.subheader("1. Distribuição de Attrition")

            # Contagem simples
            attrition_counts = df['Attrition'].value_counts().reset_index()
            attrition_counts.columns = ['Attrition', 'Count']

            fig_pie = px.pie(
                attrition_counts, 
                values='Count', 
                names='Attrition', 
                title='Proporção de Funcionários que Saíram vs Ficaram',
                color='Attrition',
                color_discrete_map={'Yes': '#FF5A5F', 'No': '#00A699'} # Cores customizadas
            )
            st.plotly_chart(fig_pie)

            st.subheader("4. Distribuição Salarial por Cargo (Boxplot)")

            fig_box = px.box(
                df, 
                x="JobRole", 
                y="MonthlyIncome", 
                color="Attrition",
                title="Salário por Cargo e Status de Saída"
            )
            st.plotly_chart(fig_box)

        with colunas[2]:
            st.subheader("2. Histogramas de Variáveis Numéricas")

            # Lista de colunas numéricas interessantes
            num_cols = ["Age", "MonthlyIncome", "DistanceFromHome", "TotalWorkingYears", "YearsAtCompany"]

            selected_col = st.selectbox("Selecione a coluna para o histograma:", num_cols)

            fig_hist = px.histogram(
                df, 
                x=selected_col, 
                color="Attrition", # Separa as cores por quem saiu ou ficou
                nbins=30, 
                title=f"Distribuição de {selected_col}",
                barmode='overlay', # ou 'group'
                opacity=0.7
            )
            st.plotly_chart(fig_hist)

            st.subheader("5. Matriz de Correlação")

            # 1. Converter Attrition para numérico temporariamente para a correlação
            df_corr = df.copy()
            df_corr['Attrition_Numeric'] = df_corr['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)

            # 2. Selecionar apenas colunas numéricas para a correlação
            # Vamos pegar apenas algumas principais para o gráfico não ficar gigante
            cols_to_corr = [
                'Attrition_Numeric', 'Age', 'DailyRate', 'DistanceFromHome', 
                'Education', 'EnvironmentSatisfaction', 'JobSatisfaction', 
                'MonthlyIncome', 'NumCompaniesWorked', 'WorkLifeBalance'
            ]

            corr_matrix = df_corr[cols_to_corr].corr()

            # Usando Plotly para Heatmap interativo
            fig_corr = px.imshow(
                corr_matrix, 
                text_auto=".2f", # Mostra os valores com 2 casas decimais
                aspect="auto",
                color_continuous_scale="RdBu_r", # Vermelho para correlação positiva, Azul para negativa
                title="Correlação entre Variáveis Selecionadas"
            )
            st.plotly_chart(fig_corr)

    else:
         st.warning("Por Favor, faça o upload do Arquivo")
          
    

elif menu == "Tratamento":
    if "df" in st.session_state:
        df = st. session_state.df 

        if st.button("Inciar Tratamento"):
            
            # ------------- LIMPEZA DE NULOS
            with st.spinner("Analisando Dados Faltantes..."):
                time.sleep(2)
            # 1. Checa se existe algum nulo no DataFrame inteiro
            tem_nulos = df.isnull().values.any()
        
            if tem_nulos:
                st.warning("⚠️ Foram encontrados valores nulos no seu dataset!")
                
                # 2. Calcula a soma de nulos por coluna
                nulos_por_coluna = df.isnull().sum()
                
                # 3. Filtra para mostrar apenas as colunas que têm problemas (> 0)
                colunas_com_problemas = nulos_por_coluna[nulos_por_coluna > 0]
                
                # Exibe a contagem
                st.write("### Contagem de Nulos por Coluna:")
                st.dataframe(colunas_com_problemas, width=400)
                
                # 4. Mostra as linhas específicas que têm nulos
                st.write("### Visualizar linhas com dados faltantes:")
                linhas_com_nulos = df[df.isnull().any(axis=1)]
                st.dataframe(linhas_com_nulos.head())
                st.caption(f"Total de linhas afetadas: {len(linhas_com_nulos)}")

                with st.spinner("Removendo Valores Nulos..."):
                    time.sleep(2)
                df = df.dropna()
                st.write("Valores nulos foram removidos do Dataset")
            else:
                st.success("✅ O dataset está limpo! Nenhum valor nulo encontrado.")
                if st.button("Remover Arquivo"):
                    st.session_state.df = None
                    st.session_state.uploaded_filae = None

            #---------ONEHOT ENCONDING 
    else:
        st.info("Faça o upload do arquivo")

elif menu == "Treino e Teste":
    st.write("Settings page")
    #st.select_slider()

check = False

colunas = st.columns(NUM_COLS)
# 2. Verifica se um arquivo foi carregado antes de tentar ler
