import streamlit as st 
import pandas as pd 



st.title("Mineração de Dados - Projeto Final")

with st.sidebar:
    st.title("Menu")

    st.select_slider()


arq = st.file_uploader(
    "Faça upload do seu arquivo (CSV ou Excel)", 
    type=['csv', 'xlsx']
)

# 2. Verifica se um arquivo foi carregado antes de tentar ler
if arq is not None:
    try:
        # 3. Lógica para determinar o formato e ler o arquivo
        if arq.name.endswith('.csv'):
            # Se for CSV
            df = pd.read_csv(arq)
            df_info = pd.DataFrame({
                'Tipo de Dado': df.dtypes,
                'Contagem Nulos': df.isnull().sum(),
                'Contagem Não-Nulos': df.count(),
                '% Nulos': (df.isnull().sum() / len(df)) * 100
            })
        else:
            # Se for Excel (xlsx)
            df = pd.read_excel(arq)
            df_info = pd.DataFrame({
                'Tipo de Dado': df.dtypes,
                'Contagem Nulos': df.isnull().sum(),
                'Contagem Não-Nulos': df.count(),
                '% Nulos': (df.isnull().sum() / len(df)) * 100
            })

        # Feedback visual de sucesso
        st.success("Arquivo carregado com sucesso!")
        
        # 4. Exibe as primeiras linhas do DataFrame e informações úteis
        # informações basicas do DF
        st.write("### Visualização dos Dados")

        st.write()
        st.dataframe(df.head())

        st.write("### Informações Basicas do Dataset: ")
        st.write(f"**Dimensões do Dataset:** {df.shape[0]} linhas e {df.shape[1]} colunas.")
        st.dataframe(df_info)

    except Exception as e:
        st.error(f"Ocorreu um erro ao processar o arquivo: {e}")
else:
    st.info("Por favor, carregue um arquivo para começar.")