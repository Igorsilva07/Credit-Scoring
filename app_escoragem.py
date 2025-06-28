# app_escoragem.py

import streamlit as st
import pandas as pd
import numpy as np
import os # Importado para verificar caminhos de arquivo

# Importar PyCaret
# Certifique-se de ter 'pip install pycaret' no seu ambiente
from pycaret.classification import load_model, predict_model # Para problemas de classificação

# --- INÍCIO DO SEU STREAMLIT APP ---
# st.set_page_config() DEVE SER A PRIMEIRA CHAMADA DO STREAMLIT NO SEU SCRIPT
st.set_page_config(layout="wide", page_title="Escoragem de Risco de Crédito")

# --- ATENÇÃO: PARÂMETROS DO TREINAMENTO ---
# Estes valores são importantes se você fez algum pré-processamento manual
# ANTES de passar os dados para o setup() do PyCaret.
# Se o PyCaret lidou com TUDO (imputação, log-transformação, capping, etc.)
# então a função 'apply_initial_transforms' e estas constantes podem ser simplificadas
# ou até removidas (se o PyCaret processa dados totalmente brutos).
# Mantenha os valores que você obteve do seu treinamento:
MEDIANA_TEMPO_EMPREGO_TREINO = 6.0466
LOWER_BOUND_RENDA_LOG_TREINO = 6.1504
UPPER_BOUND_RENDA_LOG_TREINO = 12.2807
LOWER_BOUND_TEMPO_EMPREGO_TREINO = -4.5158
UPPER_BOUND_TEMPO_EMPREGO_TREINO = 17.2212


# --- DEFINIÇÕES DE COLUNAS ---
# Estas são as colunas EXATAS que o seu CSV de entrada deve ter,
# e as que foram usadas no setup() do PyCaret para treinamento.
COLUNAS_ORIGINAIS_PARA_PYCARET = [
    'qtd_filhos', 'idade', 'tempo_emprego', 'qt_pessoas_residencia', 'renda',
    'sexo', 'posse_de_veiculo', 'posse_de_imovel', 'tipo_renda',
    'educacao', 'estado_civil', 'tipo_residencia',
    'data_ref',
    'index'
]

# --- 1. Funções de Pré-processamento MANUAL (se houver, antes de passar ao PyCaret) ---
# Esta função é crucial SE você fez transformações como agrupamento de categorias
# ou imputação/capping MANUAIS ANTES de chamar pycaret.setup().
# Se o pycaret.setup() lidou com TUDO, você pode simplificar esta função para
# apenas retornar o df_input.copy() com as colunas certas.
def apply_initial_transforms(df_input):
    """
    Aplica as transformações iniciais que foram feitas MANUAMENTE antes do setup() do PyCaret.
    Se o PyCaret lidou com tudo (imputação, agrupamento, etc.), revise ou simplifique esta função.
    """
    df = df_input.copy()

    # --- Validação de colunas de entrada ---
    missing_initial_cols = [col for col in COLUNAS_ORIGINAIS_PARA_PYCARET if col not in df.columns]
    if missing_initial_cols:
        st.error(f"Erro: As seguintes colunas essenciais não foram encontradas no seu CSV de entrada: {missing_initial_cols}")
        st.info("Verifique se o CSV que você está subindo tem todas as colunas que foram usadas no treinamento, incluindo 'data_ref' e 'index'.")
        return None

    # --- Conversão de 'data_ref' para datetime (CRUCIAL PARA ESTE ERRO) ---
    # Se 'data_ref' não for datetime, o PyCaret vai reclamar.
    # Certifique-se de que o formato da data no CSV seja compatível com pd.to_datetime.
    try:
        df['data_ref'] = pd.to_datetime(df['data_ref'])
    except Exception as e:
        st.error(f"Erro ao converter a coluna 'data_ref' para formato de data: {e}")
        st.info("Verifique se as datas na coluna 'data_ref' do seu CSV estão em um formato válido (ex: 'YYYY-MM-DD').")
        return None
    
    # --- Agrupamento de Categorias (se você fez isso antes do PyCaret setup) ---
    # Se o PyCaret foi configurado para lidar com categorias diretamente (sem agrupamento prévio),
    # COMENTE OU REMOVA ESTAS LINHAS.
    df['educacao'] = df['educacao'].replace({'Fundamental': 'Outros_Educacao', 'Pós graduação': 'Superior_Completo'})
    df['estado_civil'] = df['estado_civil'].replace({'União': 'Outros_Civil', 'Separado': 'Outros_Civil', 'Viúvo': 'Outros_Civil'})
    df['tipo_residencia'] = df['tipo_residencia'].replace({'Com os pais': 'Outros_Residencia', 'Governamental': 'Outros_Residencia', 'Aluguel': 'Outros_Residencia', 'Estúdio': 'Outros_Residencia', 'Comunitário': 'Outros_Residencia'})

    # --- Imputação de Missings em 'tempo_emprego' (se você fez isso antes do PyCaret setup) ---
    # Se o PyCaret foi configurado para imputar missings (e.g., numeric_imputation='median'),
    # COMENTE OU REMOVA ESTA LINHA.
    df['tempo_emprego'] = df['tempo_emprego'].fillna(MEDIANA_TEMPO_EMPREGO_TREINO)

    # --- Log-transformação e Capping (se você fez isso antes do PyCaret setup) ---
    # Se o PyCaret foi configurado para transformar ('feature_transformation=True')
    # ou normalizar ('normalize=True'), ou lidar com outliers ('remove_outliers=True'),
    # COMENTE OU REMOVA ESTAS LINHAS.
    # df['renda_log'] = np.log(df['renda'] + 1) # Note: se você renomeou a coluna, PyCaret espera o nome final
    # df['renda_log'] = np.clip(df['renda_log'], LOWER_BOUND_RENDA_LOG_TREINO, UPPER_BOUND_RENDA_LOG_TREINO)
    # df['tempo_emprego'] = np.clip(df['tempo_emprego'], LOWER_BOUND_TEMPO_EMPREGO_TREINO, UPPER_BOUND_TEMPO_EMPREGO_TREINO)


    # Retorna apenas as colunas que o PyCaret espera, com as transformações manuais aplicadas.
    return df[COLUNAS_ORIGINAIS_PARA_PYCARET]


# --- 2. Carregamento do Modelo PyCaret ---

@st.cache_resource # Usa o cache do Streamlit para carregar o modelo apenas uma vez
def load_pycaret_model(filename_base='final_lightgbm_model_pycaret'): # SEM .pkl AQUI
    """Carrega o modelo treinado com PyCaret."""
    
    # Pega o diretório atual do script Python
    current_dir = os.path.dirname(__file__) if '__file__' in locals() else os.getcwd()
    full_path_with_pkl = os.path.join(current_dir, f"{filename_base}.pkl") # Adiciona .pkl para verificação

    st.info(f"Tentando carregar o modelo de: {full_path_with_pkl}")
    
    if not os.path.exists(full_path_with_pkl):
        st.error(f"Erro Crítico: O arquivo '{filename_base}.pkl' NÃO existe no caminho esperado: '{full_path_with_pkl}'.")
        st.info("Por favor, verifique se o arquivo '.pkl' está na mesma pasta que 'app_escoragem.py'.")
        return None

    try:
        # load_model adiciona automaticamente o .pkl, então passamos apenas o nome base
        model = load_model(filename_base)
        st.success(f"Modelo '{filename_base}.pkl' carregado com sucesso!")
        return model
    except Exception as e:
        st.error(f"Erro ao carregar o modelo PyCaret: {e}")
        st.error(f"Detalhes do erro: {type(e).__name__}: {e}")
        st.info("Certifique-se de que o modelo foi salvo corretamente com `pycaret.classification.save_model()` (passando o nome base, sem .pkl) e que a versão do PyCaret/Python é compatível.")
        return None

# Carregar o modelo globalmente no app. Esta função será executada apenas uma vez.
MODELO_PYCARET = load_pycaret_model()


# --- 3. Função de Escoragem Principal ---

def perform_scoring_pycaret(df_input_raw):
    """
    Realiza o pré-processamento (via PyCaret) e a escoragem dos dados de entrada.
    """
    if MODELO_PYCARET is None:
        st.error("Modelo PyCaret não carregado. Não é possível escorar os dados.")
        return None

    # Aplica transformações manuais iniciais, SE HOUVEREM sido feitas ANTES do setup() do PyCaret.
    df_processed_for_pycaret = apply_initial_transforms(df_input_raw)
    if df_processed_for_pycaret is None:
        return None # Erro já foi exibido por apply_initial_transforms

    try:
        # Remover a coluna 'mau' (target) se ela estiver presente no CSV de entrada,
        # pois predict_model espera apenas as features.
        if 'mau' in df_processed_for_pycaret.columns:
            df_processed_for_pycaret = df_processed_for_pycaret.drop(columns=['mau'])

        # Usa a função predict_model do PyCaret, que aplica o pré-processamento interno do PyCaret
        # e faz a previsão. 'raw_score=True' obtém as probabilidades.
        predictions = predict_model(estimator=MODELO_PYCARET, data=df_processed_for_pycaret, raw_score=True)

        # --- AJUSTE FINAL: Mapeando as colunas de saída do PyCaret ---
        # Conforme a imagem que você enviou, as colunas são:
        # 'prediction_label', 'prediction_score_0', 'prediction_score_1'
        # Assumindo que a classe 'mau' é a classe 1, queremos 'prediction_score_1' para probabilidade
        # e 'prediction_label' para a classe prevista.
        prob_col = 'prediction_score_1' # Probabilidade da classe 1 (mau)
        pred_col = 'prediction_label'   # Classe prevista (0 ou 1)

        # Verificar se as colunas esperadas realmente existem no DataFrame de 'predictions'
        if prob_col not in predictions.columns or pred_col not in predictions.columns:
            st.error(f"Erro: As colunas de previsão esperadas ('{prob_col}' e '{pred_col}') não foram encontradas no DataFrame retornado pelo PyCaret.")
            st.info("Aqui estão as primeiras linhas do DataFrame de previsão retornado pelo PyCaret para depuração:")
            st.dataframe(predictions.head()) # Exibe o dataframe para depuração
            return None
        # --- FIM DO AJUSTE ---

        # Adicionar resultados ao DataFrame original de entrada
        df_output = df_input_raw.copy()
        df_output['probabilidade_mau'] = predictions[prob_col]
        df_output['predicao_mau'] = predictions[pred_col].astype(int)

        return df_output

    except Exception as e:
        st.error(f"Erro durante a escoragem com o modelo PyCaret: {e}")
        st.info("Verifique se as colunas do CSV de entrada correspondem às colunas esperadas pelo modelo PyCaret e se o pré-processamento manual está correto.")
        st.exception(e) # Exibe o traceback completo para depuração
        return None

# --- SEÇÃO PRINCIPAL DO STREAMLIT APP (Interface do Usuário) ---

st.title("Sistema de Escoragem de Risco de Crédito (PyCaret LightGBM)")
st.markdown("Faça o upload de um arquivo CSV com dados de clientes para prever a probabilidade de 'mau' (inadimplência) usando um modelo LightGBM treinado com PyCaret.")

# Criar um carregador de CSV no Streamlit
uploaded_file = st.file_uploader("Selecione um arquivo CSV para escoragem", type="csv")

if uploaded_file is not None:
    st.success("Arquivo CSV carregado com sucesso!")
    
    # Exibir prévia dos dados carregados
    df_input_raw = pd.read_csv(uploaded_file)
    st.subheader("Prévia dos Dados Carregados:")
    st.dataframe(df_input_raw.head())

    # Botão para executar a escoragem
    if st.button("Executar Escoragem"):
        with st.spinner('Processando e escorando os dados...'):
            df_scored_results = perform_scoring_pycaret(df_input_raw)

        if df_scored_results is not None:
            st.subheader("Resultados da Escoragem:")
            st.dataframe(df_scored_results)

            # Oferecer botão para download dos resultados
            csv_output = df_scored_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Baixar Resultados Escorados (CSV)",
                data=csv_output,
                file_name="resultados_escoragem_credit_scoring_pycaret.csv",
                mime="text/csv",
            )
        else:
            st.error("Não foi possível gerar os resultados da escoragem. Verifique as mensagens de erro acima.")
            st.info("Certifique-se de que o arquivo CSV possui as colunas esperadas e que o modelo PyCaret foi carregado corretamente.")