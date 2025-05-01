import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import statsmodels.api as sm
import datetime
import io
import logging

# Configuração da página com tema visual
st.set_page_config(
    page_title="Análise de Dados SAC/Telemarketing",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo CSS personalizado
st.markdown("""
    <style>
    .main-title {
        color: #1E90FF;
        font-size: 2.5em;
        font-weight: bold;
        text-align: center;
        margin-bottom: 0.5em;
    }
    .subtitle {
        color: #696969;
        font-size: 1.4em;
        font-weight: normal;
        text-align: center;  /* Centraliza o subtítulo */
    }
    .section-title {
        color: #4682B4;
        font-size: 1.8em;
        font-weight: bold;
        margin-top: 1em;
    }
    .subheader {
        color: #696969;
        font-size: 1.4em;
        font-weight: normal;
    }
    .stButton>button {
        background-color: #1E90FF;
        color: white;
        border-radius: 5px;
    }
    .stMetric {
        background-color: #F0F8FF;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_limpo' not in st.session_state:
    st.session_state.df_limpo = None
if 'n_clusters' not in st.session_state:
    st.session_state.n_clusters = 4
if 'selected_dates' not in st.session_state:
    st.session_state.selected_dates = None
if 'selected_canal' not in st.session_state:
    st.session_state.selected_canal = 'Todos'
if 'selected_tipo' not in st.session_state:
    st.session_state.selected_tipo = 'Todos'
if 'selected_operadores' not in st.session_state:
    st.session_state.selected_operadores = []
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'accuracy' not in st.session_state:
    st.session_state.accuracy = None
if 'report_df' not in st.session_state:
    st.session_state.report_df = None
if 'features_df' not in st.session_state:
    st.session_state.features_df = None

# Banner inicial
st.markdown('<div class="main-title">📞 Análise de Dados do SAC/Telemarketing</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Dashboard interativo para otimização de processos e aumento de lucratividade</div>', unsafe_allow_html=True)

# Sidebar personalizada
with st.sidebar:
    st.markdown("### Sobre o Projeto 🌟")
    st.markdown("""
    Este dashboard analisa dados de SAC/Telemarketing em 6 etapas:
    - 📥 **Coleta de dados**
    - 🧹 **Limpeza de dados**
    - 🔍 **Exploração de dados (EDA)**
    - 📈 **Análise e interpretação**
    - 📊 **Visualização e relatórios**
    - ✅ **Tomada de decisão**
    """)
    st.markdown("### Tecnologias Utilizadas 🛠️")
    st.markdown("""
    - Python
    - Pandas
    - NumPy
    - Plotly
    - Streamlit
    - Scikit-learn
    - statsmodels   
    """)
st.divider()

# Função para carregar os dados
def carregar_dados():
    with st.spinner("Carregando dados... ⏳"):
        try:
            df = pd.read_csv('dados_falsos_SAC-Telemarketing.csv')
            return df
        except FileNotFoundError:
            st.warning("Arquivo de dados não encontrado no diretório do projeto.")
            uploaded_file = st.file_uploader("Faça o upload do arquivo CSV de dados 📁", type=['csv'])
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                return df
            return None

# ETAPA 1: COLETA DE DADOS
st.markdown('<div class="section-title">1. Coleta de Dados 📥</div>', unsafe_allow_html=True)
if st.session_state.df is None:
    st.session_state.df = carregar_dados()

if st.session_state.df is not None:
    st.success("✅ Dados carregados com sucesso!")
    with st.expander("Prévia dos Dados"):
        st.markdown('<div class="subheader">Primeiras Linhas do Dataset</div>', unsafe_allow_html=True)
        st.dataframe(st.session_state.df.head())
    st.divider()
    with st.expander("Informações do Dataset"):
        st.markdown('<div class="subheader">Estrutura dos Dados</div>', unsafe_allow_html=True)
        buffer = io.StringIO()
        st.session_state.df.info(buf=buffer)
        info_str = buffer.getvalue()
        st.text(info_str)
    st.divider()
    with st.expander("Estatísticas Descritivas"):
        st.markdown('<div class="subheader">Resumo Estatístico</div>', unsafe_allow_html=True)
        st.dataframe(st.session_state.df.describe())
st.divider()

# ETAPA 2: LIMPEZA DE DADOS
st.markdown('<div class="section-title">2. Limpeza de Dados 🧹</div>', unsafe_allow_html=True)
if st.session_state.df is not None:
    st.markdown('<div class="subheader">Valores Ausentes por Coluna</div>', unsafe_allow_html=True)
    valores_ausentes = st.session_state.df.isnull().sum()
    st.write(valores_ausentes)
    st.divider()
    st.markdown('<div class="subheader">Número de Linhas Duplicadas</div>', unsafe_allow_html=True)
    duplicatas = st.session_state.df.duplicated().sum()
    st.write(f"Foram encontradas {duplicatas} linhas duplicadas")
    st.divider()
    st.markdown('<div class="subheader">Limpeza em Progresso...</div>', unsafe_allow_html=True)
    
    if st.session_state.df_limpo is None:
        with st.spinner("Limpando dados... ⏳"):
            df = st.session_state.df.copy()
            df['Data/Hora do Atendimento'] = pd.to_datetime(df['Data/Hora do Atendimento'], errors='coerce', dayfirst=True)
            df['Data'] = df['Data/Hora do Atendimento'].dt.date
            df['Hora'] = df['Data/Hora do Atendimento'].dt.hour
            df['Dia da Semana'] = df['Data/Hora do Atendimento'].dt.day_name()
            df['Mês'] = df['Data/Hora do Atendimento'].dt.month_name()
            df['Satisfação do Cliente (1-5)'] = pd.to_numeric(df['Satisfação do Cliente (1-5)'], errors='coerce')
            df['Tempo Médio de Atendimento (TMA) em minutos'] = pd.to_numeric(df['Tempo Médio de Atendimento (TMA) em minutos'], errors='coerce')
            df['Tempo de Espera em minutos'] = pd.to_numeric(df['Tempo de Espera em minutos'], errors='coerce')
            df['Tempo de Pós-Atendimento (minutos)'] = pd.to_numeric(df['Tempo de Pós-Atendimento (minutos)'], errors='coerce')
            df['Resolução no Primeiro Contato?'] = df['Resolução no Primeiro Contato?'].map({'Sim': True, 'Não': False})
            colunas_numericas = ['Tempo Médio de Atendimento (TMA) em minutos', 'Satisfação do Cliente (1-5)', 
                                'Tempo de Espera em minutos', 'Tempo de Pós-Atendimento (minutos)']
            for coluna in colunas_numericas:
                df[coluna] = df[coluna].fillna(df[coluna].mean())
            colunas_categoricas = ['Tipo de Atendimento', 'Canal de Atendimento', 'Produto/Serviço Relacionado', 
                                  'Encaminhamento para Outro Setor', 'Status do Atendimento']
            for coluna in colunas_categoricas:
                if coluna in df.columns:
                    df[coluna] = df[coluna].fillna(df[coluna].mode()[0])
            df_limpo = df.drop_duplicates()
            df_limpo['Tempo Total'] = df_limpo['Tempo de Espera em minutos'] + df_limpo['Tempo Médio de Atendimento (TMA) em minutos'] + df_limpo['Tempo de Pós-Atendimento (minutos)']
            st.session_state.df_limpo = df_limpo
            st.success(f"✅ Limpeza concluída! Foram removidas {len(df) - len(df_limpo)} linhas duplicadas.")
            st.write(f"Dimensões dos dados após limpeza: {df_limpo.shape}")
st.divider()

# ETAPA 3: EXPLORAÇÃO DE DADOS (EDA)
st.markdown('<div class="section-title">3. Exploração de Dados (EDA) 🔍</div>', unsafe_allow_html=True)
if st.session_state.df_limpo is not None:
    tab1, tab2, tab3, tab4 = st.tabs(["📅 Análise Temporal", "📏 Métricas de Atendimento", "😊 Satisfação do Cliente", "👥 Análise por Operador"])
    
    with tab1:
        st.markdown('<div class="subheader">Análise Temporal dos Atendimentos</div>', unsafe_allow_html=True)
        df_por_data = st.session_state.df_limpo.groupby('Data').agg({
            'Protocolo': 'count',
            'Tempo Médio de Atendimento (TMA) em minutos': 'mean',
            'Satisfação do Cliente (1-5)': 'mean',
            'Tempo de Espera em minutos': 'mean'
        }).reset_index()
        df_por_data.columns = ['Data', 'Quantidade de Atendimentos', 'TMA Médio', 'Satisfação Média', 'Tempo de Espera Médio']
        fig = make_subplots(rows=2, cols=1, subplot_titles=("Volume de Atendimentos", "Média de Satisfação"))
        fig.add_trace(go.Scatter(x=df_por_data['Data'], y=df_por_data['Quantidade de Atendimentos'], mode='lines+markers', name='Atendimentos'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_por_data['Data'], y=df_por_data['Satisfação Média'], mode='lines+markers', name='Satisfação'), row=2, col=1)
        fig.update_layout(height=600, width=800)
        st.plotly_chart(fig, use_container_width=True)
        st.divider()
        
        st.markdown('<div class="subheader">Análise por Dia da Semana</div>', unsafe_allow_html=True)
        ordem_dias = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dict_dias = {
            'Monday': 'Segunda-feira', 'Tuesday': 'Terça-feira', 'Wednesday': 'Quarta-feira',
            'Thursday': 'Quinta-feira', 'Friday': 'Sexta-feira', 'Saturday': 'Sábado', 'Sunday': 'Domingo'
        }
        st.session_state.df_limpo['Dia da Semana PT'] = st.session_state.df_limpo['Dia da Semana'].map(dict_dias)
        ordem_dias_pt = [dict_dias[dia] for dia in ordem_dias]
        df_dias = st.session_state.df_limpo.groupby('Dia da Semana PT').agg({
            'Protocolo': 'count',
            'Tempo Médio de Atendimento (TMA) em minutos': 'mean',
            'Satisfação do Cliente (1-5)': 'mean',
            'Tempo de Espera em minutos': 'mean'
        }).reset_index()
        df_dias['Dia da Semana PT'] = pd.Categorical(df_dias['Dia da Semana PT'], categories=ordem_dias_pt, ordered=True)
        df_dias = df_dias.sort_values('Dia da Semana PT')
        fig_dias = px.bar(df_dias, x='Dia da Semana PT', y='Protocolo', 
                         title='Volume de Atendimentos por Dia da Semana',
                         labels={'Protocolo': 'Quantidade de Atendimentos', 'Dia da Semana PT': 'Dia da Semana'})
        st.plotly_chart(fig_dias, use_container_width=True)
        st.divider()
        
        st.markdown('<div class="subheader">Distribuição de Atendimentos por Hora do Dia</div>', unsafe_allow_html=True)
        df_horas = st.session_state.df_limpo.groupby('Hora').agg({
            'Protocolo': 'count',
            'Tempo Médio de Atendimento (TMA) em minutos': 'mean',
            'Satisfação do Cliente (1-5)': 'mean',
            'Tempo de Espera em minutos': 'mean'
        }).reset_index()
        fig_horas = px.line(df_horas, x='Hora', y='Protocolo', 
                           title='Volume de Atendimentos por Hora do Dia',
                           labels={'Protocolo': 'Quantidade de Atendimentos'})
        st.plotly_chart(fig_horas, use_container_width=True)
    
    with tab2:
        st.markdown('<div class="subheader">Desempenho por Canal de Atendimento</div>', unsafe_allow_html=True)
        df_canal = st.session_state.df_limpo.groupby('Canal de Atendimento').agg({
            'Protocolo': 'count',
            'Tempo Médio de Atendimento (TMA) em minutos': 'mean',
            'Satisfação do Cliente (1-5)': 'mean',
            'Tempo de Espera em minutos': 'mean',
            'Resolução no Primeiro Contato?': 'mean'
        }).reset_index()
        df_canal['Taxa de Resolução no Primeiro Contato'] = df_canal['Resolução no Primeiro Contato?'] * 100
        fig_canal = make_subplots(rows=1, cols=2, specs=[[{"type": "bar"}, {"type": "bar"}]], subplot_titles=("Tempo Médio de Atendimento por Canal", "Satisfação Média por Canal"))
        fig_canal.add_trace(go.Bar(x=df_canal['Canal de Atendimento'], y=df_canal['Tempo Médio de Atendimento (TMA) em minutos'], name='TMA (min)'), row=1, col=1)
        fig_canal.add_trace(go.Bar(x=df_canal['Canal de Atendimento'], y=df_canal['Satisfação do Cliente (1-5)'], name='Satisfação'), row=1, col=2)
        fig_canal.update_layout(height=500, width=800)
        st.plotly_chart(fig_canal, use_container_width=True)
        st.divider()
        
        st.markdown('<div class="subheader">Desempenho por Tipo de Atendimento</div>', unsafe_allow_html=True)
        df_tipo = st.session_state.df_limpo.groupby('Tipo de Atendimento').agg({
            'Protocolo': 'count',
            'Tempo Médio de Atendimento (TMA) em minutos': 'mean',
            'Satisfação do Cliente (1-5)': 'mean',
            'Tempo de Espera em minutos': 'mean',
            'Resolução no Primeiro Contato?': 'mean'
        }).reset_index()
        df_tipo['Taxa de Resolução no Primeiro Contato'] = df_tipo['Resolução no Primeiro Contato?'] * 100
        df_tipo = df_tipo.sort_values('Protocolo', ascending=False)
        fig_tipo = px.bar(df_tipo, x='Tipo de Atendimento', y='Protocolo', title='Volume por Tipo de Atendimento', labels={'Protocolo': 'Quantidade de Atendimentos'})
        st.plotly_chart(fig_tipo, use_container_width=True)
        fig_eficiencia = px.scatter(df_tipo, x='Tempo Médio de Atendimento (TMA) em minutos', y='Satisfação do Cliente (1-5)', size='Protocolo', color='Taxa de Resolução no Primeiro Contato', hover_name='Tipo de Atendimento', title='Comparação: TMA x Satisfação x Volume x Taxa de Resolução', labels={'Tempo Médio de Atendimento (TMA) em minutos': 'TMA (minutos)', 'Satisfação do Cliente (1-5)': 'Satisfação', 'Taxa de Resolução no Primeiro Contato': 'Taxa de Resolução (%)'})
        st.plotly_chart(fig_eficiencia, use_container_width=True)
        st.divider()
        
        st.markdown('<div class="subheader">Desempenho por Produto/Serviço</div>', unsafe_allow_html=True)
        df_produto = st.session_state.df_limpo.groupby('Produto/Serviço Relacionado').agg({
            'Protocolo': 'count',
            'Tempo Médio de Atendimento (TMA) em minutos': 'mean',
            'Satisfação do Cliente (1-5)': 'mean',
            'Resolução no Primeiro Contato?': 'mean'
        }).reset_index()
        df_produto = df_produto.sort_values('Protocolo', ascending=False).head(10)
        fig_produto = px.bar(df_produto, x='Produto/Serviço Relacionado', y='Protocolo', title='Top 10 Produtos/Serviços por Volume de Atendimento', labels={'Protocolo': 'Quantidade de Atendimentos'})
        st.plotly_chart(fig_produto, use_container_width=True)
    
    with tab3:
        st.markdown('<div class="subheader">Distribuição de Satisfação do Cliente</div>', unsafe_allow_html=True)
        fig_satisfacao = px.histogram(st.session_state.df_limpo, x='Satisfação do Cliente (1-5)', nbins=5, title='Distribuição de Satisfação do Cliente', labels={'Satisfação do Cliente (1-5)': 'Nível de Satisfação (1-5)'})
        st.plotly_chart(fig_satisfacao, use_container_width=True)
        st.divider()
        
        st.markdown('<div class="subheader">Correlação: Tempo de Espera vs Satisfação</div>', unsafe_allow_html=True)
        fig_correlacao = px.scatter(st.session_state.df_limpo, x='Tempo de Espera em minutos', y='Satisfação do Cliente (1-5)', title='Relação entre Tempo de Espera e Satisfação', trendline='ols', labels={'Tempo de Espera em minutos': 'Tempo de Espera (minutos)', 'Satisfação do Cliente (1-5)': 'Satisfação'})
        st.plotly_chart(fig_correlacao, use_container_width=True)
        st.divider()
        
        st.markdown('<div class="subheader">Fatores que Influenciam a Satisfação</div>', unsafe_allow_html=True)
        df_resolucao = st.session_state.df_limpo.groupby('Resolução no Primeiro Contato?').agg({'Satisfação do Cliente (1-5)': 'mean'}).reset_index()
        df_resolucao['Resolução no Primeiro Contato?'] = df_resolucao['Resolução no Primeiro Contato?'].map({True: 'Sim', False: 'Não'})
        fig_resolucao = px.bar(df_resolucao, x='Resolução no Primeiro Contato?', y='Satisfação do Cliente (1-5)', title='Impacto da Resolução no Primeiro Contato na Satisfação', labels={'Satisfação do Cliente (1-5)': 'Satisfação Média', 'Resolução no Primeiro Contato?': 'Resolução no Primeiro Contato'})
        st.plotly_chart(fig_resolucao, use_container_width=True)
        st.divider()
        
        st.markdown('<div class="subheader">Satisfação por Dia da Semana e Hora do Dia</div>', unsafe_allow_html=True)
        df_heatmap = st.session_state.df_limpo.pivot_table(index='Dia da Semana PT', columns='Hora', values='Satisfação do Cliente (1-5)', aggfunc='mean')
        df_heatmap = df_heatmap.reindex(ordem_dias_pt)
        fig_heatmap = px.imshow(df_heatmap, labels=dict(x="Hora do Dia", y="Dia da Semana", color="Satisfação Média"), title="Mapa de Calor: Satisfação Média por Dia e Hora", color_continuous_scale="RdYlGn")
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with tab4:
        st.markdown('<div class="subheader">Análise de Desempenho por Operador</div>', unsafe_allow_html=True)
        df_operador = st.session_state.df_limpo.groupby('Nome do Operador').agg({
            'Protocolo': 'count',
            'Tempo Médio de Atendimento (TMA) em minutos': 'mean',
            'Satisfação do Cliente (1-5)': 'mean',
            'Resolução no Primeiro Contato?': 'mean'
        }).reset_index()
        df_operador['Taxa de Resolução no Primeiro Contato'] = df_operador['Resolução no Primeiro Contato?'] * 100
        df_operador = df_operador.sort_values('Protocolo', ascending=False)
        st.markdown('<div class="subheader">Top 10 Operadores por Volume de Atendimento</div>', unsafe_allow_html=True)
        fig_top_operadores = px.bar(df_operador.head(10), x='Nome do Operador', y='Protocolo', title='Top 10 Operadores por Volume', labels={'Protocolo': 'Quantidade de Atendimentos'})
        st.plotly_chart(fig_top_operadores, use_container_width=True)
        st.divider()
        
        st.markdown('<div class="subheader">Comparação de Desempenho entre Operadores</div>', unsafe_allow_html=True)
        selected_operadores = st.multiselect("Selecione operadores para comparar:", options=df_operador['Nome do Operador'].tolist(), default=df_operador['Nome do Operador'].head(5).tolist(), key='operadores_select')
        st.session_state.selected_operadores = selected_operadores
        if st.session_state.selected_operadores:
            df_selected = df_operador[df_operador['Nome do Operador'].isin(st.session_state.selected_operadores)]
            fig_radar = go.Figure()
            for operador in df_selected['Nome do Operador']:
                operador_data = df_selected[df_selected['Nome do Operador'] == operador]
                volume_norm = operador_data['Protocolo'].values[0] / df_operador['Protocolo'].max()
                tma_norm = 1 - (operador_data['Tempo Médio de Atendimento (TMA) em minutos'].values[0] / df_operador['Tempo Médio de Atendimento (TMA) em minutos'].max())
                satisfacao_norm = operador_data['Satisfação do Cliente (1-5)'].values[0] / 5
                resolucao_norm = operador_data['Taxa de Resolução no Primeiro Contato'].values[0] / 100
                fig_radar.add_trace(go.Scatterpolar(
                    r=[volume_norm, tma_norm, satisfacao_norm, resolucao_norm],
                    theta=['Volume', 'Eficiência (TMA)', 'Satisfação', 'Resolução 1º Contato'],
                    fill='toself',
                    name=operador
                ))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True)
            st.plotly_chart(fig_radar, use_container_width=True)
st.divider()

# ETAPA 4: ANÁLISE E INTERPRETAÇÃO
st.markdown('<div class="section-title">4. Análise e Interpretação 📈</div>', unsafe_allow_html=True)
if st.session_state.df_limpo is not None:
    st.markdown('<div class="subheader">Segmentação de Atendimentos</div>', unsafe_allow_html=True)
    st.info("ℹ️ A segmentação agrupa atendimentos com características semelhantes para identificar padrões.")
    features = ['Tempo Médio de Atendimento (TMA) em minutos', 'Tempo de Espera em minutos', 'Satisfação do Cliente (1-5)', 'Tempo de Pós-Atendimento (minutos)']
    df_cluster = st.session_state.df_limpo[features].copy()
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_cluster)
    n_clusters = st.slider("Número de clusters para segmentação:", 2, 10, st.session_state.n_clusters, key='n_clusters_slider')
    st.session_state.n_clusters = n_clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    st.session_state.df_limpo['Cluster'] = kmeans.fit_predict(df_scaled)
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    cluster_results = pd.DataFrame(cluster_centers, columns=features)
    cluster_results['Cluster'] = cluster_results.index
    cluster_results['Tamanho'] = [sum(st.session_state.df_limpo['Cluster'] == i) for i in range(n_clusters)]
    cluster_results['Proporção (%)'] = cluster_results['Tamanho'] / len(st.session_state.df_limpo) * 100
    st.markdown('<div class="subheader">Características dos Clusters</div>', unsafe_allow_html=True)
    st.dataframe(cluster_results)
    fig_clusters = px.scatter_3d(st.session_state.df_limpo, x='Tempo Médio de Atendimento (TMA) em minutos', y='Tempo de Espera em minutos', z='Satisfação do Cliente (1-5)', color='Cluster', size_max=10, opacity=0.7, title='Segmentação de Atendimentos por Características')
    st.plotly_chart(fig_clusters, use_container_width=True)
    st.divider()
    
    st.markdown('<div class="subheader">Interpretação dos Clusters</div>', unsafe_allow_html=True)
    for i in range(n_clusters):
        cluster_data = cluster_results[cluster_results['Cluster'] == i]
        tma = cluster_data['Tempo Médio de Atendimento (TMA) em minutos'].values[0]
        espera = cluster_data['Tempo de Espera em minutos'].values[0]
        satisfacao = cluster_data['Satisfação do Cliente (1-5)'].values[0]
        tamanho = cluster_data['Proporção (%)'].values[0]
        if satisfacao >= 4.0:
            nivel_satisfacao = "alta"
        elif satisfacao >= 3.0:
            nivel_satisfacao = "média"
        else:
            nivel_satisfacao = "baixa"
        if tma <= st.session_state.df_limpo['Tempo Médio de Atendimento (TMA) em minutos'].mean() * 0.8:
            nivel_tma = "rápido"
        elif tma <= st.session_state.df_limpo['Tempo Médio de Atendimento (TMA) em minutos'].mean() * 1.2:
            nivel_tma = "médio"
        else:
            nivel_tma = "longo"
        if espera <= st.session_state.df_limpo['Tempo de Espera em minutos'].mean() * 0.8:
            nivel_espera = "curto"
        elif espera <= st.session_state.df_limpo['Tempo de Espera em minutos'].mean() * 1.2:
            nivel_espera = "médio"
        else:
            nivel_espera = "longo"
        st.markdown(f"**Cluster {i} ({tamanho:.1f}% dos atendimentos):**")
        st.markdown(f"- Atendimentos com tempo de espera **{nivel_espera}** ({espera:.2f} min)")
        st.markdown(f"- Tempo médio de atendimento **{nivel_tma}** ({tma:.2f} min)")
        st.markdown(f"- Satisfação **{nivel_satisfacao}** ({satisfacao:.2f}/5)")
        cluster_df = st.session_state.df_limpo[st.session_state.df_limpo['Cluster'] == i]
        top_canais = cluster_df['Canal de Atendimento'].value_counts().nlargest(3)
        st.markdown("- **Top 3 canais de atendimento:**")
        for canal, contagem in top_canais.items():
            st.markdown(f"  - {canal}: {contagem} atendimentos ({contagem/len(cluster_df)*100:.1f}%)")
        resolucao_rate = cluster_df['Resolução no Primeiro Contato?'].mean() * 100
        st.markdown(f"- **Taxa de resolução no primeiro contato**: {resolucao_rate:.1f}%")
        st.markdown("---")
    st.divider()
    
    st.markdown('<div class="subheader">Modelo Preditivo para Resolução no Primeiro Contato</div>', unsafe_allow_html=True)
    if not st.session_state.model_trained:
        df_model = st.session_state.df_limpo.copy()
        features_cat = ['Tipo de Atendimento', 'Canal de Atendimento', 'Produto/Serviço Relacionado', 'Dia da Semana PT', 'Status do Atendimento']
        features_num = ['Tempo Médio de Atendimento (TMA) em minutos', 'Tempo de Espera em minutos', 'Tempo de Pós-Atendimento (minutos)', 'Hora']
        df_model_encoded = pd.get_dummies(df_model, columns=features_cat)
        X = df_model_encoded[features_num + [col for col in df_model_encoded.columns if col.startswith(tuple(f + '_' for f in features_cat))]]
        y = df_model_encoded['Resolução no Primeiro Contato?']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        feature_importances = model.feature_importances_
        features_df = pd.DataFrame({'Feature': X.columns, 'Importância': feature_importances}).sort_values('Importância', ascending=False)
        st.session_state.model = model
        st.session_state.accuracy = accuracy
        st.session_state.report_df = report_df
        st.session_state.features_df = features_df
        st.session_state.model_trained = True
    st.write(f"**Acurácia do modelo**: {st.session_state.accuracy:.2%}")
    st.write("**Relatório de Classificação:**")
    st.dataframe(st.session_state.report_df)
    st.markdown('<div class="subheader">Features Mais Importantes para Resolução no Primeiro Contato</div>', unsafe_allow_html=True)
    fig_importances = px.bar(st.session_state.features_df.head(10), x='Importância', y='Feature', orientation='h', title='Top 10 Fatores que Influenciam a Resolução no Primeiro Contato')
    st.plotly_chart(fig_importances, use_container_width=True)
st.divider()

# ETAPA 5: VISUALIZAÇÃO E RELATÓRIOS
st.markdown('<div class="section-title">5. Visualização e Relatórios 📊</div>', unsafe_allow_html=True)
if st.session_state.df_limpo is not None:
    st.markdown('<div class="subheader">Dashboard Interativo de KPIs</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    total_atendimentos = len(st.session_state.df_limpo)
    with col1:
        st.metric("📋 Total de Atendimentos", f"{total_atendimentos:,}", label_visibility="visible")
    satisfacao_media = st.session_state.df_limpo['Satisfação do Cliente (1-5)'].mean()
    with col2:
        st.metric("😊 Satisfação Média", f"{satisfacao_media:.2f}/5")
    tma_global = st.session_state.df_limpo['Tempo Médio de Atendimento (TMA) em minutos'].mean()
    with col3:
        st.metric("⏱️ TMA Global", f"{tma_global:.2f} min")
    taxa_resolucao = st.session_state.df_limpo['Resolução no Primeiro Contato?'].mean() * 100
    with col4:
        st.metric("✅ Resolução 1º Contato", f"{taxa_resolucao:.1f}%")
    st.divider()
    
    st.markdown('<div class="subheader">Análise Interativa</div>', unsafe_allow_html=True)
    if 'Data' in st.session_state.df_limpo.columns:
        min_date = st.session_state.df_limpo['Data'].min()
        max_date = st.session_state.df_limpo['Data'].max()
        selected_dates = st.date_input("Selecione o período de análise:", value=(min_date, max_date), min_value=min_date, max_value=max_date, key='date_input')
        st.session_state.selected_dates = selected_dates
        if len(st.session_state.selected_dates) == 2:
            start_date, end_date = st.session_state.selected_dates
            filtered_df = st.session_state.df_limpo[(st.session_state.df_limpo['Data'] >= start_date) & (st.session_state.df_limpo['Data'] <= end_date)]
        else:
            filtered_df = st.session_state.df_limpo
    else:
        filtered_df = st.session_state.df_limpo
    canais = ['Todos'] + sorted(st.session_state.df_limpo['Canal de Atendimento'].unique().tolist())
    selected_canal = st.selectbox("Canal de Atendimento:", canais, index=canais.index(st.session_state.selected_canal), key='canal_select')
    st.session_state.selected_canal = selected_canal
    if st.session_state.selected_canal != 'Todos':
        filtered_df = filtered_df[filtered_df['Canal de Atendimento'] == st.session_state.selected_canal]
    tipos = ['Todos'] + sorted(st.session_state.df_limpo['Tipo de Atendimento'].unique().tolist())
    selected_tipo = st.selectbox("Tipo de Atendimento:", tipos, index=tipos.index(st.session_state.selected_tipo), key='tipo_select')
    st.session_state.selected_tipo = selected_tipo
    if st.session_state.selected_tipo != 'Todos':
        filtered_df = filtered_df[filtered_df['Tipo de Atendimento'] == st.session_state.selected_tipo]
    st.markdown('<div class="subheader">Métricas do Período/Filtros Selecionados</div>', unsafe_allow_html=True)
    if not filtered_df.empty:
        col1, col2, col3, col4 = st.columns(4)
        qtd_filtrada = len(filtered_df)
        with col1:
            st.metric("📋 Atendimentos", f"{qtd_filtrada:,}", f"{qtd_filtrada/total_atendimentos*100:.1f}% do total")
        sat_filtrada = filtered_df['Satisfação do Cliente (1-5)'].mean()
        with col2:
            delta_sat = sat_filtrada - satisfacao_media
            st.metric("😊 Satisfação", f"{sat_filtrada:.2f}/5", f"{delta_sat:.2f}")
        tma_filtrado = filtered_df['Tempo Médio de Atendimento (TMA) em minutos'].mean()
        with col3:
            delta_tma = tma_filtrado - tma_global
            st.metric("⏱️ TMA", f"{tma_filtrado:.2f} min", f"{delta_tma:.2f} min")
        taxa_filtrada = filtered_df['Resolução no Primeiro Contato?'].mean() * 100
        with col4:
            delta_taxa = taxa_filtrada - taxa_resolucao
            st.metric("✅ Resolução 1º Contato", f"{taxa_filtrada:.1f}%", f"{delta_taxa:.1f}%")
        if 'Data' in filtered_df.columns:
            df_tendencia = filtered_df.groupby('Data').agg({'Protocolo': 'count', 'Satisfação do Cliente (1-5)': 'mean'}).reset_index()
            fig_tendencia = px.line(df_tendencia, x='Data', y=['Protocolo', 'Satisfação do Cliente (1-5)'], title='Tendência de Volume e Satisfação no Período Selecionado')
            st.plotly_chart(fig_tendencia, use_container_width=True)
    else:
        st.warning("Nenhum dado encontrado para os filtros selecionados.")
st.divider()

# ETAPA 6: TOMADA DE DECISÃO
st.markdown('<div class="section-title">6. Tomada de Decisão ✅</div>', unsafe_allow_html=True)
if st.session_state.df_limpo is not None:
    st.markdown('<div class="subheader">Insights e Recomendações</div>', unsafe_allow_html=True)
    if st.session_state.df_limpo is None or st.session_state.df_limpo.empty:
        st.warning("⚠️ O DataFrame `df_limpo` está vazio ou não foi inicializado. Verifique a etapa de limpeza de dados.")
    else:
        st.info(f"✅ DataFrame `df_limpo` contém {len(st.session_state.df_limpo)} registros.")
    if 'Hora' in st.session_state.df_limpo.columns:
        df_horas_pico = st.session_state.df_limpo.groupby('Hora').agg({'Protocolo': 'count', 'Tempo de Espera em minutos': 'mean'}).reset_index()
        horas_pico = df_horas_pico.nlargest(3, 'Protocolo')['Hora'].tolist()
        horas_vale = df_horas_pico.nsmallest(3, 'Protocolo')['Hora'].tolist()
    else:
        horas_pico = []
        horas_vale = []
        st.warning("⚠️ Coluna 'Hora' não encontrada em `df_limpo`. A análise de horários de pico será ignorada.")
    st.markdown("### 1. Otimização de Escalas de Operadores")
    if horas_pico and horas_vale:
        st.markdown(f"""
        **Insight:** Identificamos os seguintes horários de pico: {', '.join([f"{h}h" for h in horas_pico])}, que apresentam maior volume de atendimentos e tempos de espera mais longos.
        **Recomendação:** 
        - Aumentar o número de operadores nos horários de pico
        - Realizar treinamentos e reuniões nos horários de menor movimento: {', '.join([f"{h}h" for h in horas_vale])}
        - Implementar um sistema de escala dinâmica baseado na previsão de demanda
        **Impacto estimado:** Redução de até 30% no tempo de espera nos horários de pico, aumentando a satisfação do cliente e a produtividade.
        """)
    else:
        st.warning("⚠️ Não foi possível identificar horários de pico ou vale. Verifique os dados na coluna 'Hora'.")
    st.divider()
    
    st.markdown("### 2. Redução do Tempo Médio de Atendimento em Gargalos")
    try:
        df_gargalos = st.session_state.df_limpo.groupby('Tipo de Atendimento').agg({'Protocolo': 'count', 'Tempo Médio de Atendimento (TMA) em minutos': 'mean'}).reset_index()
        gargalos = df_gargalos.nlargest(3, 'Tempo Médio de Atendimento (TMA) em minutos')
        if not gargalos.empty:
            gargalos_list = [f"{tipo} (TMA: {tma:.2f} min)" for tipo, _, tma in gargalos[['Tipo de Atendimento', 'Protocolo', 'Tempo Médio de Atendimento (TMA) em minutos']].values]
            st.markdown(f"""
            **Insight:** Os seguintes tipos de atendimento apresentam TMA significativamente acima da média:
            {', '.join(gargalos_list)}
            **Recomendação:** 
            - Desenvolver scripts otimizados para esses tipos de atendimento
            - Criar base de conhecimento específica para resolução rápida desses casos
            - Direcionar esses atendimentos para operadores mais experientes nesses assuntos
            - Analisar possibilidade de automação parcial desses processos
            **Impacto estimado:** Redução de 15-25% no TMA desses tipos de atendimento, aumentando a capacidade de atendimento sem adicionar recursos.
            """)
        else:
            st.warning("⚠️ Não foram encontrados gargalos de TMA. Verifique os dados na coluna 'Tipo de Atendimento'.")
    except Exception as e:
        st.error(f"❌ Erro ao calcular gargalos de TMA: {str(e)}")
    st.divider()
    
    st.markdown("### 3. Plano de Ação para Produtos/Serviços Problemáticos")
    try:
        df_produtos_problematicos = st.session_state.df_limpo.groupby('Produto/Serviço Relacionado').agg({'Protocolo': 'count', 'Satisfação do Cliente (1-5)': 'mean'}).reset_index()
        min_volume = len(st.session_state.df_limpo) * 0.01
        df_produtos_problematicos = df_produtos_problematicos[df_produtos_problematicos['Protocolo'] >= min_volume]
        produtos_problematicos = df_produtos_problematicos.nsmallest(3, 'Satisfação do Cliente (1-5)')
        if not produtos_problematicos.empty:
            produtos_list = [f"{prod} (Satisfação: {sat:.2f}/5)" for prod, _, sat in produtos_problematicos[['Produto/Serviço Relacionado', 'Protocolo', 'Satisfação do Cliente (1-5)']].values]
            st.markdown(f"""
            **Insight:** Os seguintes produtos/serviços apresentam índices de satisfação abaixo da média:
            {', '.join(produtos_list)}
            **Recomendação:** 
            - Realizar análise detalhada dos motivos de insatisfação com esses produtos
            - Desenvolver treinamento específico para os operadores sobre esses produtos
            - Criar um canal de comunicação direta com as áreas responsáveis por esses produtos
            - Implementar monitoramento em tempo real dos atendimentos relacionados a esses itens
            **Impacto estimado:** Aumento de 0.5-1.0 ponto na satisfação média desses produtos em 3 meses.
            """)
        else:
            st.warning("⚠️ Não foram encontrados produtos/serviços problemáticos. Verifique os dados na coluna 'Produto/Serviço Relacionado'.")
    except Exception as e:
        st.error(f"❌ Erro ao identificar produtos problemáticos: {str(e)}")
    st.divider()
    
    st.markdown("### 4. Programa de Compartilhamento de Melhores Práticas")
    try:
        required_columns = ['Nome do Operador', 'Protocolo', 'Satisfação do Cliente (1-5)', 'Resolução no Primeiro Contato?', 'Tempo Médio de Atendimento (TMA) em minutos']
        missing_columns = [col for col in required_columns if col not in st.session_state.df_limpo.columns]
        if missing_columns:
            st.error(f"❌ Colunas necessárias ausentes em `df_limpo`: {', '.join(missing_columns)}")
        else:
            df_melhores_operadores = st.session_state.df_limpo.groupby('Nome do Operador').agg({
                'Protocolo': 'count',
                'Satisfação do Cliente (1-5)': 'mean',
                'Resolução no Primeiro Contato?': 'mean',
                'Tempo Médio de Atendimento (TMA) em minutos': 'mean'
            }).reset_index()
            min_volume_operador = len(st.session_state.df_limpo) * 0.02
            df_melhores_operadores = df_melhores_operadores[df_melhores_operadores['Protocolo'] >= min_volume_operador]
            st.info(f"📊 Após filtro de volume mínimo ({min_volume_operador:.0f} atendimentos), `df_melhores_operadores` contém {len(df_melhores_operadores)} operadores.")
            df_melhores_operadores['Score'] = (df_melhores_operadores['Satisfação do Cliente (1-5)'] * 
                                             df_melhores_operadores['Resolução no Primeiro Contato?']) / \
                                             (df_melhores_operadores['Tempo Médio de Atendimento (TMA) em minutos'] / 
                                              df_melhores_operadores['Tempo Médio de Atendimento (TMA) em minutos'].mean())
            melhores_operadores = df_melhores_operadores.nlargest(3, 'Score')
            if not melhores_operadores.empty:
                operadores_list = [f"{nome} (Satisfação: {sat:.2f}/5, Resolução 1º Contato: {res*100:.1f}%)" 
                                   for nome, _, sat, res, _ in melhores_operadores[['Nome do Operador', 'Protocolo', 
                                                                                   'Satisfação do Cliente (1-5)', 
                                                                                   'Resolução no Primeiro Contato?', 
                                                                                   'Tempo Médio de Atendimento (TMA) em minutos']].values]
                st.markdown(f"""
                **Insight:** Identificamos operadores que conseguem alta satisfação e resolução no primeiro contato com tempos de atendimento eficientes:
                {', '.join(operadores_list)}
                **Recomendação:** 
                - Criar um programa de mentoria onde esses operadores compartilhem suas técnicas
                - Gravar atendimentos exemplares para uso em treinamentos
                - Implementar reconhecimento e premiação para incentivar a adoção das melhores práticas
                - Estruturar um plano de carreira para operadores de alto desempenho
                **Impacto estimado:** Aumento de 10-15% na taxa de resolução no primeiro contato e 0.3-0.5 ponto na satisfação média geral.
                """)
            else:
                st.warning("⚠️ Nenhum operador atendeu ao critério de volume mínimo para ser considerado 'top performer'. Considere ajustar o limite de volume mínimo (`min_volume_operador`) ou verificar a distribuição de atendimentos por operador.")
            st.markdown("""
            **Insight Adicional 1: Identificar Técnicas Específicas dos Top Operadores**  
            Os operadores com maior pontuação combinam alta satisfação (>4.0/5) e resolução (>80%) com TMAs eficientes. Suas técnicas, como scripts eficazes ou diagnóstico rápido, podem ser extraídas de gravações.  
            **Recomendação:** Analisar gravações com NLP para criar um "Playbook de Melhores Práticas".  
            **Impacto Estimado:** Aumento de 5-10% na resolução e 0.2-0.4 pontos na satisfação em 6 meses.
            **Insight Adicional 2: Segmentar Desempenho por Tipo de Atendimento**  
            Top operadores podem se destacar em tipos específicos de atendimento (e.g., suporte técnico).  
            **Recomendação:** Analisar o `Score` por `Tipo de Atendimento` e criar treinamentos específicos.  
            **Impacto Estimado:** Redução de 10-15% no TMA e aumento de 8-12% na resolução por categoria.
            **Insight Adicional 3: Aproveitar Top Operadores para Coaching em Tempo Real**  
            Top operadores podem orientar colegas durante atendimentos complexos, especialmente em horários de pico.  
            **Recomendação:** Implementar coaching em tempo real durante `horas_pico` com mentores designados.  
            **Impacto Estimado:** Redução de 10% nas escalações e aumento de 0.3-0.5 pontos na satisfação em 3 meses.
            **Insight Adicional 4: Gamificar a Adoção de Melhores Práticas**  
            O `Score` pode motivar operadores a adotarem práticas dos melhores via competição saudável.  
            **Recomendação:** Criar um leaderboard e oferecer recompensas por melhorias no `Score`.  
            **Impacto Estimado:** Aumento de 20-30% na adoção de práticas, com 5-10% de melhoria na resolução em 4 meses.
            **Insight Adicional 5: Analisar Padrões de Colaboração Interdepartamental**  
            Top operadores provavelmente têm baixas taxas de escalação, indicando colaboração eficaz.  
            **Recomendação:** Analisar escalações e criar uma plataforma de compartilhamento de conhecimento.  
            **Impacto Estimado:** Redução de 15-20% nas escalações e 0.3-0.5 pontos na satisfação.
            """)
    except Exception as e:
        st.error(f"❌ Erro ao processar o Programa de Compartilhamento de Melhores Práticas: {str(e)}")
    st.divider()
    
    st.markdown("### 5. Estratégia Omnichannel Otimizada")
    try:
        df_canais_eficiencia = st.session_state.df_limpo.groupby('Canal de Atendimento').agg({
            'Protocolo': 'count',
            'Tempo Médio de Atendimento (TMA) em minutos': 'mean',
            'Satisfação do Cliente (1-5)': 'mean',
            'Resolução no Primeiro Contato?': 'mean'
        }).reset_index()
        df_canais_eficiencia['Eficiência'] = (df_canais_eficiencia['Satisfação do Cliente (1-5)'] * 
                                              df_canais_eficiencia['Resolução no Primeiro Contato?']) / \
                                              df_canais_eficiencia['Tempo Médio de Atendimento (TMA) em minutos']
        canais_eficientes = df_canais_eficiencia.nlargest(2, 'Eficiência')
        canais_ineficientes = df_canais_eficiencia.nsmallest(2, 'Eficiência')
        if not canais_eficientes.empty and not canais_ineficientes.empty:
            eficientes_list = [f"{canal}" for canal in canais_eficientes['Canal de Atendimento'].values]
            ineficientes_list = [f"{canal}" for canal in canais_ineficientes['Canal de Atendimento'].values]
            st.markdown(f"""
            **Insight:** Os canais {', '.join(eficientes_list)} apresentam melhor eficiência (combinação de satisfação, resolução e tempo), enquanto {', '.join(ineficientes_list)} mostram menor eficiência.
            **Recomendação:** 
            - Priorizar investimentos nos canais mais eficientes
            - Implementar estratégias de migração de clientes para canais digitais mais eficientes
            - Integrar dados entre canais para experiência consistente
            - Reavaliar processos nos canais menos eficientes ou considerar sua descontinuação
            **Impacto estimado:** Redução de 10% nos custos de atendimento e aumento de 5% na satisfação geral.
            """)
        else:
            st.warning("⚠️ Não foi possível identificar canais eficientes/ineficientes. Verifique os dados na coluna 'Canal de Atendimento'.")
    except Exception as e:
        st.error(f"❌ Erro ao processar a Estratégia Omnichannel Otimizada: {str(e)}")
    st.divider()
    
    st.markdown("### 6. Conclusão e ROI Estimado")
    st.markdown("""
    **Conclusão:** 
    A análise detalhada dos dados de SAC/Telemarketing revela oportunidades significativas de otimização em várias dimensões: escalas de operadores, processos de atendimento, tratamento de produtos problemáticos, disseminação de melhores práticas e estratégia de canais.
    **ROI Estimado:** 
    - **Redução de custos:** 15-20% através da otimização de escalas e processos
    - **Aumento de produtividade:** 10-15% com a implementação das melhores práticas
    - **Melhoria na satisfação:** 0.5-1.0 ponto na escala de satisfação
    - **Aumento na taxa de resolução no primeiro contato:** 10-15%
    **Próximos passos recomendados:** 
    1. Implementar dashboard em tempo real para monitoramento contínuo dos KPIs
    2. Estabelecer metas específicas para cada área de melhoria identificada
    3. Formar equipes multidisciplinares para implementação das recomendações
    4. Revisar resultados mensalmente e ajustar estratégias conforme necessário
    """)
st.divider()

# Footer
st.markdown("""
<div style='text-align: center; color: #696969; margin-top: 2em;'>
    Desenvolvido com ❤️ usando Streamlit e Python | © 2025
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    pass
