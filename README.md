📞 Análise de Dados SAC/Telemarketing
Bem-vindo ao Análise de Dados SAC/Telemarketing! 🎉 Este é um dashboard interativo desenvolvido para otimizar processos de atendimento ao cliente e aumentar a lucratividade. Ele analisa dados de SAC e telemarketing, fornecendo insights acionáveis por meio de visualizações avançadas e modelos preditivos.

🚀 Visão Geral
Este projeto é um aplicativo web construído com Streamlit que realiza uma análise completa de dados de SAC/Telemarketing em 6 etapas:

📥 Coleta de Dados  
🧹 Limpeza de Dados  
🔍 Exploração de Dados (EDA)  
📈 Análise e Interpretação  
📊 Visualização e Relatórios  
✅ Tomada de Decisão

O dashboard oferece gráficos interativos, segmentação de atendimentos, modelos preditivos e recomendações práticas para melhorar a eficiência operacional e a satisfação do cliente.

🛠️ Tecnologias Utilizadas
Aqui estão as principais tecnologias e bibliotecas usadas no projeto:

Python 🐍: Linguagem principal para desenvolvimento.
Streamlit 🌐: Framework para criar o dashboard interativo.
Pandas 📊: Manipulação e limpeza de dados.
NumPy 🔢: Operações numéricas e cálculos.
Plotly 📈: Visualizações interativas e gráficos dinâmicos.
Scikit-learn 🤖: Modelos de machine learning (KMeans para clustering e RandomForest para predição).
Statsmodels 📉: Análise estatística avançada.


📋 Pré-requisitos
Antes de começar, você precisa ter o seguinte instalado no seu ambiente:

Python 3.8 ou superior 🐍
pip (gerenciador de pacotes do Python)
Um editor de código como VS Code ou PyCharm ✍️
Opcional: Git para clonar o repositório 🚀


🖥️ Passo a Passo para Executar o Projeto
Siga estas etapas para configurar e rodar o projeto localmente:
1. Clone o Repositório 📂
Clone este repositório para sua máquina local usando o comando abaixo:
git clone https://github.com/seu-usuario/analise-telemarketing.git
cd analise-telemarketing

2. Crie um Ambiente Virtual 🌍
É recomendado usar um ambiente virtual para isolar as dependências do projeto:
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate

3. Instale as Dependências 📦
O projeto inclui um arquivo requirements.txt com todas as dependências necessárias. Instale-as com:
pip install -r requirements.txt

O conteúdo do requirements.txt é:
streamlit==1.36.0
pandas==2.2.2
numpy==1.26.4
plotly==5.22.0
scikit-learn==1.5.0
statsmodels==0.14.2

4. Prepare os Dados 📊
O projeto espera um arquivo CSV chamado dados_falsos_SAC-Telemarketing.csv no diretório principal. Você pode:

Usar o arquivo de exemplo: Certifique-se de que ele contém as colunas necessárias (veja a seção "Estrutura do CSV" abaixo).
Fazer upload pelo app: O dashboard permite que você carregue um CSV personalizado.

Estrutura do CSV
O arquivo CSV deve conter as seguintes colunas:



Coluna
Descrição



Data/Hora do Atendimento
Data e hora do atendimento (formato: AAAA-MM-DD HH:MM:SS)


Protocolo
Identificador único do atendimento


Nome do Operador
Nome do operador que realizou o atendimento


Tempo Médio de Atendimento (TMA) em minutos
Tempo médio de atendimento (em minutos)


Satisfação do Cliente (1-5)
Nível de satisfação do cliente (1 a 5)


Tempo de Espera em minutos
Tempo de espera do cliente (em minutos)


Tempo de Pós-Atendimento (minutos)
Tempo pós-atendimento (em minutos)


Resolução no Primeiro Contato?
"Sim" ou "Não" (resolução no 1º contato)


Tipo de Atendimento
Tipo de atendimento (ex.: Suporte Técnico)


Canal de Atendimento
Canal usado (ex.: Telefone, Chat)


Produto/Serviço Relacionado
Produto ou serviço relacionado


Encaminhamento para Outro Setor
"Sim" ou "Não" (se foi encaminhado)


Status do Atendimento
Status (ex.: Concluído, Pendente)


Exemplo de Linha no CSV:
Data/Hora do Atendimento,Protocolo,Nome do Operador,Tempo Médio de Atendimento (TMA) em minutos,Satisfação do Cliente (1-5),Tempo de Espera em minutos,Tempo de Pós-Atendimento (minutos),Resolução no Primeiro Contato?,Tipo de Atendimento,Canal de Atendimento,Produto/Serviço Relacionado,Encaminhamento para Outro Setor,Status do Atendimento
2023-01-01 08:00:00,12345,João,5.2,4,2.1,1.0,Sim,Suporte Técnico,Telefone,Internet,Não,Concluído

5. Execute o Aplicativo 🚀
Inicie o dashboard com o comando abaixo:
streamlit run app.py


Uma janela do navegador abrirá automaticamente com o app rodando em http://localhost:8501.
Se o arquivo dados_falsos_SAC-Telemarketing.csv não estiver presente, você poderá fazer upload de um CSV diretamente no app.

6. Explore o Dashboard 🌟

Carregue os dados e siga as 6 etapas do dashboard.
Interaja com os gráficos, filtros e modelos para obter insights.
Veja as recomendações na seção de Tomada de Decisão para otimizar seu SAC/Telemarketing.


🌟 Funcionalidades do Dashboard

Visualizações Interativas 📈: Gráficos Plotly para análise temporal, desempenho por canal, satisfação do cliente e mais.
Segmentação de Atendimentos 🔍: Usa KMeans para agrupar atendimentos por características semelhantes.
Modelo Preditivo 🤖: RandomForest para prever a resolução no primeiro contato.
KPIs e Filtros Dinâmicos 🎛️: Filtre por datas, canais e tipos de atendimento para análises personalizadas.
Recomendações Práticas ✅: Insights para otimizar escalas, reduzir TMA e melhorar a satisfação.


🐛 Solução de Problemas
Erro: "Colunas ausentes no dataset"

Certifique-se de que seu CSV contém todas as colunas listadas na seção "Estrutura do CSV".
Verifique se os nomes das colunas estão exatamente iguais (incluindo espaços e maiúsculas/minúsculas).

Erro: "ModuleNotFoundError"

Confirme que todas as dependências foram instaladas corretamente com pip install -r requirements.txt.
Verifique se o ambiente virtual está ativado (source venv/bin/activate ou venv\Scripts\activate no Windows).

Desempenho Lento

Para datasets grandes, o app pode ser mais lento. Teste com um subconjunto de dados:# No início do app.py, adicione:
st.session_state.data['df'] = st.session_state.data['df'].head(1000)



Gráficos Não Carregam

Atualize sua versão do Plotly: pip install --upgrade plotly.
Certifique-se de que sua conexão com a internet está ativa (Streamlit pode precisar dela para renderizar gráficos).


📈 Melhorias Futuras

Integração com APIs 🌐: Adicionar suporte para carregar dados em tempo real.
Automação Avançada 🤖: Implementar chatbots para pré-atendimento baseado nos insights.
Exportação de Relatórios 📄: Permitir exportar gráficos e recomendações em PDF.
Monitoramento em Tempo Real ⏰: Criar um sistema de alertas para KPIs críticos.


🤝 Contribuições
Contribuições são bem-vindas! 🚀 Siga estas etapas:

Faça um fork do repositório 🍴
Crie uma branch para sua feature: git checkout -b minha-feature
Faça suas alterações e commit: git commit -m "Adiciona minha feature"
Envie para o repositório remoto: git push origin minha-feature
Abra um Pull Request 📥


📜 Licença
Este projeto está licenciado sob a MIT License.

💡 Sobre o Autor
Desenvolvido com ❤️ por [Lucas Brito].📧 Entre em contato: Lucasbrito_dev@outlook.com.br🌐 GitHub: Lucas-Brito-Dev

Obrigado por usar o Análise de Dados SAC/Telemarketing! 😊 Se gostou, deixe uma ⭐ no GitHub!
