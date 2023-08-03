import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

label_colors = {
    'canario_do_amazonas':'#ffc600', #amarelo
    'chupim':'#9e4347', #marrom
    'rolinha':'#2d3752', #acizentado
    'sanhaco_da_amazonia':'#072ac8', #azul
    'sanhaco_do_coqueiro':'#2b9348' #verde
}  

def make_epoch_history_traces(epoch_history_path, showlegend = False):
    '''
    Função responsável por criar os traces que serão adicionados às figuras dos gráficos. 
    
    Parâmetros
    ----------
        epoch_history_path : caminho para o CSV contendo o histórico de epochs
        showlegend : se as legendas serão apresentadas 
    
    Retorno 
    ----------
        Uma lista com os traces de cada coluna (train_loss, valid_loss e mAP)
    '''
    

    df_column_names = ['train_loss','valid_loss', 'mAP']
    trace_names = {'train_loss':'Loss de Treinamento', 'valid_loss':'Loss de validação', 'mAP':'mAP'}
    line_color_values = {'train_loss':'#636efa', 'valid_loss':'#EF553B', 'mAP':'#00cc96'}

    epoch_history_df = pd.read_csv(epoch_history_path)
    traces = []

    
    for column_name in df_column_names:

        # especifica os valores das epochs
        x = epoch_history_df['epochs'] 

        # especifica os valores das colunas do dataframe (train_loss, valid_loss ou mAP) 
        y = epoch_history_df[column_name] 

        # especifica o nome do trace
        trace_name = trace_names[column_name] 

        # especifica a cor da linha de acordo com o tipo de coluna do dataframe
        line_color = {'color': line_color_values[column_name]} 

        # lista de textos customizados apresentando o tipo de métrica seguida de seu valor para cada epoch
        custom_text = [f'{trace_name}: {metric:.4f}' for metric in y] 

        # template do texto a ser exibido no hover
        hovertemplate = 'Epoch: %{x} <br>%{text}<extra></extra>'
        
        # acrescenta o trace à lista
        traces.append(go.Scatter(
            x = x, 
            y = y,
            line = line_color,
            name =  trace_name,
            text = custom_text, 
            hovertemplate = hovertemplate,
            showlegend = showlegend
        ))
        
    return traces

def make_epoch_history_graph(model_path):
    '''
    Função responsável por criar o gráfico dinâmico do histórico de epochs.
    
    O gráfico é salvo como um arquivo HTML. 
    
    Parâmetros
    ----------
        model_path : caminho da pasta do modelo
    
    Retorno 
    ----------
        Sem retorno, apenas salva o gráfico como um arquivo HTML
    '''
     
    fig = go.Figure() # cria figura
    
    epoch_history_path = model_path+'epoch_history.csv' # caminho para o arquivo CSV contendo o histório das epochs

    traces = make_epoch_history_traces(epoch_history_path, showlegend = True) # cria os traces da figura

    for trace in traces: 
        fig.add_trace(trace) # adiciona os traces à figura

    fig.update_layout(
        title="Histórico de Epochs", # especifica o título
        xaxis_title="Epochs", # especifica o título do eixo x
        yaxis_title="Metricas", # especifica o título do eixo y
        legend_title="Métricas: ", # especifica o título das legendas
        font=dict(size=18) # especifica o tamanho da fonte
    ) 
    
    fig.write_html(model_path+'epoch_history.html')


def make_species_proportion_bar_plot(filepath, species_proportion_filename): 
    '''
    Adicionar docstring
    '''
    
    label_map = {
        'canario_do_amazonas': 'Canário do Amazonas', 
        'chupim': 'Chupim', 
        'rolinha':'Rolinha', 
        'sanhaco_da_amazonia':'Sanhaço da Amazônia', 
        'sanhaco_do_coqueiro':'Sanhaço do Coqueiro'
    }
    
    fig = go.Figure()

    # abre o arquivo com a proporção de anotações do conjunto parcial 
    df = pd.read_csv(filepath)

    # reordena as colunas em ordem alfabética
    df = df.reindex(sorted(df.columns), axis=1)
    
    for species in df:
        # mapeia a label para  o nome da espécie
        fig.add_trace(
            go.Bar(
                x = [label_map[species]], 
                y = df[species], # especifica o total de anotações
                name = species, 
                text = df[species], # especifica o texto exibindo o total de anotações
                showlegend= False, 
                #textfont = go.bar.Textfont(color='black', size=18),
                textfont=dict(
                        size=24,
                        color="black"
                ),
                textposition = 'outside',
                marker_color = label_colors[species], 
                hovertemplate = f'<b>{species}</b><br>'+'Anotrações: %{y}<extra></extra>'
            )
        )

    fig.update_layout(
        height=900,
        #title = "Total de anotação do subconjunto",
        #title_font = dict(size=30)
        #plot_bgcolor = '#FFFFFF'
    )

    fig.update_xaxes(
        tickangle=45,
        tickfont = dict(size=24)
    )

    fig.update_yaxes(
        title = "Anotações",
        titlefont = dict(size=24),
        tickfont = dict(size=24)
    )  

    fig.write_image(species_proportion_filename, format='png', width=1280, height=900)


def visualize_epoch_history():

    #gráfico com plotly

    hovertemplate = 'epoch: %{x} <br>metric: %{y:.4f} <extra></extra>'
    hovernames = ['Train Loss', 'Valid Loss', 'COCOMetric']
    labels = {'value':'metric', 'variable':'metrics:'}

    fig_px = px.line(epoch_history, 
        x='epochs', 
        y=['train_loss', 'valid_loss', 'COCOMetric'],
        color_discrete_map = {'train_loss':'#636efa', 'valid_loss': '#EF553B', 'COCOMetric':'#00cc96'},
        labels = labels)

    for i, name in enumerate(hovernames):
        fig_px.data[i].update(hovertemplate = f'<b>{name}</b><br><br>' + hovertemplate)

    fig_px.update_layout(
        hovermode='x'
    )

    #salvando figuras e hitórico de epochas

    if not os.path.isdir(path_dir):
        os.mkdir(path_dir)
        
    filename = path_dir + 'epoch_history'
    fig_px.write_image(filename+'.png')
    fig_px.write_html(filename+'.html')
    epoch_history.to_csv(filename+'.csv', index=False)
