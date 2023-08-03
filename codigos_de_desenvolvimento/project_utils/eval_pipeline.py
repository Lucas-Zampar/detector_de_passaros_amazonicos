from icevision.all import *
from icevision.models import *

import pandas as pd
import fiftyone as fo
import plotly.express as px 

def eval_fo_dataset(fo_dataset, 
                    eval_key = None,
                    iou = 0.5, 
                    classwise=True, 
                    compute_mAP = False, 
                    iou_threshs = None):
    
    '''
    Função responsável por avaliar o desempenho do modelo com base nas perdições armazenadas no dataset fiftyone.

    Parâmetros
    ----------
        fo_dataset: dataset fiftyone com as predições do modelo
        eval_key: chave empregada para acessar dados de desempenho (como TP, FP e FN)
        iou: threshould de intersection over union (IoU) (por padrão é 0.5) 
        classwise: se a avaliação irá considerar detecções apenas entre objetos da mesma classe
        compute_map: se será computada a mAP do modelo 
        iou_threshs: uma lista de threshoulds de IoU que podem ser utilizadas no cálculo da mAP
    
    Retorno
    ----------
        results: objeto por meio do qual os resultados das avaliações podem ser acessados
    '''

    # avalia as predições com base no método coco a partir dos valores de ground truth

    results = fo_dataset.evaluate_detections(
        pred_field = "prediction", # campo do dataset referente às predições
        gt_field = "ground_truth", # campo do dataset referente aos valores ground truth
        method = "coco", # método empregado para a valiação será do dataset COCO 
        iou = iou, 
        eval_key = eval_key, 
        classwise = classwise,
        compute_mAP = compute_mAP,
        iou_threshs = iou_threshs
    )

    return results

def make_confusion_matrix(results, path_to_save):
    '''
    Constrói o gráfico da matriz de confusão

    Parâmetros
    ----------
        results: resultados da avaliação do dataset 
        path_to_save: caminho do diretório onde a figura do gráfico será salva
    '''
        
    # armazena os dados da matriz de confusão
    #cm  = results._confusion_matrix()[0][:-1,:] # só é selecionado até a penúltima linha de modo a excluir a linha background
    # passa a selecionar todas as linhas, não ignorando a de background
    cm  = results._confusion_matrix()[0] 

    # as labels das espécies, ordenadas em ordem alfabética
    #gt_species = ['Canário do Amazonas', 'Chupim', 'Rolinha', 'Sanhaço da Amazônia', 'Sanhaço do Coqueiro']
    # adiciona a label background, fazendo referência aos objetos que não foram detectados
    #pred_species = gt_species + ['Background']
    
    labels = ['canário-do-amazonas', 'chupim', 'rolinha', 'sanhaço-da-amazônia', 'sanhaço-do-coqueiro', 'background']

    # juntas, predições e background totalizam a quantidade de anotações ground truth para cada classe 

    # altura e largura da imagem
    h, w = 1200, 1200
    
    # constroi o gráfico da matriz de confusão 
    fig = px.imshow(cm, 
        x=labels, 
        y=labels, 
        color_continuous_scale='Blues', # emprega a escala de cores Blues 
        text_auto=True) # ajusta o texto automaticamente no gráfico

    # algumas atualizações no estilo do gráfico são feitas 
    fig.update_layout(
        height = h,
        width = w, 
        coloraxis_showscale=False, # não mostra a barra de escala de cores 
        font_family="Courier New", # altera a fonte da família
        font_size = 28, # aumenta a fonte
        font_color = 'black' # altera a cor da fonte para preto
    )

    # a figura é então salva em um arquivo PNG
    fig.write_image(path_to_save+'confusion_matrix.png', format='png', width=w, height=h)

def make_metrics_per_class(fo_dataset, results, path_to_save):
    '''
    Salva as métricas predicion, recall e f1-score, bem como o total de TP, FP e FN 
    de cada classe de pássaro.

    Parâmetros
    ----------
        fo_dataset: dataset fiftyone com as predições do modelo
        results: resultados da avaliação do dataset (espera-se uma avaliação com os parâmetros padrões)
        path_to_save: caminho do diretório onde o arquivo CSV será salvo
    '''
    
    # armazena as classes do dataset
    classes = results.classes

    TP = {classe:0 for classe in classes} # dict para contar TP por classe
    FP = {classe:0 for classe in classes} # dict para contar FP por classe

    for sample in fo_dataset: 
        # itera sobre as amostras do fo_dataset
        for detection in sample['prediction']['detections']: 
            # itera sobre as predições da amostra
            label = detection['label']

            if detection['eval'] == 'tp': 
                TP[label] += 1  # adiciona um TP para a classe, se a predição for TP 
            if detection['eval'] == 'fp':
                FP[label] += 1 # adiciona um FP para a classe, se a predição for FP

    # conta o total de anotações ground truth
    count_gt = fo_dataset.count_values("ground_truth.detections.label")

    # conta o total de FN por classe(FN = GT - TP)
    FN = {classe:count_gt[classe] - TP[classe] for classe in classes}

    # dicionário contendo as métricas atuais (precison, reall e f1-score) por classe 
    metrics_per_class = results.report(classes=classes)
    
    # transforma dicionários de TP, FP e FN em DataFrames
    tp_df = pd.DataFrame(TP.values(), columns=['TP'], index=TP.keys())
    fp_df = pd.DataFrame(FP.values(), columns=['FP'], index=FP.keys())
    fn_df = pd.DataFrame(FN.values(), columns=['FN'], index=FN.keys())

    # transforma dicionário de métricas em dataframe 
    metrics_per_class = pd.DataFrame(metrics_per_class)

    # transpõe dataframe
    metrics_per_class = metrics_per_class.T 
    
    # elimina colunas e linhas desnecessárias
    metrics_per_class = metrics_per_class.drop(['support'], axis=1)
    metrics_per_class = metrics_per_class.drop(['micro avg', 'macro avg', 'weighted avg'], axis=0)

    # concatena com os colunas de TP, FP e FN
    metrics_per_class = pd.concat([metrics_per_class, tp_df, fp_df, fn_df], axis=1)

    # salva o dataframe em um arquivo CSV
    metrics_per_class.to_csv(path_to_save+'metrics_per_class.csv')

    # salva o dataframe em um arquivo latex
    metrics_per_class.to_latex(path_to_save+'metrics_per_class_latex.txt')

def make_dataset_final_metrics(fo_dataset, results, results_mAP, path_to_save):
    '''
    Cria um arquivo CSV contendo as métricas finais do modelo (precison, recall, f1-score, mAP, mAP50 e mAP75), 
    bem como total de TP, FP e FN
    
    Parâmetros
    ----------
        fo_dataset: dataset fiftyone com as predições do modelo
        results: resultados da avaliação do dataset (espera-se uma avaliação com os parâmetros padrões)
        results_mAP: resultados da avaliação do dataset (espera-se uma avaliação com os parâmetros específicos para o cálculo das mAPs)
        path_to_save: caminho do diretório onde o arquivo CSV será salvo
    '''

    # dicionária com as métricas (precision, recall e f1-score) computadas
    final_metrics = results.metrics(average='macro')
    
    for AP, value in results_mAP.items(): 
        # adiciona as métricas de mAP ao dicionário
        final_metrics[AP] = value

    # adiciona também o total de TP, FP, TN
    final_metrics['TP'] = fo_dataset.sum("eval_tp")
    final_metrics['FP'] = fo_dataset.sum("eval_fp")
    final_metrics['FN'] = fo_dataset.sum("eval_fn")

    # transforma dicionário de métricas finais em DataFrame a fim de ser salvo
    final_metrics = pd.DataFrame(final_metrics, index=[0])
    final_metrics.to_csv(path_to_save + 'final_metrics.csv', index=False) 


def apply_eval_pipeline(fo_dataset, path_to_save):
    # avalia com os parâmetros padrões
    # cria a chave eval contendo TP, FP e FN
    results = eval_fo_dataset(fo_dataset, eval_key='eval')
    
    # avalia com os parâmetros específicos da matriz de confusão
    # no caso, classwise = False indica que detecções entre classes de objetos serão aceitas
    # com isso, é possível visualizar entre quais espécies o modelo está se confundindo de fato
    results_cm = eval_fo_dataset(fo_dataset, classwise=False)

    # avalia as mAP de acordo com os critérios do COCO
    # mAP: média das áreas sob a cura PR considerandos thresholds de IoU entre .5 e .95 em passos de .05
    # mAP@50: média da área sob a curva PR considerando threshold de IoU de .50
    # mAP@75: média da área sob a curva PR considerando threshold de IoU de .50

    results_AP = eval_fo_dataset(fo_dataset, compute_mAP=True).mAP() 
    results_AP50 = eval_fo_dataset(fo_dataset, compute_mAP=True, iou_threshs=[.50]).mAP()
    results_AP75 = eval_fo_dataset(fo_dataset, compute_mAP=True, iou_threshs=[.75]).mAP()
    
    # transforma resultados das mAP em dict
    results_mAP = {'AP': results_AP, 'AP50': results_AP50, 'AP75': results_AP75}

    # cria um arquivo csv contendo as métricas por classe
    make_metrics_per_class(fo_dataset, results, path_to_save)
    
    # cria a matriz confusão do modelo
    make_confusion_matrix(results_cm, path_to_save)
    
    # gera o gráfico da curva PR (precision-recall) por classe
    #pr_curve_per_class = results_mAP.plot_pr_curves(classes=classes)
    #pr_curve_per_class.save(path_to_save + 'pr_curve_per_class.html')

    # gera o arquivo com as métricas finais do modelo
    make_dataset_final_metrics(fo_dataset, results, results_mAP, path_to_save)
