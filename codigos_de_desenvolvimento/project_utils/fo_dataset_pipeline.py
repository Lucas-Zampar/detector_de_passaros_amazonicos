from icevision.all import *
from icevision.models import *
from .training_pipeline import *

import pandas as pd
import fiftyone as fo

def get_fo_dataset_name(model_path):
    '''
    Função responsável por retornar o nome do dataset fiftyone.

    Para tanto, recebe uma string que representa o caminho até a pasta do modelo no formato: ../TIPO_DE_TESTE/ARQUITETURA/MODEL_ID/. 
    A partir dela, o nome do dataset fiftyone será: TIPO_DE_TESTE_ARQUITERUA_MODEL_ID.

    Parâmetros
    ----------
        model_path : caminho até a pasta do modelo 
    
    Retorno
    ----------
        String contendo o nome do dataset fiftyone.
    '''

    return '_'.join(model_path.split('/')[-4:-1])


def get_model_from_checkpoint(checkpoint_path):
    '''
    Função responsável por instanciar o modelo treinado a partir do seu checkpoint, contido em um arquivo .pth.

    Além disso, retorna algumas dados de treinamento como a especificação do tipo de modelo, o tamanho das imagens de treinamento 
    e o mapa de classes do conjunto de dados, de modo que eles possam ser empregados para a criação do dataset FiftyOne.


    Parâmetros
    ----------
        checkpoint_path : caminho para arquivo .pth contendo o checkpoint do modelo.
    
    Retorno
    ----------
        model_type : especificação da IceVision sobre o tipo do modelo
        model : instância do modelo treinado
        img_size : tamanho de imagem empregada no treinamento 
        class_map : mapa de classes do conjunto com o qual o modelo foi treiando
    '''
     
    # a partir do caminho do checkpoint do modelo, carrega os dados
    checkpoint_and_model = model_from_checkpoint(checkpoint_path)

    # recupera a especificação do tipo de modelo empregado
    model_type = checkpoint_and_model["model_type"]

    # recupera o modelo com os parâmetros treinados
    model = checkpoint_and_model["model"]
    
    # recupera o tamanho da imagem empregada para o treinamento do modelo
    img_size = checkpoint_and_model["img_size"]

    # recupera o mapa de classes 
    class_map = checkpoint_and_model["class_map"]

    return model_type, model, img_size, class_map

def make_sorted_predictions_by_loss(model_type, model, infer_ds, sort_by='loss_total'):
    '''
    Função responsável por realizar as predições ordenadas pela loss. 

    Para tanto, emprega o método plot_top_losses que recebe como argumentos a instância do modelo treinado,
    o dataset de inferência e o tipo de loss que se deve empregar para ordenar as predições. O tipo de loss 
    pode variar de acordo com o modelo.

    Parâmetros
    ----------
        model_type : especificação da IceVision sobre o tipo do modelo
        model: instância do modelo treinado
        infer_ds: o dataset de inferência 
        sort_by: especifica qual loss deve ser empregada a fim de ordenar as predições

    Retorno
    ----------
        sorted_samples: amostras ordenadas
        sorted_preds: predições ordenadas
        stats: estatísticas gerais das predições
    '''

    # faz predições ordenando-as pela loss (podem ser várias, por padrão será a total)
    return model_type.interp.plot_top_losses(model=model,dataset=infer_ds, sort_by=sort_by)

def save_loss_per_sample(sorted_samples, path_dir):
    '''
    Função responsável por salvar as losses de cada imagem em um arquvio CSV, de modo 
    que elas possam ser visualizadas futuramente para interpretação do modelo. 

    Parâmetros
    ----------
        sorted_samples: as amostras ordenadas por loss 
        path_dir: caminho onde o arquivo csv será salvo
        
    Retorna
    ----------
        sample_losses_df: dataframe contendo as losses de cada imagem
    '''

    # retorna uma lista de dicionários contendo as losses e o caminho da amostra
    sample_losses = get_samples_losses(sorted_samples)
    
    # para cada dicionário 
    for sample_loss in sample_losses:
        
        # recupera o objeto Posix referente ao caminho da amostra
        filepath = str(sample_loss['filepath'])
        
        # recupera somento o nome do arquivo da amostra
        filename = filepath.split('/')[-1]

        # remove a chave filepath
        sample_loss.pop('filepath')
        
        # adiciona o nome do arquivo somente com a chave filename
        sample_loss['filename'] = filename
    
    
    # salva as losses de cada amostra em um dataframe
    sample_losses_df = pd.DataFrame(sample_losses)
    
    # inverte a ordem das colunas, já que a coluna filename e loss_total se encontram no final
    sample_losses_df = sample_losses_df[sample_losses_df.columns[::-1]]

    # salva o dataframe como um arquivo csv
    sample_losses_df.to_csv(path_dir+'losses_per_sample.csv', index=False)
    
    return sample_losses_df


def get_sample_losses():
    pass


def make_fo_dataset_from_icevision(preds, dataset_name, tfms, sample_tags=None):
    '''
    Função responsável por criar um dataset fiftyone a partir de um dataset da IceVision

    Para cada amostra, adicionar também a loss daquela amostra.
    
    Parâmetros
    ----------
        preds: predições realizadas pelo modelo
        dataset_name: nome atribuído ao dataset do fiftyone
        tfms: as transformações aplicadas sobre as imagens de inferência
        sample_tags: informações adicionais que podem ser inseridas nas amostras (imagens)
    
    Retorna
    ----------
        fo_dataset: um dataset do fiftyone
    '''

    # persistent = True mantem o dataset no disco
    
    fo_dataset = data.create_fo_dataset(
        detections=preds, 
        dataset_name= dataset_name, 
        transformations= tfms.tfms_list,
        persistent = True, # o dataset é persistente, isto é, fica salvo depois que a app é reninicializada
        exist_ok=False) # não adiciona imagens se o dataset já existir

    # tas tags adicionadas nas amostras são as losses delas. 
    if sample_tags is not None:       
       # deve ser um dataframe a partir do qual todas as colunas serão acessadas
        for sample, row in zip(fo_dataset, sample_tags.iterrows()):
            sample['tags'] = [f'{key}:{value}'for key, value in row[1].to_dict().items()]
            sample.save()
        fo_dataset.save()

    return fo_dataset


def apply_fo_dataset_pipeline(model_path, dataset_name, dataset_type='partial'):
    '''
    Função responsável por criar o dataset fiftyone por meio do qual é possível visualizar as predições, 
    além de já avaliar o modelo com base nestas predições.
    
    A ordem das amostras no dataset se baseia na loss total das amostras. Assim, as primeiras
    apresentam as maiores losses (indicando confusão do modelo).
    
    Parâmetros
    ----------
        model_path: path onde se encontram os arquivos necessários para construir o dataset fiftyone
        dataset_name: nome do dataset fiftyone
        dataset_type: se o conjunto dos dados se refere ao parcial ou completo 
    '''

    # se o dataset já existir, interrompa a execução da função neste ponto
    if dataset_name in fo.list_datasets(): return
    
    # encontra o arquivo de checkpoint do modelo 
    for file in os.listdir(model_path):
        if '.pth' in file: 
            checkpoint_filename = file
    
    # define o caminho para o arquivo de checkpoint
    checkpoint_path = model_path + checkpoint_filename
    
    # carregando o modelo
    model_type, model, img_size, _ = get_model_from_checkpoint(checkpoint_path)

    # parse do conjunto de dados
    train_records, valid_records, class_map = parse_data(dataset_type=dataset_type)

    # transformações para os dados de inferência
    valid_tfms = get_tfms(img_size, tfms_type='valid')

    # dataset de inferência
    infer_ds = get_dataset(valid_records, valid_tfms)
    
    # predições ordenadas por loss
    sorted_samples, sorted_preds, stats = make_sorted_predictions_by_loss(
        model_type = model_type,
        model = model, 
        infer_ds = infer_ds
    )
    
    # salvando losses de cada amostra
    sample_losses_df = save_loss_per_sample(
        path_dir = model_path,                                     
        sorted_samples = sorted_samples
    )
    
    # cria o dataset
    make_fo_dataset_from_icevision(
        preds=sorted_preds,
        dataset_name = dataset_name,
        tfms = valid_tfms,
        sample_tags = sample_losses_df
    )