import os 

from icevision.all import *
from icevision.models import *

import pandas as pd
import fiftyone as fo
import plotly.express as px

## dicionário com as cores padrões das bounding boxes dos pássaros

class Hyperparameters():
    '''
    Classe responsável por armazenar os hiperparâmetros do modelo
    por meio de seus atributos 

    Atributos
    ----------
        model_name : nome válido do modelo disponível pela IceVision
        backbone_name : nome válido da backbone disponível para o modelo 
        learning_rate : taxa de aprendizagem
        num_epochs : número de epochs do treinamento
        train_shuffle : se haverá embaralhamento ou não do conjunto em cada epoch
        batch_size : tamanho do batch
        img_size : tamanho para o qual as imagens serão redimensionadas
        presize : tamanho para o qual as imagens serão redimensionadas no presizing
    '''

    def __init__(self, 
                 model_name, 
                 backbone_name, 
                 learning_rate,
                 num_epochs, 
                 train_shuffle,
                 batch_size, 
                 img_size,
                 presize):

        self.model_name = model_name
        self.backbone_name = backbone_name
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.train_shuffle = train_shuffle
        self.batch_size = batch_size
        self.img_size = img_size
        self.presize = presize

def parse_data(dataset_type='partial'):
    '''
    Função responsável por realizar o parseamento dos dados, além de dividir o conjunto em treinamento e validação.

    Inicialmente, os caminhos das imagens e anotações do dataset são repassadas para a classe VOCBBoxParser. 
    A IceVision possui as classes COCOBBoxParser e VOCBBoxParser que interpretam anotações nos formatos dos datasets COCO e VOC Pascal respectivamente.
    Além disso, é possível criar também Parsers customizados com base no tipo de anotação do conjunto de dados.

    Em seguida, o dataset é dividido por meio da classe RandomSplitter na proporção 70-30 (train/valid) 
    sempre da mesma forma devido a seed estática.

    Por fim, são geradas instâncias da classe RecordCollection, que representam o conjunto de treinamento e de validação respectivamente.
    Nessas coleções, encontram-se instâncias da classe BaseRecord responsável por associar as imagens com suas respectivas anotações.
    Assim, é possível acessar o caminho da imagem, o tamanho original da imagem, as classes na imagem, bem como as respectivas bounding boxes (BBox).

    URL para implementação da classe RecordCollections: https://github.com/airctic/icevision/blob/master/icevision/data/record_collection.py
    URL para implementação da classe BaseRecord: https://github.com/airctic/icevision/blob/master/icevision/core/record.py

    Parâmetros
    ----------
        dataset_type : especifica qual o tipo de dataset a ser carregado (partial para o conjunto parcial ou total para o conjunto completo)
    
    Retorna
    ----------
        train_records : records de treinamento
        valid_records : records de validação
        class_maps : mapa de classes de pássaros do dataset
    '''

    # se o conjunto for o parcial, busque nestes caminhos
    if dataset_type == 'partial':
        images_dir = '../dataset/partial_dataset'
        annotations_dir = '../dataset/partial_dataset'
    
    # se conjunto for o completo, busque nestes caminhos
    if dataset_type == 'total':
        images_dir = '../dataset/total_dataset'
        annotations_dir = '../dataset/total_dataset'
    
    # parseamento dos dados com anotações no formato VOC Pascal
    parser = parsers.VOCBBoxParser(annotations_dir=annotations_dir, images_dir=images_dir)

    # divisão do dataset na proporção 70-30 (train/valid)
    data_splitter = RandomSplitter([0.7, 0.3], seed=42) 

    # armazena os records para treinamento e validação
    train_records, valid_records = parser.parse(data_splitter = data_splitter)

    # gera o mapa de classes, especificando as classes de pássaros do dataset
    class_map = parser.class_map.get_classes()

    return train_records, valid_records, class_map

def get_tfms(img_size, presize=None, tfms_type ='valid'):
    '''
    Função responsável por especificir os tipos de transformações 
    aplicadas sobre as imagens para o redimensionamento ou aumento de dados.
    
    Vale ressaltar que a IceVision emprega a biblioteca Albumentations para realizar as transformações.
    Além disso, as transformações são aplicadas on-the-fly, isto é, elas existem em tempo de execução apenas, 
    não havendo a criação de imagens estáticas transformadas. 

    Caso as transformações sejam referentes aos dados de treinamento, é empregada a função tfms.A.aug_tfms 
    que retorna uma lista de transformações da biblioteca Albumentations pré-definidas pela IceVision a saber: 

        - HorizontalFlip : https://albumentations.ai/docs/api_reference/full_reference/#albumentations.augmentations.geometric.transforms.HorizontalFlip
        - ShiftScaleRotate : https://albumentations.ai/docs/api_reference/full_reference/#albumentations.augmentations.geometric.transforms.ShiftScaleRotate        
        - RGBShift : https://albumentations.ai/docs/api_reference/full_reference/#albumentations.augmentations.transforms.RGBShift
        - RandomBrightnessContrast : https://albumentations.ai/docs/api_reference/full_reference/#albumentations.augmentations.transforms.RandomBrightnessContrast
        - Blur : https://albumentations.ai/docs/api_reference/full_reference/#albumentations.augmentations.blur.transforms.Blur
        - RandomSizedBBoxSafeCrop: https://albumentations.ai/docs/api_reference/full_reference/#albumentations.augmentations.crops.transforms.RandomSizedBBoxSafeCrop
        - PadIfNeeded : https://albumentations.ai/docs/api_reference/full_reference/#albumentations.augmentations.geometric.transforms.PadIfNeeded
    
    Outro ponto importante a ser ressaltado é que as transformações são aplicadas aleatoriamente. Dessa forma, nem sempre todas 
    as transformações especificadas serão aplicadas, além disso os valores que as especificam pode variar.

    Caso as transformações sejam referentes aos dados de validação, é empregada a função tfms.A.resize_and_pad que apenas
    especifica transformações de redimensionamento (com padding), a saber:   
        - LongestMaxSize : https://albumentations.ai/docs/api_reference/full_reference/#albumentations.augmentations.geometric.resize.LongestMaxSize
        - PadIfNeeded : https://albumentations.ai/docs/api_reference/full_reference/#albumentations.augmentations.geometric.transforms.PadIfNeeded

    Eem ambos os casos, é realizada a normalização das imagens por meio da classe Normalize da bilioteca Albumentations.
    A normalização é dada pela fórmula img = (img - mean * max_pixel_value) / (std * max_pixel_value). 
        - Normalize : https://albumentations.ai/docs/api_reference/full_reference/#albumentations.augmentations.transforms.Normalize

    Parâmetros
    ----------
        img_size : tamanho da imagem a ser redimensionada
        presize : tamanho do presizing (técnica introduzida pela biblioteca Fastai)
        tfms_type : tipo de transformação (train para conjunto de treinamento ou valid para conjunto de validação)
    
    Retorna
    ----------
        Uma lista com transformações especificadas
    '''

    # se os dados forem de treinamento
    if tfms_type  == 'train':
        return tfms.A.Adapter([*tfms.A.aug_tfms(size=img_size, presize=presize), tfms.A.Normalize()]) 

    # se os dados forem de validação
    if tfms_type == 'valid':
        return tfms.A.Adapter([*tfms.A.resize_and_pad(img_size), tfms.A.Normalize()])
        
    
def get_dataset(records, tfms):
    '''
    Cria uma instância da classe Dataset, que por sua vez é um container para uma lista de records e transformações.

    Cada vez que um item do dataset é acessado, seu respectivo record é coletado e as transformações especificadas são aplicadas. 
    Um ponto importante a ser destacado é que as transformações só são aplicadas quando a imagem é acessada. Além disso,  
    uma vez que as transformações possuem um fator aleatório, a imagem transformada sempre será ligeiramente diferente. Assim, 
    o modelo será apresentado a uma imagem ligeiramente diferente da original, o que além de aumentar o conjunto de dados, 
    auxilia no processo de aprendizagem.

    URL para implementação da classe Dataset: https://github.com/airctic/icevision/blob/master/icevision/data/dataset.py

    Parâmetros
    ----------
        records : os records dos dados que podem ser de treinamento ou de validação
        tfms : uma lista de transformações a serem aplicadas
        
    Retorna
    ----------
        Uma instância da classe Dataset contendo records e transformações especificadas
    '''

    return Dataset(records, tfms)


def get_model_type(model_name, backbone_name, img_size=None):
    '''
    Função responsável por especifiar o tipo do modelo suportado pela Icevision, 
    bem como sua respectiva rede de backbone.

    Parâmetros
    ----------
        model_name : um nome de modelo válido suportado pela IceVision
        backbone_name : um nome válido de uma rede de backbone suportada pelo modelo selecionado
        img_size : para alguns modelos, como YOLOV5, é necessário especificar também o tamanho da imagem

    Retorna
    ----------
        model_type : especificação do tipo de modelo 
        backbone_type : especificação do tipo de backbone
        extra_args : argumentos extras que alguns modelos necessitam como o tamanho de imagem
    '''

    extra_args = None

    if model_name == 'mmdet.faster_rcnn':
        
        model_type = models.mmdet.faster_rcnn

        if backbone_name == 'resnet50_fpn_1x':
            backbone_type = model_type.backbones.resnet50_fpn_1x(pretrained=True)

        if backbone_name == 'resnet50_fpn_2x':
            backbone_type = model_type.backbones.resnet50_fpn_2x(pretrained=True)
        
        if backbone_name == 'resnet101_fpn_1x':
            backbone_type = model_type.backbones.resnet101_fpn_1x(pretrained=True)
        
        if backbone_name == 'resnet101_fpn_2x':
            backbone_type = model_type.backbones.resnet101_fpn_2x(pretrained=True)

        if backbone_name == 'resnext101_32x4d_fpn_1x':
            backbone_type = model_type.backbones.resnext101_32x4d_fpn_1x(pretrained=True)

        if backbone_name == 'resnext101_32x4d_fpn_2x':
            backbone_type = model_type.backbones.resnext101_32x4d_fpn_2x(pretrained=True)
            
        if backbone_name == 'resnext101_64x4d_fpn_1x':
            backbone_type = model_type.backbones.resnext101_64x4d_fpn_1x(pretrained=True)
    
        if backbone_name == 'resnext101_64x4d_fpn_2x':
            backbone_type = model_type.backbones.resnext101_64x4d_fpn_2x(pretrained=True)
        
    if model_name == 'ultralytics.yolov5':
        # É necessário informar o tamanho da imagem como um argumento extra        
        
        model_type = models.ultralytics.yolov5

        if backbone_name == 'small':
            backbone_type = model_type.backbones.small(pretrained=True)

        if backbone_name == 'medium':
            backbone_type = model_type.backbones.medium(pretrained=True)

        if backbone_name == 'large':
            backbone_type = model_type.backbones.large(pretrained=True)

        if backbone_name == 'extra_large':
            backbone_type = model_type.backbones.extra_large(pretrained=True)

        if backbone_name == 'small_p6':
            backbone_type = model_type.backbones.small_p6(pretrained=True)

        if backbone_name == 'medium_p6':
            backbone_type = model_type.backbones.medium_p6(pretrained=True)

        if backbone_name == 'large_p6':
            backbone_type = model_type.backbones.large_p6(pretrained=True)

        if backbone_name == 'extra_large_p6':
            backbone_type = model_type.backbones.extra_large_p6(pretrained=True)
        
        extra_args = {}
        extra_args['img_size'] = img_size
    

    return model_type, backbone_type, extra_args


def get_model(model_type, backbone_type, num_classes, extra_args=None):
    '''
    Função responsável por instanciar o modelo. 

    Parâmetros
    ----------
        model_type : especificação do tipo de modelo 
        backbone_type : especificação do tipo de backbone
        num_classes : número de classes do dataset
        extra_args : argumentos extras que alguns modelos necessitam como o tamanho de imagem

    Retorna
    ----------
        O modelo que pode variar dependendo da implementação.
    '''

    if extra_args is not None:
        return model_type.model(backbone=backbone_type, num_classes=num_classes, **extra_args)
    else: 
        return model_type.model(backbone=backbone_type, num_classes=num_classes)


def get_dataloader(model_type, 
                   dataset, 
                   batch_size = 1, 
                   train_shuffle=False, 
                   num_workers = 2, 
                   dataloader_type='valid'):
    '''
    Função responsável por criar um data loader, o qual é responsável por obter os itens de um dataset
    e organizá-los em batches no formato específico requerido por cada modelo. Assim, como cada data loader depende
    também do modelo especificado, essa etapa é separada da criação do dataset. 

    Parâmetros
    ----------
        model_type : especificação do tipo de modelo 
        dataset : dataset contendo records e transformações especificadas
        batch_size : tamanho do batch [por padrão será 1] 
        train_shuffle : se haverá embaralhamento dos dados em cada epoch [por padrão será False]
        num_workers : número de workers responsáveis por transferir os dados do disco para GPU
        dataloader_type : se o data loader será de treinamento (train) ou de validação (valid) [por padrão será valid]
    
    Retorno 
    ----------
        Um data loader específico do tipo de modelo selecionado
    '''

    if dataloader_type == 'valid':
        return model_type.valid_dl(dataset, 
                                   batch_size=batch_size, 
                                   num_workers=num_workers, 
                                   shuffle=False)
    
    if dataloader_type == 'train':
        return model_type.train_dl(dataset, 
                                   batch_size=batch_size, 
                                   num_workers=num_workers, 
                                   shuffle=train_shuffle)

def get_bbox_metrics():
    '''
    Função responsável por retornar instância da classe COCOMetric que calcula a métrica de avaliação 
    mAP (mean Avarage Precision), que será empregada para avaliar as predições do modelo no conjunto de validação.
    
    Para tanto, a IceVision emprega a cocoapi: https://github.com/cocodataset/cocoapi
    
    Vale lembrar que o valor da mAP calculada durante o treinamento pode diferir daquela obtida pela biblioteca FiftyOne, 
    já que as transformações de redimensionamento das imagens de inferência podem ser diferentes.    

    Retorno 
    ----------
        Instância da classe COCOMetric responsável por calcular a métrica de avaliação mAP
    '''

    return [COCOMetric(metric_type=COCOMetricType.bbox)]

def get_learner(model_type, dls, model, metrics):
    '''
    Escrever DocString
    '''

    return model_type.fastai.learner(dls=dls, model=model, metrics=metrics)

def get_learning_rate(learner):
    '''
    Escrever DocString
    '''
    return learner.lr_find().valley

def train(learner, num_epochs, learning_rate = None, freeze_epochs=1,  method = 'fine_tune'):
    '''
    Função responsável por realizar o treinamento do modelo.

    Parâmetros
    ----------
        learner : (TO DO) 
        num_epochs: número de epochs
        learning_rate : taxa de aprendizagem empregada
        freeze_epochs: quantidade de epochs nas quais os parâmetros das primeiras camadas não serão ajustados
        method : fine_tune ou fit_one_cycle, porém o último não está implementado ainda
    Retorno 
    ----------
        None
    '''

    if method == 'fine_tune':
        learner.fine_tune(num_epochs, learning_rate, freeze_epochs=freeze_epochs)
    
def save_epoch_history(learner, path_to_save):
    '''
    Função responsável por salvar a progressão das métricas em cada.

    Parâmetros
    ----------
        learner : TO DO
        path_dir : caminho onde os 
    '''

    # armazena o histórico de epochs incluindo valores das losses de treinamento, validação e métrica de mAP
    record_values = learner.recorder.values
    # armazena o histórico em uma tabela na forma de um DataFrame
    epoch_history_df = pd.DataFrame(record_values, columns = ['train_loss', 'valid_loss', 'mAP'])
    # adiciona uma coluna indicando a epoch ao resetar os índices
    epoch_history_df = epoch_history_df.reset_index().rename(columns = {'index':'epochs'})
    epoch_history_df.to_csv(path_to_save+'epoch_history.csv', index=False)
    
def save_hyperparameters(hyperparameters, path_to_save):
    '''
    Função responsável por salvar os hiperparâmetros do modelo em um arquivo CSV.

    Parâmetros
    ----------
        hyperparameters : instância da classe Hyperparameters contendo os hiperparâmetros do modelo
        path_to_save : caminho onde o arquivo CSV será armazenado
        
    Retorno
    ----------
        Sem retorno
    '''

    # armazena os valores dos atributos do objeto em um dicionário
    hyperparameters_dict = hyperparameters.__dict__
    # os valores dos hiperparâmetros são armazenados em um DataFrame Pandas
    hyperparameters_df = pd.DataFrame(hyperparameters_dict, index=[0])
    # os DataFrame é convertido para um arquivo CSV
    hyperparameters_df.to_csv(path_to_save+'hyperparameters.csv', index=False)

def save_model(model, hyperparameters, class_map, path_to_save):
    '''
    Função responsável por salvar o checkpoint do modelo treinado.

    Parâmetros
    ----------
        model : modelo treinado
        hyperparameters : hiperparâmetros de treinamento do modelo 
        class_map : mapa de classes do dataset
        path_to_save : caminho onde o checkpoint será salvo
    
    Retorno
    ----------
        Sem retorno
    '''
    
    # nome do arquivo do checkpoint será composto pelo nome do modelo e da backbobe
    filename = f'{hyperparameters.model_name}_{hyperparameters.backbone_name}.pth'
    # especifica o caminho para salvar o checkpoint do modelo 
    checkpoint_path = path_to_save + filename
    # salva o modelo
    save_icevision_checkpoint(model, 
                        model_name= hyperparameters.model_name, 
                        backbone_name= hyperparameters.backbone_name,
                        img_size= hyperparameters.img_size, 
                        classes = class_map, 
                        filename= checkpoint_path,
                        meta={'icevision_version': '0.12.0'})

def get_path_to_save_model(base_path):
    '''
    Função responsável por criar o caminho da pasta onde o modelo será armazendado. 

    A nomenclatura das pastas segue a seguinte regra: model_id 
    (em que id é um inteiro que identifica o modelo).

    Para tanto, inicialmente são recuperados os nomes das patas empregando a função os.listdir. 

    Em seguida, é especificada uma função lambda que separa o id no nome da pasta 
    e o converte para um valor inteiro. Então, a função é aplica sobre cada nome das pastas 
    por meio de um mapeamento, sendo obtido o maior id. 

    Por fim, o id é incrementado e adicionado no caminho base.

    Parâmetros
    ----------
        base_path : caminho do diretório onde as pastas dos modelos devem se encontrar
        
    Retorno
    ----------
        Uma string que representa o caminho da pasta onde o modelo treinado deve ser armazenado
    '''

    model_folder_names = os.listdir(base_path)
    model_folder_names


    if model_folder_names != []:
        model_id_to_int = lambda folder_name: int(folder_name.split('model_')[-1])
        max_id = max(map(model_id_to_int, model_folder_names)) 
        new_id = max_id + 1
    else:
        new_id = 0

    path_to_save_model = base_path + f'model_{new_id}/'
    os.mkdir(path_to_save_model)    

    return path_to_save_model

