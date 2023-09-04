
![236798_Happy birds eating seeds inside a feeder_xl-1024-v1-0 (1)](https://github.com/Lucas-Zampar/detector_de_passaros_amazonicos/assets/75434421/5d79e9f7-30e7-440b-855e-41657af449be)

# Detector de Pássaros Amazônicos (TCC)

Pássaros atraem a atenção humana por sua beleza e diversidade, o que estimula pessoas a obsersarvá-los como lazer. Nesse processo, é possível registrar pássaros em imagens e compartilhá-las em plataformas de ciência cidadã como [WikiAves](https://www.wikiaves.com.br/) e [eBird](https://ebird.org/home). Com isso, pessoas podem contribuir significativamente com pesquisas científicas que visam compreender e preservar espécies de pássaros. 

A região amazônica pode oferecer uma excelente experiência de observação de pássaros dada a diversidade de espécies existentes. Algumas delas se adaptaram inclusive ao meio urbano, sendo possível encontrar pássaros se alimentando até nos jardins de residências. Dessa forma, os moradores locais podem observar pássaros ao atraí-los por meio de alimentos colocados em comedouros abertos.

Nesse conexto, notou-se a oportunidade de empregar _webcams_ para registrar pássaros amazônicos se alimentando em comedouros residenciais. Além disso, questionou-se se não seria possível empregar inteligência artificial, através de modelos de Deep Learning, para detectar automaticamente as espécies dos pássaros registrados. 

# Objetivos

Assim, esse projeto teve como objetivo geral levantar uma abordagem baseada em Deep Learning para detectar automaticamente espécies de pássaros amazônicos a partir de um contexto residencial. Para tanto, 

- Levantar um conjunto de imagens de pássaros que frequentam um comedouro residencial;
- Anotar as imagens levantadas para a tarefa de detecção de objetos;
- Treinar e avaliar modelos de Deep Learning em diferentes configurações para detectar as espécies desses pássaros; 
- Avaliar os modelos, selecionando o que mais se destacasse;


# Proposta 

## Conjunto de Dados

O estudo teve acesso ao comedouro de uma residência no estado do Amapá que pode ser visualizado na figura abaixo. 

<img src="https://github.com/Lucas-Zampar/detector_de_passaros_amazonicos/assets/75434421/a0c60cdd-c422-49c0-9599-b37959cf570e" alt="comedouro" height=60% width=60%>

No comedouro, foram instaladas três _webcams_ Logitech C270 HD conectadas a um notebook a fim de gravar os pássaros se alimentando, conforme colocado na figura abaixo. A captura das gravações foi realizada por meio de um [script com auxílio da biblioteca OpenCV](https://github.com/Lucas-Zampar/detector_de_passaros_amazonicos/blob/main/dataset/dataset_utils/script_opencv.py).

<img src="https://github.com/Lucas-Zampar/detector_de_passaros_amazonicos/assets/75434421/1304c715-bcda-47d3-848e-c604306c0b06" alt="esquema_gravacao" height=60% width=60%>

As imagens do conjunto de dados foram obtidas a partir de _frames_ extraídos dessas gravações. De modo a facilitar a extração, foi desenvolvida uma [aplicação em Streamlit](https://github.com/Lucas-Zampar/detector_de_passaros_amazonicos/tree/main/streamlit_app) demonstrada no vídeo abaixo. Por meio dela, é possível selecionar _frames_ aleatórios ou específicos das gravações, além de filtrá-las pela data e pela espécie predominante.

https://github.com/Lucas-Zampar/detector_de_passaros_amazonicos/assets/75434421/f92f70fb-00c9-42a9-be28-88ab50c10324

As espécies dos pássaros foram determinadas por meio de consultas realizadas nas plataformas de ciência cidadã [WikiAves](https://www.wikiaves.com.br/) e [eBird](https://ebird.org/home). Assim, foi possível identificar cinco espécies conhecidas popularmente pelos nomes de __canário-do-amazonas__, __chupim__, __rolinha__, __sanhaço-da-amazônia__ e __sanhaço-do-coqueiro__.

As imagens adquiridas foram anotadas por meio da plataforma [RoboFlow](https://roboflow.com/). Ao todo, foram levantadas 940 imagens e 1.836 anotações no formato Pascal VOC. A distribuição das anotações por espécie pode ser visualizada no gráfico abaixo. O conjunto de dados produzido pode ser encontrado na pasta [dataset](https://github.com/Lucas-Zampar/detector_de_passaros_amazonicos/tree/main/dataset). 

<img src="https://github.com/Lucas-Zampar/detector_de_passaros_amazonicos/assets/75434421/278633e3-965a-4707-a144-de9da2c4d431" alt="esquema_gravacao" height=60% width=60%>



## Treinamento 

Entre os algoritmos de Deep Learning voltados para a detecção de objetos, é possível encontrar o [Faster R-CNN](https://arxiv.org/abs/1506.01497) que recai na categoria de detectores de dois estágios. Nessa categoria, o modelo primeiro propõe regiões com possíveis objetos denominadas de RoI (Region of Interest). Em seguida, ele utiliza essas regiões para realizar as detecções. Em geral, detectores de dois estágios tendem a ser mais precisos. Por essa razão, foi decidido utilizar o Faster R-CNN neste trabalho. 

Neste trabalho, houve duas fases consecutivas de treinamento denominadas respectivamente de:

- __fase preliminar__: modelos foram treinados com uma porção menor de dados em diferentes configurações. Nesta fase, foi selecionada a configuração de treinamento do modelo que apresentou os melhores resultados. Esse modelo foi definido também como uma _baseline_. 
- __fase final__: um único modelo definitivo foi treinado com a totalidade dos dados utilizando a configuração de treinamento selecionada anteriormente. Além disso, o modelo definitivo foi comparado com a _baseline_.

O treinamento foi conduzido por meio do _framework_ [IceVision](https://github.com/airctic/icevision) que utiliza os modelos disponibilizados pelo [MMDetection](https://github.com/open-mmlab/mmdetection). A configuração selecionada para treinar tanto a _baseline_, quanto o modelo definitivo pode ser visualizada na tabela abaixo.

|  Hiperparâmetros                          | Valores                  |
|-------------------------------------------|--------------------------|
| Backbone                                  | ResNeXt101 32x4d FPN 1x  |
| Número de epochs                          | 20                       |
| Embaralhamento do conjunto de treinamento | Não                      |
| Taxa de aprendizagem                      | 10-4                     |
| Tamanho do batch                          | 1                        |
| Tamanho da imagem                         | 896x896                  |
| Tamanho de redimensionamento no presizing | 1024x1024                |


A avaliação foi conduzida pelo _framework_ [FiftyOne](https://github.com/voxel51/fiftyone) utilizando as três métricas de _mean Average Precision_ (mAP) definidas pelos critérios de avaliação do conjunto [COCO](https://cocodataset.org/#detection-eval). Dessa forma, os resultados alcançados tanto pela _baselie_, quanto pelo modelo definitivo podem ser visualizados na tabela abaixo. 

|       Modelo      |    mAP  | mAP@.50  | mAP@.75  |
|:-----------------:|:-------:|:--------:|:--------:|
| Baseline          | 0,7529  | 0,9459   | 0,8851   |
| Modelo Definitivo | 0,8189  | 0,9833   | 0,9572   |

Nota-se que o fornecimento de mais dados de treinamento beneficiou o modelo definitivo já que a mAP registrou um crescimento percentual de 8,77%. Vale destacar também que ao se considerar o _thresold_ de _Intersection over Union_ (IoU) em 50% apenas, a mAP alcançada pelo modelo definitivo foi de 98,33%. 

