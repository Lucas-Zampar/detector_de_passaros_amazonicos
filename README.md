
![236798_Happy birds eating seeds inside a feeder_xl-1024-v1-0 (1)](https://github.com/Lucas-Zampar/detector_de_passaros_amazonicos/assets/75434421/5d79e9f7-30e7-440b-855e-41657af449be)

# Detecção de Pássaros Amazônicos (TCC)

Neste repositório, você encontrará a implementação do meu trabalho de conclusão de curso (TCC).  

O trabalho teve como  objetivo detectar espécies de pássaros amazônicos que frequentam comedouros residenciais através de uma abordagem baseada em Deep Learning. Para tanto, foi levantado um conjunto de 940 imagens anotadas para a tarefa de detecção de objetos. Em seguida, a base de dados foi empregada para treinar diferentes configurações do modelo Faster R-CNN através do _framework_ [IceVision](https://github.com/airctic/icevision). Abaixo, é possível verificar a aplicação do modelo produzido: 

https://github.com/Lucas-Zampar/detector_de_passaros_amazonicos/assets/75434421/e58b7a8e-edac-4e0e-ad77-226130b5126d


# Proposta 

## Conjunto de Dados

De modo a capturar as imagens dos pássaros, o estudo teve acesso ao comedouro de uma residência no estado do Amapá. Os moradores locais utilizam o comedouro para observar os pássaros se alimentando das sementes e frutas ali colocadas. O comedouro pode ser visualizado a seguir:

![comedouro](https://github.com/Lucas-Zampar/detector_de_passaros_amazonicos/assets/75434421/a0c60cdd-c422-49c0-9599-b37959cf570e)

No comedouro, foram instaladas três webcams Logitech C270 HD para registrar os pássaros se alimentando. Elas eram conectadas a um notebook que ficava do lado do comedouro. Então, era executado um [script para capturar as gravações](https://github.com/Lucas-Zampar/detector_de_passaros_amazonicos/blob/main/dataset/dataset_utils/script_opencv.py) com auxílio da biblioteca OpenCV. O esquema geral de aquisição dos dados é exposto abaixo:

![aquisicao_de_dados](https://github.com/Lucas-Zampar/detector_de_passaros_amazonicos/assets/75434421/1304c715-bcda-47d3-848e-c604306c0b06)

As gravações foram realizadas entre os dias 07/09/2022 e 13/07/2022. Em seguida, elas foram recortadas manualmente para remover momentos com ausência de pássaros. Os recortes realizados foram então organizados com base na data de gravação e na espécie predominante. Por fim, _frames_ foram extraídos dos recortes para compor as imagens do conjunto de dados. 

De modo a facilitar a extração dos _frames_, foi desenvolvida uma [aplicação em streamlit](https://github.com/Lucas-Zampar/detector_de_passaros_amazonicos/tree/main/streamlit_app). Por meio dela, é possível navegar entre os recortes, além de filtrá-los pela data e pela espécie. Os _frames_ podem ser selecionados aleatoriamente ou em momentos específicos também. A interface da aplicação é demonstrada abaixo: 

https://github.com/Lucas-Zampar/detector_de_passaros_amazonicos/assets/75434421/f92f70fb-00c9-42a9-be28-88ab50c10324

Após serem extraídos, os _frames_ foram carregados na plataforma [RoboFlow](https://roboflow.com/) que conta com ferramentas de anotação de imagens para tarefas de visão compuatacional. Nesse contexto, o processo de anotação ocorreu ao se desenhar as _bounding boxes_ (caixas delimitadoras) sobre os pássaros a fim de localizá-los. Além disso, foram especificadas também as espécies de cada pássaro de modo a classificá-las.  


As espécies dos pássaros foram determinadas por meio de consultas realizadas nas plataformas de ciência cidadã [WikiAves](https://www.wikiaves.com.br/) e [eBird](https://ebird.org/home). Assim, foi possível identificar cinco espécies conhecidas popularmente pelos nomes de canário-do-amazonas, chupim, rolinha, sanhaço-do-coqueiro e sanhaço-da-amazônia. Com isso, foi possível levantar e disponibilizar publicamente um [conjnunto de dados](https://github.com/Lucas-Zampar/detector_de_passaros_amazonicos/tree/main/dataset) composto por 940 imagens e 1.836 anotações no formato Pascal VOC. A distribuição das anotações por espécie pode ser visualizada abaixo:

![total_proportion](https://github.com/Lucas-Zampar/detector_de_passaros_amazonicos/assets/75434421/f42e5a3e-6c5f-43f4-a7ac-c95327cf5c6d)

## Treinamento 

Entre os modelos de deep learning voltados para a detecção de objetos, é possível encontrar o [Faster R-CNN](https://arxiv.org/abs/1506.01497) que recai na categoria de detectores de dois estágios. Nessa categoria, o modelo primeiro propõe regiões com possíveis objetos denominadas de RoI (Region of Interest). Em seguida, ele utiliza essas regiões para realizar as detecções. Em geral, detectores de dois estágios tendem a ser mais precisos. Por essa razão, foi decidido utilizar o Faster R-CNN neste trabalho. Além disso, ele foi um dos primeiros detectores bem-sucedidos a empregar redes neurais convolucionais. 

Neste trabalho, houve duas fases consecutivas de treinamento denominadas respectivamente de 

- __fase preliminar__: na qual foi definida uma _baselina_, bem com uma configuração de treinamento ideal. 
- __fase final__: um único modelo definitivo foi treinado com a configuração definida, sendo comparado com a _baseline_.

Além disso, cada fase utilizou um conjunto de dados diferentes:

- [conjunto parcial](https://github.com/Lucas-Zampar/detector_de_passaros_amazonicos/tree/main/dataset/partial_dataset): composto por 30% dos dados levantados (282 imagens e 560 anotações)
- [conjunto total](https://github.com/Lucas-Zampar/detector_de_passaros_amazonicos/tree/main/dataset/total_dataset): composto pela totalidade dos dados levantados (940 imagens e 1.836 anotações)

A relação entre os dois conjuntos pode ser visualizada no diagrama abaixo:

![Conjunto total](https://github.com/Lucas-Zampar/detector_de_passaros_amazonicos/assets/75434421/92e0beb9-9af2-4a06-9116-7a079b910e7a)

Durante a fase preliminar, foram experimentadas diferentes configurações de treinamento utilizando o [conjunto parcial](https://github.com/Lucas-Zampar/detector_de_passaros_amazonicos/tree/main/dataset/partial_dataset). Nesse contexto, a configuração que mais se destacou foi a seguinte:

|  Hiperparâmetros                          | Valores                  |
|-------------------------------------------|--------------------------|
| Backbone                                  | ResNeXt101 32x4d FPN 1x  |
| Número de epochs                          | 20                       |
| Embaralhamento do conjunto de treinamento | Não                      |
| Taxa de aprendizagem                      | 10-4                     |
| Tamanho do batch                          | 1                        |
| Tamanho da imagem                         | 896x896                  |
| Tamanho de redimensionamento no presizing | 1024x1024                |

A partir disso, foi decido utilizar essa configuração para realizar o treinamento de dois modelos. O primeiro foi treinado com o [conjunto parcial](https://github.com/Lucas-Zampar/detector_de_passaros_amazonicos/tree/main/dataset/partial_dataset) a fim de definr uma _baseline_. Já o segundo, chamado de __definitivo__, foi treinado com [conjunto total](https://github.com/Lucas-Zampar/detector_de_passaros_amazonicos/assets/75434421/92e0beb9-9af2-4a06-9116-7a079b910e7). A avaliação dos modelos foi realizada pelo _framework_ [FiftyOne](https://github.com/voxel51/fiftyone) utilizando a métrica _mean Average Precision_ (mAP) conforme calculada pelo conjunto [COCO](https://cocodataset.org/#detection-eval).

A tabela abaixo compara os resultados alcançados pelos dois modelos:

|       Modelo      |    mAP  | mAP@.50  | mAP@.75  |
|:-----------------:|:-------:|:--------:|:--------:|
| Baseline          | 0,7529  | 0,9459   | 0,8851   |
| Modelo Definitivo | 0,8189  | 0,9833   | 0,9572   |

Nota-se que o fornecimento de mais dados de treinamento beneficiou o modelo definitivo já que a mAP registrou um crescimento percentual de 8.77%. Vale destacar que as demais métricas mAP@.50 e mAP@.75, que utilizam os _thresholds_ de IoU em 50% e 75% respectivamente, foram utilizadas como referência apenas. 


