
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

De modo a capturar as imagens dos pássaros, o estudo teve acesso ao comedouro de uma residência no estado do Amapá. O comedouro pode ser visualizado a seguir:

<img src="https://github.com/Lucas-Zampar/detector_de_passaros_amazonicos/assets/75434421/a0c60cdd-c422-49c0-9599-b37959cf570e" alt="comedouro" height=60% width=60%>

No comedouro, foram instaladas três _webcams_ Logitech C270 HD conectadas a um notebook a fim de registrar os pássaros se alimentando, conforme colocado na figura abaixo. A captura das gravações era feita por meio de um [script que empregou a biblioteca OpenCV](https://github.com/Lucas-Zampar/detector_de_passaros_amazonicos/blob/main/dataset/dataset_utils/script_opencv.py)

<img src="https://github.com/Lucas-Zampar/detector_de_passaros_amazonicos/assets/75434421/1304c715-bcda-47d3-848e-c604306c0b06" alt="comedouro" height=60% width=60%>


As gravações foram realizadas entre os dias 07/09/2022 e 13/07/2022. Em seguida, elas foram recortadas manualmente para remover momentos com ausência de pássaros. Os recortes realizados foram então organizados com base na data de gravação e na espécie predominante. Por fim, _frames_ foram extraídos dos recortes para compor as imagens do conjunto de dados. 

De modo a facilitar a extração dos _frames_, foi desenvolvida uma [aplicação em streamlit](https://github.com/Lucas-Zampar/detector_de_passaros_amazonicos/tree/main/streamlit_app). Por meio dela, é possível navegar entre os recortes, além de filtrá-los pela data e pela espécie. Os _frames_ podem ser selecionados aleatoriamente ou em momentos específicos também. A interface da aplicação é demonstrada abaixo: 

https://github.com/Lucas-Zampar/detector_de_passaros_amazonicos/assets/75434421/f92f70fb-00c9-42a9-be28-88ab50c10324

Após serem extraídos, os _frames_ foram carregados na plataforma [RoboFlow](https://roboflow.com/) que conta com ferramentas de anotação de imagens para tarefas de visão compuatacional. Nesse contexto, o processo de anotação ocorreu ao se desenhar as _bounding boxes_ (caixas delimitadoras) sobre os pássaros a fim de localizá-los. Além disso, foram especificadas também as espécies de cada pássaro de modo a classificá-las.  


As espécies dos pássaros foram determinadas por meio de consultas realizadas nas plataformas de ciência cidadã [WikiAves](https://www.wikiaves.com.br/) e [eBird](https://ebird.org/home). Assim, foi possível identificar cinco espécies conhecidas popularmente pelos nomes de canário-do-amazonas, chupim, rolinha, sanhaço-do-coqueiro e sanhaço-da-amazônia. Com isso, foi possível levantar e disponibilizar publicamente um [conjnunto de dados](https://github.com/Lucas-Zampar/detector_de_passaros_amazonicos/tree/main/dataset) composto por 940 imagens e 1.836 anotações no formato Pascal VOC. A distribuição das anotações por espécie pode ser visualizada abaixo:

![total_proportion](https://github.com/Lucas-Zampar/detector_de_passaros_amazonicos/assets/75434421/f42e5a3e-6c5f-43f4-a7ac-c95327cf5c6d)

## Treinamento 

Entre os modelos de deep learning voltados para a detecção de objetos, é possível encontrar o [Faster R-CNN](https://arxiv.org/abs/1506.01497) que recai na categoria de detectores de dois estágios. Nessa categoria, o modelo primeiro propõe regiões com possíveis objetos denominadas de RoI (Region of Interest). Em seguida, ele utiliza essas regiões para realizar as detecções. Em geral, detectores de dois estágios tendem a ser mais precisos. Por essa razão, foi decidido utilizar o Faster R-CNN neste trabalho. Além disso, ele foi um dos primeiros detectores bem-sucedidos a empregar redes neurais convolucionais. 

Neste trabalho, houve duas fases consecutivas de treinamento denominadas respectivamente de:

- __fase preliminar__: na qual foi definida uma _baselina_, bem com uma configuração de treinamento ideal. 
- __fase final__: um único modelo definitivo foi treinado com a configuração definida, sendo comparado com a _baseline_.

Além disso, cada fase utilizou um conjunto de dados diferente:

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


