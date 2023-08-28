# Detecção de Pássaros Amazônicos (TCC)

Neste repositório, você encontrará a implementação do meu trabalho de conclusão de curso (TCC).  

O trabalho teve como  objetivo detectar espécies de pássaros amazônicos que frequentam comedouros residenciais através de uma abordagem baseada em Deep Learning. Para tanto, foi levantado um conjunto de 940 imagens anotadas para a tarefa de detecção de objetos. Em seguida, a base de dados foi empregada para treinar diferentes configurações do modelo Faster R-CNN através do _framework_ [IceVision](https://github.com/airctic/icevision). Abaixo, é possível verificar a aplicação do modelo selecionado: 

https://github.com/Lucas-Zampar/detector_de_passaros_amazonicos/assets/75434421/e58b7a8e-edac-4e0e-ad77-226130b5126d

Infelizmente, não foi possível armazenar todos os arquivos neste repositório. Porém, eles estão publicamente disponíveis [no repositório completo compartilhado pelo Google Drive](https://drive.google.com/drive/folders/12ueqV4UuxU2ebdD4YYV4xpQZ3hxHhIk-?usp=drive_link).

# Proposta 

## Conjunto de Dados

De modo a capturar as imagens dos pássaros, o estudo teve acesso ao comedouro de uma residência no estado do Amapá. Os moradores locais utilizam o comedouro para observar os pássaros se alimentando das sementes e frutas que ali colocam. O comedouro pode ser visualizado abaixo:

![comedouro](https://github.com/Lucas-Zampar/detector_de_passaros_amazonicos/assets/75434421/a0c60cdd-c422-49c0-9599-b37959cf570e)

Três webcams Logitech C270 HD foram instaladas no comedouro para registrar os pássaros se alimentando. As webcams eram conectadas a um notebook que executava um [script para capturar as gravações](https://github.com/Lucas-Zampar/detector_de_passaros_amazonicos/blob/main/dataset/dataset_utils/script_opencv.py) com auxílio da biblioteca OpenCV. O esquema geral de aquisição dos dados é exposto abaixo:

![aquisicao_de_dados](https://github.com/Lucas-Zampar/detector_de_passaros_amazonicos/assets/75434421/1304c715-bcda-47d3-848e-c604306c0b06)

As gravações foram realizadas entre os dias 07/09/2022 e 15/07/2022. A gravações entre os dias 07 e 13 passaram por um processo de recorte manual no qual momentos com ausência de pássaros foram descartados. Os recortes foram organizados com base na data de gravação e na espécie predominante. A partir desses recortes, _frames_ foram extraídos para comporem as imagens do conjunto de dados. 

De modo a facilitar a extração dos _frames_, foi desenvolvida uma [aplicação em streamlit](https://github.com/Lucas-Zampar/detector_de_passaros_amazonicos/tree/main/streamlit_app). Por meio dela, era possível navegar entre os recortes, além de filtrá-los pela data e pela espécie. Os _frames_ podiam ser selecionados aleatoriamente ou em momentos específicos. A interface da aplicação é demonstrada abaixo: 

https://github.com/Lucas-Zampar/detector_de_passaros_amazonicos/assets/75434421/f92f70fb-00c9-42a9-be28-88ab50c10324


Após a extração dos _frames_, seguiu-se para a etapa de anotação. Para tanto, os _frames_ foram carregados na plataforma [RoboFlow](https://roboflow.com/). O processo de anotação ocorreu ao se desenhar caixas delimitadoras sobre os pássaros, além de especificar suas respectivas espécies. As espécies foram determinadas ao se consultar registros encontrados nas plataformas de ciência cidadã [WikiAves](https://www.wikiaves.com.br/) e [eBird](https://ebird.org/home).  

Com isso, foi possível levantar um [conjnunto de dados disponibilizado publicamente](https://github.com/Lucas-Zampar/detector_de_passaros_amazonicos/tree/main/dataset) que é composto por 940 imagens e 1.836 anotações no formato Pascal VOC. A distribuição das anotações por espécie pode ser visualizada abaixo:


![total_proportion](https://github.com/Lucas-Zampar/detector_de_passaros_amazonicos/assets/75434421/f42e5a3e-6c5f-43f4-a7ac-c95327cf5c6d)

## Treinamento 















# Estrutura do Repositório

A estrutura deste repositório está organizada da seguinte forma: 

- __codigos_de_desenvolvimento__: contem todos os códigos e notebooks utilizados para treinar e avaliar o modelo, bem como para anotar novas imagens e vídeos. 
- __dataset__: contem o conjunto de dados utilizado para treinar o modelo. 
- __faster_rcnn__: contem o checkpoint e metadados do modelo definitivo e dos intermediários produzidos durante o trabalho. 
    - OBS: apenas o modelo definitivo está disponíveis no GitHub. Os demais se encontram [no repositório completo compartilhado pelo Google Drive](https://drive.google.com/drive/folders/12ueqV4UuxU2ebdD4YYV4xpQZ3hxHhIk-?usp=drive_link). 
- __streamlit_app__: contem os códigos da interface gráfica empregada para extrair os frames a partir dos cortes realizados.
    - OBS: os cortes não estão disponíveis no GitHub, apenas [no repositório completo compartilhado pelo Google Drive](https://drive.google.com/drive/folders/12ueqV4UuxU2ebdD4YYV4xpQZ3hxHhIk-?usp=drive_link).

