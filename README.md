# Detecção de Pássaros Amazônicos (TCC)

Neste repositório, você encontrará a implementação do meu trabalho de conclusão de curso (TCC).  

O trabalho teve como  objetivo detectar espécies de pássaros amazônicos que frequentam comedouros residenciais através de uma abordagem baseada em Deep Learning. Para tanto, foi levantado um conjunto de 940 imagens anotadas para a tarefa de detecção de objetos. Em seguida, a base de dados foi empregada para treinar diferentes configurações do modelo Faster R-CNN através do _framework_ [IceVision](https://github.com/airctic/icevision). Abaixo, é possível verificar a aplicação do modelo selecionado: 

https://github.com/Lucas-Zampar/detector_de_passaros_amazonicos/assets/75434421/e58b7a8e-edac-4e0e-ad77-226130b5126d

Infelizmente, não foi possível armazenar todos os arquivos neste repositório. Porém, eles estão publicamente disponíveis [no repositório completo compartilhado pelo Google Drive](https://drive.google.com/drive/folders/12ueqV4UuxU2ebdD4YYV4xpQZ3hxHhIk-?usp=drive_link).

# Estrutura do Repositório

A estrutura deste repositório está organizada da seguinte forma: 

- __codigos_de_desenvolvimento__: contem todos os códigos e notebooks utilizados para treinar e avaliar o modelo, bem como para anotar novas imagens e vídeos. 
- __dataset__: contem o conjunto de dados utilizado para treinar o modelo. 
- __faster_rcnn__: contem o checkpoint e metadados do modelo definitivo e dos intermediários produzidos durante o trabalho. 
    - OBS: apenas o modelo definitivo está disponíveis no GitHub. Os demais se encontram [no repositório completo compartilhado pelo Google Drive](https://drive.google.com/drive/folders/12ueqV4UuxU2ebdD4YYV4xpQZ3hxHhIk-?usp=drive_link). 
- __streamlit_app__: contem os códigos da interface gráfica empregada para extrair os frames a partir dos cortes realizados.
    - OBS: os cortes não estão disponíveis no GitHub, apenas [no repositório completo compartilhado pelo Google Drive](https://drive.google.com/drive/folders/12ueqV4UuxU2ebdD4YYV4xpQZ3hxHhIk-?usp=drive_link).

