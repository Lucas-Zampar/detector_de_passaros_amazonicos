# Conteúdo 

Nesta página, você encontrará:

## Arquivos 

- __detect_image.ipynb__: notebook empregado para anotar uma imagem.
- __detect_vidoe.ipynb__: notebook empregado para anotar um vídeo.
- __make_report_csv_files.ipynb__: notebook empregado para gerar as tabelas com os resultados de cada modelo.
- __train_model.ipynb__: notebook empregado para treinar e avaliar os modelos. 
- __demo.jpg__: imagem demonstrativa para anotação.
- __demo.mp4__: vídeo demonstrativa para anotação.


## Pastas

- __graficos_barra__: contem arquivos CSV com a proporção de anotações por espécie encontrada em cada conjunto de dados, além de um notebook que gera os gráficos de barra a partir deles. 
- __project_utils__: pasta contendo arquivos com wrappers implementadas em Python de funções do IceVision, FifityOne e OpenCV frequentemente empregadas durante o desenvolvimento do trabalho.

# Instalações 

De modo a executar os notebooks em uma máquina local, é necessário instalar o framework IceVision e FifityOne. É válido destacar que o IceVision oferece suporte apenas para Linux e MacOS.  

Inicialmente, criamos um ambiente virtual através do gerenciador mamba: 

```
mamba create -n ICE python=3.9 -y
```

Então, ativamos o ambiente recém-criado: 

```     
mamba activate ICE
```

De modo a ter acesso aos modelos de detecção de objetos, é necessário instalar os pacotes mmcv-full e mmdet respectivamente: 

```
pip install mmcv-full==1.3.17 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10.0/index.html
pip install mmdet==2.17.0
```

Com isso, já é possível instalar a versão 0.12 do framework IceVision empregada neste trabalho: 

```
pip install icevision[all]==0.12.0
```

É importante destacar que a versão 0.12 do IceVision possui incompatibilidade com versões do pacote `setuptools` superiores a 59.5.0. Então, é necessário reduzir a versão deste pacote: 

```
pip uninstall setuptools -y
pip install setuptools==59.5.0 
```

O mesmo é válido para o pacote `sahi` que deve ser inferior a versão 0.10:

```
pip uninstall sahi -y
pip install sahi==0.8.19
```

Bem como o pacote `numpy` que deve ser inferior a versão 1.25: 

```
pip uninstall numpy -y
pip install numpy==1.23
```

Com isso, o framework FiftyOne já pode ser instalado: 

```
pip install fiftyone
```

Por fim, é necessário instalar o Jupyter Notebook para ter acesso ao conteúdo presente nos notebooks: 

```
pip install jupyter
```


__OBS__: o framework IceVision é experimental. Portanto, novos problemas de versionamento não identificados durante o trabalho podem surgir futuramente.










