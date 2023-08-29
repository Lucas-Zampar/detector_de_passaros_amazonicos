# Extração de Frames

Após as gravações terem sido realizadas no comedouro por meio de webcams, cortes manuais foram realizados de modo a excluir períodos de ausência de pássaros. Além disso, eles foram organizados com base na espécie predominante, bem como na data de gravação. 

Assim, foi desenvolvida uma aplicação em Streamlit capaz de selecionar frames a partir dos cortes realizados. Os frames podem ser selecionadas com base na espécie predominante e na data de gravação do corte. Além disso, a seleção pode ser aleatória ou em um momentos específicos do corte. 

Abaixo, há um vídeo demonstrando a utilização da interface: 


# Conteúdo

## Arquivos

- __app_utils.py__: código auxiliar para acessar e salvar os frames.
- __app.py__: implementação da interface gráfica.

## Pastas

- __cortes_por_especie__: pasta onde devem estar presentes os cortes. Devido a limitação de espaço, os cortes estão disponíveis no Google Drive clicando [aqui](https://drive.google.com/drive/folders/12ueqV4UuxU2ebdD4YYV4xpQZ3hxHhIk-?usp=drive_link).
- __local_dataset__: pasta onde os frames selecionados são salvos. 


# Instalações

Inicialmente, criamos um ambiente virtual através do gerenciador mamba: 

```
mamba create -n APP python=3.9 -y
```

Então, ativamos o ambiente recém-criado: 

```
mamba activate APP 
```

Em seguida, instalamos o Streamlit: 

```
pip install streamlit
```

Além disso, é necessário instalar também os pacotes Numpy e OpenCV:

```
pip install numpy
pip install opencv-python
```

Dessa forma, basta iniciar a aplicação: 

```
streamlit run app.py
```

