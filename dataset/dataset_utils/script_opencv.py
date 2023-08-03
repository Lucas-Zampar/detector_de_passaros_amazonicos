import cv2 # import da biblioteca OpenCV
import datetime # import da biblioteca datetime para anotação da data de gravação

camera_id = 1 # id da câmera entre as três conectadas

webcam = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW) # objeto de captura do conteúdo da webcam
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # define a largura para 1280 (720p)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) # define a altura para 720 (720p)

if not webcam.isOpened(): 
    # caso a câmera não esteja disponível
    # exibe mensagem de erro
    print("Erro ao abrir a câmera!!!")
    exit() 

frame_width = int(webcam.get(3)) # recupera a largura de gravação
frame_height = int(webcam.get(4)) # recupera a altura de gravação
resolution = (frame_width, frame_height) # define a resolução
codec = cv2.VideoWriter_fourcc(*"MJPG") # código FourCC que identifica o codec do vídeo utilizado (compatível com a extensão AVI)
file_name_template = "{}-{}-{} {}-{}-{}.avi" # f-string referente ao nome do arquivo que conterá o timestamp aproxido da gravação seguido da extensão .AVI
file_path = "" # caminho onde as gravações serão armazenadas
fps = 10 # FPS do vídeo

date_reading = datetime.datetime.now() # lê a data atual
day, month, year = date_reading.day, date_reading.month, date_reading.year # recupera dia, mês e ano

time_interval = 5*60 # define o intervalo de tempo da gravação (5 min)

# ao todo, serão realizadas 12 gravações de 5 minutos por execução de script totalizando 1 hora de gravações. 
# após esse período, o script deve ser executado novamente.
# dessa forma, haverá 12 loops nos quais uma gravação de 5 minutos será realizada.

loop_counting = 0 
amount_of_loop = 12 

while loop_counting < amount_of_loop: 
    # leitura do horário em que a gravação começou
    first_time_reading  = datetime.datetime.now()
    #recupera hora, minuto e segundo
    hour, minute, second  = first_time_reading.hour, first_time_reading.minute, first_time_reading.second
    # define o nome do arquivo com o timestamp do início da gravação
    file_name = file_path + file_name_template.format(day, month, year, hour, minute, second)
    # objeto que gravará os vídeos localmente
    video_writer = cv2.VideoWriter(file_name, codec, fps, resolution)

    while True:
        # horário de leitura do próximo frame
        last_time_reading = datetime.datetime.now() 
        # leitura do próximo frame
        ret, frame = webcam.read()
        if ret:
            # se o frame tiver sido lido corretamente, mostre o frame na janela
            cv2.imshow(f"CAM {camera_id}", frame)
            # grave o frame
            video_writer.write(frame)
        else: 
            # caso contrário, indique que houve falha na leitura
            print("A leitura do frame falhou!")
    
        if cv2.waitKey(1) == ord("q"):
            # se a tecla q for pressionada, encerra a gravação
            video_writer.release()
            loop_counting = 100
            break
        
        if (last_time_reading - first_time_reading).seconds >= time_interval:
            # se o tempo de gravação for atingido, encerre a gravação 
            video_writer.release()
            break
    # inicia um novo ciclo de gravação
    loop_counting +=1 

# no final das 12 gravações, libere a webcam    
webcam.release()
# libera todas as janelas
cv2.destroyAllWindows()

# encerramento do script