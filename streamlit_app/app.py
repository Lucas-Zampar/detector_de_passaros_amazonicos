from app_utils import *

path = 'species/'
species = get_dircontent(path)

st.set_page_config(  layout = 'centered' )

selected_species = st.sidebar.selectbox(
     label = 'Espécies: ',
     options = species,
)

path = path + selected_species + '/'
dates = get_dircontent(path)

selected_date = st.sidebar.selectbox(
     label = 'Datas: ',
     options = dates
)

path = path + selected_date + '/'
files = get_dircontent(path)

selected_file = st.sidebar.selectbox(
      label = 'Arquivos: ',
      options = files,
      key = 'selected_file'
)

path = path + selected_file

video = get_video(path)
frame_count = get_frame_count(video)

selected_frame = st.slider(
    label = 'Frames: ',
    min_value = 0,
    max_value = frame_count-1,
    value = (frame_count-1)//2,
    step = 1,
    key="test_slider"
)

frame = get_frame(video, selected_frame)
st.image(frame, caption = f'Arquivo {files.index(selected_file) + 1} de {len(files)}')



placeholder = st.empty()

c1, c2, c3, c4 = st.columns(4)


with c1: 
      st.button('Anterior ⬅️', on_click=update_selected_file, kwargs={'direction':'left', 'value': selected_file, 'files':files})

with c2: 
     st.button('Random 🎲', on_click= update_slider, kwargs={"value": np.random.randint(0, frame_count-1)})

with c3:
     save_button = st.button('Salvar 💾', on_click=save_frame , kwargs={'frame':frame, 'placeholder': placeholder})

with c4: 
     st.button('Próximo ➡️', on_click=update_selected_file, kwargs={'direction':'right', 'value': selected_file, 'files':files})




# if save_button:     
#      placeholder.success('Salvo!')
#      save_frame(frame)
#      time.sleep(0.2)
#      placeholder.text('')

    















