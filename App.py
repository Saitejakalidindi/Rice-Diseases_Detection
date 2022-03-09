import streamlit as st
print(st.__version__)
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import pickle
from PIL import Image
import os

labels = ['Bacterial leaf blight', 'Brown spot', 'Leaf smut']
with open(r'./Model/rice_pred.pkl', 'rb') as f:
    model = pickle.load(f)
st.set_page_config(
   page_title="Rice Diseases Detection"
)

dis_info={
'Bacterial leaf blight':"Bacterial blight is caused by Xanthomonas oryzae pv oryzae. It causes wilting of seedlings and yellowing and drying of leaves.",
'Brown spot':"Brown spot has been historically largely ignored as one of the most common and most damaging rice diseases.",
'Leaf smut':"Leaf smut causes chalkiness of grains which leads to reduction in grain weight. It also reduces seed germination."
}
dis_occurs={
'Bacterial leaf blight':"The disease is most likely to develop in areas that have weeds and stubbles of infected plants. It can occur in both tropical and temperate environments, particularly in irrigated and rainfed lowland areas. In general, the disease favors temperatures at 25−34°C, with relative humidity above 70%. It is commonly observed when strong winds and continuous heavy rains occur, allowing the disease-causing bacteria to easily spread through ooze droplets on lesions of infected plants. Bacterial blight can be severe in susceptible rice varieties under high nitrogen fertilization.",
'Brown spot':"The disease can develop in areas with high relative humidity (86−100%) and temperature between 16 and 36°C. It is common in unflooded and nutrient-deficient soil, or in soils that accumulate toxic substances. For infection to occur, the leaves must be wet for 8−24 hours. The fungus can survive in the seed for more than four years and can spread from plant to plant through air.",
'Leaf smut':'''The disease can occur in areas with high relative humidity (>90%) and temperature ranging from 25−35 ºC.  Rain, high humidity, and soils with high nitrogen content also favors disease development. Wind can spread the fungal spores from plant to plant.'''
}
dis_manage={
'Bacterial leaf blight':'''| Solution        |
                           | ------------- |
                           | Use balanced amounts of plant nutrients, especially nitrogen.      | 
                           | Allow fallow fields to dry in order to suppress disease agents in the soil and plant residues.    | 
                           | Ensure good drainage of fields (in conventionally flooded crops) and nurseries. | ''',

'Brown spot':'''           | Solution        |
                           | ------------- |
                           | Monitor soil nutrients regularly.   | 
                           | Apply required fertilizers.    | 
                           | For soils that are low in silicon, apply calcium silicate slag before planting. | ''',

'Leaf smut':'''            | Solution        |
                           | ------------- |
                           | Treat seeds at 52°C for 10 min..   | 
                           | Remove infected seeds, panicles, and plant debris after harvest.    | 
                           | Keep the field clean & use certified seeds|
                           |Remove infected seeds, panicles, and plant debris after harvest.|''',
}
def Rice_Disease_Detection(file):
    dimension=(104, 104)
    flat_data = []
    img = imread(file)
    img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')
    flat_data.append(img_resized.flatten())
    pred = model.predict(flat_data)
    pred = [str(i) for i in pred]
    pred = int("".join(pred))
    pred = labels[pred]
    return pred

def run():
    st.title("🌾 Rice Disease Prediction")
    img_file = st.file_uploader("Choose an Image", type=["jpg", "png",'jpeg'])
    if img_file is not None:
        img = Image.open(img_file)
        width, height = img.size
        img = img.resize((440,144))
        st.image(img,use_column_width=False)
        save_image_path = './Upload_Images/'+img_file.name

        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())
        result= Rice_Disease_Detection(save_image_path)
        st.subheader('✅ **Prediction of '+ img_file.name+ '**')
        st.success("Predicted Rice Disease is: "+result)
        st.subheader('✅ **Description about '+str(result)+'**')
        st.info(dis_info[result])
        st.subheader('✅ **Why and Where it occurs?**')
        st.warning(dis_occurs[result])
        st.subheader('✅ **How to Manage '+str(result)+'?**')
        st.markdown(dis_manage[result],unsafe_allow_html=True)
        st.subheader('✅ **Information of '+ img_file.name+ '**')
        img_info_dict = {'Image Name':img_file.name,'Width':str(width)+' pixels','Height':str(height)+' pixels','Size':f"{os.path.getsize(save_image_path) / float(1 << 20):,f} MB"}
        st.json(img_info_dict)


    else:
        st.warning('Upload Image to Continue..!')
run()