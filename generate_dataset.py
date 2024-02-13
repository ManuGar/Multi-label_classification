import os
import shutil



dataset = "dataset_to_train_38frames"
# if not os.path.exists(dataset):
#     os.mkdir(dataset)


path_of_images = "dataset/CNR-STIIMA"
for carpeta in list(os.walk(path_of_images)):
    if "images" in carpeta[0]:
        id_video = carpeta[0].split(os.path.sep)[2].split("_")[0]
        # Aquí lo que hacemos es obtener la anotación del vídeo sobre el que estamos trabajando para poder
        # generar la anotación de las imágenes que creemos
        video_path = carpeta[0][:carpeta[0].rfind(os.path.sep)]
        print(video_path)
        shutil.copytree(os.path.join(video_path,"images"),os.path.join(dataset,carpeta[0].split(os.sep)[2],"images"))
        shutil.copy(os.path.join(video_path,id_video+"_annotation_images.csv"),os.path.join(dataset,carpeta[0].split(os.sep)[2]))
        shutil.copy(os.path.join(video_path,id_video+"_annotation_images_labelSmoothing.csv"),os.path.join(dataset,carpeta[0].split(os.sep)[2]))









