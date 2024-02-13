import argparse
import pandas as pd
import os
import numpy as np
import cv2
import annotation_images

# Función para normalizar los valores de las posiciones de los joint. Tienen que estar entre 0 y 255
# para poder guardarlos en los píxeles de la imagen
def normalize(value: float, lower_bound: float, higher_bound: float, max_value: int, min_value: int) -> float:
    if value > higher_bound:
        ret_value = max_value
    elif value < lower_bound:
        ret_value = min_value
    else:
        ret_value = (max_value * ((value - lower_bound) / (higher_bound - lower_bound)))
    return ret_value

def calc_max_min_coord(skeleton_list, video_path):

    max_x = 0
    min_x = 0

    max_y = 0
    min_y = 0

    max_z = 0
    min_z = 0

    if(len(skeleton_list)>0):
        frame = pd.read_csv(os.path.join(video_path, skeleton_list[0]), sep="\t", header=0)
        frame = pd.DataFrame(frame)
        for _, line in frame.iterrows():
            max_x = line[3]
            min_x = line[3]
            max_y = line[4]
            min_y = line[4]
            max_z = line[5]
            min_z = line[5]
            break
    for skeleton in skeleton_list:
        frame = pd.read_csv(os.path.join(video_path, skeleton), sep="\t", header=0)
        frame = pd.DataFrame(frame)
        for _, line in frame.iterrows():
            if(line[3]>max_x): max_x=line[3]
            if(line[3]<min_x): min_x=line[3]

            if (line[4] > max_y): max_y = line[4]
            if (line[4] < min_y): min_y = line[4]

            if (line[5] > max_z): max_z = line[5]
            if (line[5] < min_z): min_z = line[5]
    return min_x, max_x, min_y, max_y, min_z, max_z

def genenerate_images_from_skeletons_list(skeleton_list,video_path, annotation_file, error, images_per_class, max_frames=68, stride=20):
    # Comprobamos que la lista no esta vacía para obtener el primer elemento y sacar la ruta para poder guardar las
    # imágenes y las anotaciones que generemos
    max_error = max_frames+25 # Esto es la mayor diferencia que vamos a aceptar entre el primer y el último frame. Se
    # pueden perder frames y eso hace que la acción a la que hace referencia se vea discontinuada. Si se supera este
    # umbral, entonces la imagen no se guardará.
    splited_path = video_path.split(os.sep)
    # name_dir = os.path.join(splited_path[0], splited_path[1],splited_path[2])
    name_dir = os.path.join("dataset_to_train_77frames", splited_path[1],splited_path[2])
    id_video = splited_path[2].split("_")[0]



    # En este vector guardamos el orden de los joint para luego poder recorrerlo en el orden que queremos
    joints_order = [1, 2, 3, 26, 27, 28, 29, 28, 27, 30, 31, 30, 27, 26, 3, 2, 4, 5, 6, 7, 8, 9, 8, 7, 10, 7, 6, 5, 4,
                    2, 11, 12, 13, 14,
                    15, 16, 15, 14, 17, 14, 13, 12, 11, 2, 1]
    # Plantilla de la imagen vacía (llena de 0) para ir completándola en cada paso
    img = np.zeros((max_frames, len(joints_order), 3), dtype=np.uint8)

    # Estos son los valores máximos y mínimos que toman los joint y que se van a usar para normalizar todos los puntos
    # a los valores que entran en una imagen 0-255
    min_x, max_x, min_y, max_y, min_z, max_z = calc_max_min_coord(skeleton_list, video_path)
    # Estos son los ficheros donde se guardaran las anotaciones en los dos distintos formatos de las imágenes
    # que estamos generando
    output_path_labelsmoothing = os.path.join(name_dir, id_video + "_annotation_images_labelSmoothing.csv")
    output_path = os.path.join(name_dir, id_video + "_annotation_images.csv")
    count_generated_frames = 0
    i = 0  # Contador para el número de frames tratados. Al llegar a max_frames se vuelca la imagen generada a disco
    l = 0  # Contador para el número de frames tratados. Para contar cuando hemos llegado al final del vídeo y volcar los últimor frames a disco
    # Vamos a generar las imágenes del video que trate recorriendo los frames que se han pasado por parámetro
    # comprobando el número de frames para ir volcando la imagen generada en disco. Hay que contemplar los casos como
    # estar en el final del vídeo o que tengamos un vídeo muy corto

    n_frames = int(skeleton_list[-1].split("/")[-1].split("_")[0].split("FrameID")[-1])
    print("Los frames que debería tener el vídeo son: " + str(n_frames+1))

    # while l < len(skeleton_list) and count_generated_frames < len(skeleton_list):
    while l <= n_frames and count_generated_frames <= n_frames:
        esta = False
        for skele in skeleton_list:
            # Estas condiciones de parada son por si lo encuentra o por si el número de frame que va a revisar ya es más
            # grande que el frame que estamos buscando
            if int(skele.split("/")[-1].split("_")[0].split("FrameID")[-1]) == l:
                skeleton = skele
                esta = True
                break
            if (int(skele.split("/")[-1].split("_")[0].split("FrameID")[-1]) > l):
                break
        if(esta):
            # skeleton = skeleton_list[l]
            # Leemos el archivo del frame para trabajar con él
            frame = pd.read_csv(os.path.join(video_path, skeleton) , sep="\t", header=0)
            frame = pd.DataFrame(frame)
            joints = {}
            if(i==0):
                frame_start = int(skeleton.split("/")[-1].split("_")[0].split("FrameID")[-1])
            # Aquí guardamos los joints y los datos que queremos en un diccionario para luego generar la imagen
            for j, line in frame.iterrows():
                # print(str(line[3]) + "\t" + str(line[4]) + "\t" + str(line[5]))
                # Los valores máximos y mínimos han sido seleccionados mirando los valores que obtienen los esqueletos
                # que ibamos a estudiar (espero que esos valores se conserven también en los otros vídeos)
                joints[int(line[1])] = [normalize(line[3], min_x, max_x, 255, 0), normalize(line[4], min_y, max_y, 255, 0),
                                        normalize(line[5], min_z, max_z, 255, 0)]
            # Aquí vamos a escribir la matriz con los datos a la imagen para guardarla. Primero comprobamos que podamos
            # seguir guardando la matriz con los datos de la imagen que generamos. Si hemos llegado al límite de frames o
            # al final del fichero, lo que haremos es escribirla en la imagen.

            # Creamos la carpeta de las imágenes en el caso de que no exista
            # Guardamos las imágenes generadas en una carpeta para que esté organizada junto al vídeo que hacen
            # referencia y poder saber fácilmente la clase de cada fila de las imágenes.
            if not os.path.exists(os.path.join(name_dir, "images")):
                os.makedirs(os.path.join(name_dir, "images"))
            if i < max_frames:
                # Guardamos el contenido de las componentes de los diferentes joints en las coordenadas de la imagen.
                # Para cada frame generamos una fila en la imagen
                for idx in range(len(joints_order)):
                    # print(joints[joints_order[idx]][0])
                    img[i, idx, 0] = joints[joints_order[idx]][0]
                    img[i, idx, 1] = joints[joints_order[idx]][1]
                    img[i, idx, 2] = joints[joints_order[idx]][2]
                if (l >= n_frames) or i >= (n_frames):
                    frame_end = int(skeleton.split("/")[-1].split("_")[0].split("FrameID")[-1])
                    if ((frame_end - frame_start) < max_error):
                        annotation_images.annotation_images_labelsmoothing(count_generated_frames,
                                                                           count_generated_frames + max_frames, 12, annotation_file,
                                                                           output_path_labelsmoothing)
                        images_per_class[annotation_images.annotation_images(count_generated_frames, count_generated_frames + max_frames, 12, annotation_file,
                                                                           output_path)]+=1
                        # Guardamos el nombre que tendrá la imagen
                        generated_image = os.path.join(name_dir, "images",
                                                       id_video + "_from_" + str(count_generated_frames) + "_to_" + (
                                                           str(count_generated_frames + max_frames)))
                        # count_generated_frames = count_generated_frames + max_frames - stride
                        cv2.imwrite(generated_image + ".jpg", img)
                        cv2.imwrite(generated_image + "X.jpg", img[:, :, 0])
                        cv2.imwrite(generated_image + "Y.jpg", img[:, :, 1])
                        cv2.imwrite(generated_image + "Z.jpg", img[:, :, 2])
                        img = np.zeros((max_frames, len(joints_order), 3), dtype=np.uint8)
                    else:
                        error+=1


                l += 1
                i += 1
            else:
                # Guardamos el id del video a partir de la ruta, con el id podremos generar a la vez también las
                # anotaciones de las imágenes que se van generando
                # Con el id del vídeo que estamos tratando, buscamos la anotación del vídeo para poder generar la
                # anotación de las imágenes
                frame_end = int(skeleton.split("/")[-1].split("_")[0].split("FrameID")[-1])
                if((frame_end-frame_start)< max_error):

                    annotation_images.annotation_images_labelsmoothing(count_generated_frames,count_generated_frames + max_frames, 12,
                                                                       annotation_file, output_path_labelsmoothing)
                    images_per_class[annotation_images.annotation_images(count_generated_frames, count_generated_frames + max_frames, 12, annotation_file,
                                                        output_path)]+=1
                    # Guardamos el archivo de la anotación de las imágenes en el mismo nivel que la carpeta de las imágenes
                    # y la de los esqueletos
                    # Aquí si cambiamos de problema o modificamos las clases que queremos estudiar habrá que cambiar ese 12
                    # que hace referencia al número de clases del problema que estamos tratando
                    # Con esta función generamos las anotaciones

                    # Guardamos el nombre que tendrá la imagen
                    generated_image = os.path.join(name_dir, "images",
                                                   id_video + "_from_" + str(count_generated_frames) + "_to_" + (
                                                       str(count_generated_frames + max_frames)))

                    # Reiniciamos los contadores de los frames tratados, guardamos las imágenes y actualizamos el valor de l
                    cv2.imwrite(generated_image + ".jpg", img)
                    cv2.imwrite(generated_image + "X.jpg", img[:, :, 0])
                    cv2.imwrite(generated_image + "Y.jpg", img[:, :, 1])
                    cv2.imwrite(generated_image + "Z.jpg", img[:, :, 2])
                    img = np.zeros((max_frames, len(joints_order), 3), dtype=np.uint8)
                    # Actualizamos el número de frames tratados
                    count_generated_frames = count_generated_frames + i - stride
                else:
                    # print(timestamp_end - timestamp_start)
                    # print(count_generated_frames)
                    error += 1
                    count_generated_frames = count_generated_frames + i - stride

                i = 0
                l = l - stride

        else:
            i+=1
            l+=1

    return error, images_per_class



# Para generar las imágenes necesitamos el path de la carpeta en la que estan los frames de los esqueletos de los videos,
# los frames que vamos a guardar en cada imagen y la cantidad de frames que va a compartir con la imagen siguiente
def generate_images(skeleton_path, annotation_path, max_frames=68, stride=20):

    # Recorremos la lista de los videos para pasar a la función encargada de generar las imágenes la lista con las rutas
    # de todos los frames que contienen ese vídeo

    error = 0
    images_per_class = {0: 0}
    for i in range(12):
        images_per_class[i + 1] = 0

    for carpeta in list(os.walk(skeleton_path)):
        if "Skeletons/" in carpeta[0]:
            id_video = carpeta[0].split(os.path.sep)[2].split("_")[0]
            # Aquí lo que hacemos es obtener la anotación del vídeo sobre el que estamos trabajando para poder
            # generar la anotación de las imágenes que creemos
            for ann_files in os.listdir(annotation_path):
                id_ann_file = ann_files.split(".")[0]

                if id_ann_file == id_video:
                    annotation_file = os.path.join(annotation_path, ann_files)
                    break

            video_path = carpeta[0]
            skeletons_carpeta = os.listdir(carpeta[0])
            skeletons_carpeta.sort()
            print("El número de frames que hay en el vídeo " + carpeta[0] + " es: " + str(len(skeletons_carpeta)))
            error, images_per_class = genenerate_images_from_skeletons_list(skeletons_carpeta,video_path,annotation_file, error, images_per_class,max_frames, stride)
            if (os.path.exists("resultados_clases.txt")):
                f = open("resultados_clases_STIIMA.txt", "a")
            else:
                f = open("resultados_clases_STIIMA.txt", "w")
            f.write("Las imágenes que no se han generado son: " + str(error) + "\n")
            f.write("las imágenes que tienen cada clase son: " + str(images_per_class) + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skeleton_path', type=str,
                        help='path of the skeleton files')
    parser.add_argument('--max_frames', type=int, default=68,
                        help='number of frames to store in each image')
    parser.add_argument('--stride', type=int, default=20,
                        help='number of frames to share with the previous image')
    parser.add_argument('--annotation_path', type=str,
                        help='path of the annotation files')
    ap = argparse.ArgumentParser()
    args = vars(ap.parse_args())
    skeleton_path = 'mal/vid'
    skeleton_path = 'pru2/vid'

    # max_frames = 68 #Este resultado lo ha dado la mediana
    # max_frames = 77 #Este resultado lo ha dado la media supuestamente bien calculada ahora
    max_frames = 77

    # max_frames = 128 #Esto es por el resultado que nos ha dado la media y la mediana
    # stride = 20  # Es el que puesto por defecto, con esto tendríamos un salto de 48 frames
    stride = max_frames - 57 # Para tener un salto de 20 frames
    stride = max_frames - 1 # Para tener un salto de 1 frames


    annotation_path = "txtAnnotation/txtAnnotationRioja"
    annotation_path = "pru2/txtAnnota"

    # skeleton_path = args["skeleton_path"]
    # max_frames = args["max_frames"]
    # stride = args["stride"]
    generate_images(skeleton_path, annotation_path, max_frames, stride)

if __name__ == "__main__":
    main()