import argparse
import os

# Esta es la versión para label smoothing y así conservar la información sobre todas las clases. Guardamos en un CSV el
# % de cuanto pertenece a cada clase. De esta forma no se pierde nada de información para hacer el entrenamiento.
# Aunque sea un proceso más costoso.
# Necesita el frame de inicio, el final, el número de clases del problema, el path de la anotación del vídeo y el path
# del archivo donde se van a guardar las anotaciones de las imágenes
def annotation_images_labelsmoothing(min_frame, max_frame, number_classes, annotation_path, output_path):
    # Creamos el diccionario para guardar la cuenta de las apariciones de cada clase. Iniciamos todas las clases a 0
    actions_count = {0:0}

    for i in range(number_classes):
        actions_count[i+1] = 0
    id_video = annotation_path.split(os.sep)[2].split(".")[0]

    # Cargamos la anotación y hacemos la cuenta de la cantidad de frames que hay en cada acción
    ann_file = open(annotation_path,"r")
    for line in ann_file:
        line = line.split(" ")
        if(int(line[0])>=min_frame and int(line[0])<max_frame):
            actions_count[int(line[1])]+=1
        if(int(line[0])>max_frame):
            break

    # Calculamos el número de frames para luego realizar las anotaciones. E inicializamos si fuera necesario el archivo
    # de salida para guardar los resultados
    number_frames = max_frame-min_frame
    if (os.path.exists(output_path)):
        output_path_csv = open(output_path, "a")
    else:
        output_path_csv = open(output_path, "w")
        head = "images"
        for i in range(number_classes+1):
            head+= ", class " + str(i)
        output_path_csv.write(head+ "\n")

    # Generamos la línea que vamos a escribir en el archivo de salida
    line_write = id_video + "_from_"+ str(min_frame)+"_to_"+str(max_frame)

    for cla in actions_count:
        line_write += ", "+str(actions_count[cla]/number_frames)
    output_path_csv.write(line_write + "\n")
    output_path_csv.close()


# En este caso, solo guardamos el porcentaje de la clase que más frames tiene en la imagen. Es un proceso más sencillo
# para realizar luego el entrenamiento. El funcionamiento es exactamente igual al anterior solo que solo escribiremos
# la clase predominante dentro del rango de frames a estudiar.
def annotation_images(min_frame, max_frame, number_classes, annotation_path, output_path):
    actions_count = {0: 0}

    for i in range(number_classes):
        actions_count[i + 1] = 0
    id_video = annotation_path.split(os.sep)[2].split(".")[0]

    # Cargamos la anotación y hacemos la cuenta de la cantidad de frames que hay en cada acción
    ann_file = open(annotation_path, "r")
    for line in ann_file:
        line = line.split(" ")
        if (int(line[0]) >= min_frame and int(line[0]) < max_frame):
            actions_count[int(line[1])] += 1
        if (int(line[0]) > max_frame):
            break

    # Calculamos el número de frames para luego realizar las anotaciones. E inicializamos si fuera necesario el archivo
    # de salida para guardar los resultados
    id = 0
    max = 0
    for i in actions_count:
        if actions_count[i] > max:
            max = actions_count[i]
            id = i

    if (os.path.exists(output_path)):
        output_path_csv = open(output_path, "a")
    else:
        output_path_csv = open(output_path, "w")
        head = "images, class\n"

        output_path_csv.write(head)

    line_write = id_video + "_from_" + str(min_frame) + "_to_" + str(max_frame) + ", " + str(id)
    output_path_csv.write(line_write + "\n")
    output_path_csv.close()
    return(id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--min_frame', type=int,
                        help='frame number at which the image starts')
    parser.add_argument('--max_frame', type=int,
                        help='frame number at which the image ends')
    parser.add_argument('--number_classes', type=int,
                        help='number of actions of the problem')
    parser.add_argument('--annotation_path', type=str,
                        help='path of the file with the annotation of the video')
    parser.add_argument('--output_path', type=str,
                        help='path of the output to store CSV with the annotation of the image')
    ap = argparse.ArgumentParser()
    args = vars(ap.parse_args())

    min_frame = args["min_frame"]
    max_frame = args["max_frame"]
    number_classes = args["number_classes"]
    annotation_path = args["annotation_path"]
    output_path = args["output_path"]

    annotation_images_labelsmoothing(min_frame, max_frame, number_classes, annotation_path, output_path)

if __name__ == "__main__":
    main()