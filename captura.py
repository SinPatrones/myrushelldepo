import cv2
from matplotlib import pyplot
import os
import imutils


def capturarImagen():
    nombre = 'Personay'
    direccion = 'E:\\Projects\\S3\\Arquitectura\\DeteccionRostro_v1.0\\Data'  # Cambia a la ruta donde hayas almacenado Data
    carpeta = direccion + '\\' + nombre

    if not os.path.exists(carpeta):
        print('Carpeta creada:', carpeta)
        os.makedirs(carpeta)

    cap = cv2.VideoCapture('Video.mp4')
    count = 0
    faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        print("ejecutando...")
        ret, frame = cap.read()
        if ret == False: break
        frame = imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        copia = frame.copy()

        caras = faceClassif.detectMultiScale(gray,1.3,5)

        print("LEN CARAS", len(caras))
        for (x, y, w, h) in caras:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cara_reg = frame[y:y+h,x:x+w]
            cara_reg = cv2.resize(cara_reg, (150, 150), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(carpeta + "\\rostro_{}.jpg".format(count), cara_reg)
            print("GUARDANDO EN -->", carpeta + "\\rostro_{}.jpg");
            count += 1
            print("capturando...", count)
        cv2.imshow("Entrenamiento", frame)

        t = cv2.waitKey(1)
        if t == 27 or count >= 100:
            break

    cap.release()
    cv2.destroyAllWindows()