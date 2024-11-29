import cv2
import numpy as np

# Función para callback de los sliders (sin funcionalidad específica en este caso)
def trackbar_callback(value):
    pass

# Crear ventana con barras de ajuste para el rango HSV
cv2.namedWindow('Ajustes')
cv2.createTrackbar('LH', 'Ajustes', 0, 255, trackbar_callback)
cv2.createTrackbar('LS', 'Ajustes', 100, 255, trackbar_callback)
cv2.createTrackbar('LV', 'Ajustes', 100, 255, trackbar_callback)
cv2.createTrackbar('UH', 'Ajustes', 25, 255, trackbar_callback)
cv2.createTrackbar('US', 'Ajustes', 255, 255, trackbar_callback)
cv2.createTrackbar('UV', 'Ajustes', 255, 255, trackbar_callback)

# Inicializar la cámara
camara = cv2.VideoCapture(0)

if not camara.isOpened():
    print("No se pudo acceder a la cámara.")
    exit()

while True:
    # Capturar un frame de la cámara
    ok, frame = camara.read()
    if not ok:
        print("No se pudo capturar el frame.")
        break

    # Obtener valores de los sliders
    l_h = cv2.getTrackbarPos('LH', 'Ajustes')
    l_s = cv2.getTrackbarPos('LS', 'Ajustes')
    l_v = cv2.getTrackbarPos('LV', 'Ajustes')
    u_h = cv2.getTrackbarPos('UH', 'Ajustes')
    u_s = cv2.getTrackbarPos('US', 'Ajustes')
    u_v = cv2.getTrackbarPos('UV', 'Ajustes')

    rango_inferior = np.array([l_h, l_s, l_v])
    rango_superior = np.array([u_h, u_s, u_v])

    # Convertir el frame a espacio de color HSV
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Crear máscara para filtrar el color en el rango especificado
    mascara = cv2.inRange(frame_hsv, rango_inferior, rango_superior)

    # Limpiar la máscara con operaciones morfológicas
    kernel = np.ones((5, 5), np.uint8)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)

    # Encontrar contornos en la máscara
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contorno in contornos:
        # Filtrar contornos según el área mínima
        if cv2.contourArea(contorno) > 500:
            # Aproximar el contorno a un polígono
            aproximacion = cv2.approxPolyDP(contorno, 0.02 * cv2.arcLength(contorno, True), True)
            if len(aproximacion) >= 4:  # Identificar figuras con al menos 4 lados
                x, y, w, h = cv2.boundingRect(aproximacion)
                proporciones = float(w) / h
                if 0.8 < proporciones < 1.2:  # Identificar figuras cuadradas
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                else:
                    # Dibujar contornos para otras figuras
                    cv2.drawContours(frame, [aproximacion], 0, (255, 0, 0), 3)

    # Mostrar el frame original y la máscara
    cv2.imshow('Video', frame)
    cv2.imshow('Mascara', mascara)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
camara.release()
cv2.destroyAllWindows()
