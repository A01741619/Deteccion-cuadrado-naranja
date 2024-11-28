import cv2
import numpy as np

# Función para detectar cuadrados naranjas
def detect_orange_square(frame):
    # Convertir la imagen al espacio de color HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Rango de color naranja en HSV
    lower_orange = np.array([10, 100, 100])  # Ajusta según el tono exacto de tu cuadrado
    upper_orange = np.array([25, 255, 255])

    # Crear una máscara para el color naranja
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # Encontrar contornos en la máscara
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Aproximar el contorno para identificar polígonos
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        area = cv2.contourArea(contour)

        # Detectar si es un cuadrado (4 lados) y tiene un área considerable
        if len(approx) == 4 and area > 500:  # Ajusta el área mínima si es necesario
            cv2.drawContours(frame, [approx], 0, (0, 255, 0), 3)  # Dibuja el contorno en verde

# Capturar video desde la cámara
cap = cv2.VideoCapture(0)

while True:
    # Leer el cuadro actual de la cámara
    ret, frame = cap.read()
    if not ret:
        break

    # Detectar cuadrados naranjas en el cuadro actual
    detect_orange_square(frame)

    # Mostrar el cuadro procesado en la ventana
    cv2.imshow('Detección de Cuadrados Naranjas', frame)

    # Presionar 'q' para salir del programa
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
