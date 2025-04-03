import cv2

vehiche_cascade = cv2.CascadeClassifier('cars.xml')

cap = cv2.VideoCapture('video1.mp4')

while True: 
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    vehicle = vehiche_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=6, minSize=(50, 50), maxSize=(500, 500))

    for (x, y, w, h) in vehicle:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('img', img)
    k = cv2.waitKey(30)

    print(f"Vehículos detectados: {len(vehicle)}")
    
    if k == 27:
        break
cap.release()