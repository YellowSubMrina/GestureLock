# save_my_face.py

import numpy as np
import cv2
import insightface

# Загружаем модель
app = insightface.app.FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)

# Открываем камеру
cap = cv2.VideoCapture(0)

print("Нажми 's', чтобы сохранить своё лицо.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    cv2.imshow("Save Your Face", frame)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        faces = app.get(frame)
        if faces:
            my_embedding = faces[0].normed_embedding
            np.save("my_face_embedding.npy", my_embedding)
            print("Лицо сохранено в файл 'my_face_embedding.npy'!")
            break
        else:
            print("Лицо не найдено, попробуй ещё раз.")

cap.release()
cv2.destroyAllWindows()
