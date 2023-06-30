from ultralytics import YOLO
import cv2

model_path = 'best.pt'

image_path = 'bobcat_10170.jpg'
img = cv2.imread(image_path)

model = YOLO(model_path)

results = model(image_path)[0]
print(results.keypoints.xy[0][0])
for keypoint_indx, keypoint in enumerate(results.keypoints.xy[0]):
        x, y = int(keypoint[0]), int(keypoint[1])
        #print(x,y)
        cv2.putText(img, str(keypoint_indx), (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.circle(img, (x, y), 4, (255, 0, 255), -1)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

