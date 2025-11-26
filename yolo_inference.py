from ultralytics import YOLO

model = YOLO("models/key_point/best.pt")  # load a pretrained YOLOv8n model

results= model.predict('input_videos/test_2.mp4', save=True)  # predict on an image

print(results)  # print results to console
print ('================================')

for box in results[0].boxes:
    print(box)