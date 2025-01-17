from ultralytics import YOLO 

# yolov8 track players
# model = YOLO('yolov8x')

# yolov5 track ball
# model = YOLO('models/yolo5_last.pt')
model = YOLO('yolov8x')

# model.predict('input_videos/image.png', save=True)

result = model.track('input_videos/input_video.mp4', conf=0.2, save=True)

# result = model.predict('input_videos/input_video.mp4', conf=0.2, save=True)
# print(result)
# print("boxes:")
# for box in result[0].boxes:
#     print(box)