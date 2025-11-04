import cv2
from camqa import CamQA

qa = CamQA(short_side=720, ema=0.2, mode='auto')  # 'day' / 'night' sabitleyebilirsin
frame = cv2.imread("test/images/img1.jpg")
res = qa.analyze(frame)
print(res["mode"])
print(res["scores"])  # {'sharpness': ..., 'brightness': ..., 'color_intensity': ..., 'lens_cleanliness': ...}
