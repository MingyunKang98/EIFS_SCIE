from ultralytics import YOLO


###########################################################################


# from PIL import Image
# import cv2
#
#
#
# model = YOLO("yolov8s-seg.pt")
# # accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
# results = model.predict(source="0")
# results = model.predict(source="folder", show=True)  # Display preds. Accepts all YOLO predict arguments
#
# # from PIL
# im1 = Image.open("bus.jpg")
# results = model.predict(source=im1, save=True)  # save plotted images
#
# # from ndarray
# im2 = cv2.imread("bus.jpg")
# results = model.predict(source=im2, save=True, save_txt=True)  # save predictions as labels
#
# # from list of PIL/ndarray
# results = model.predict(source=[im1, im2]
#

###########################################################################





# if __name__ == '__main__':
#     model = YOLO("runs/segment/train7_51/weights/best.pt")
#     model.train(data='./ultralytics/datasets/data.yaml', epochs=200, patience=100)
#     predict = model.predict("./KakaoTalk_20230213_173129749_jpg.rf.f136f9cb13bf03c39c6f25d21035d84c.jpg", save=True, save_txt=True, conf=0.43)

import os


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    model = YOLO("./runs/segment/train7/weights/best.pt")
    model.train(data='./datasets/data.yaml', epochs=50, patience=300)
    # predict = model.predict("./20230509_114835.jpg", save=True, save_txt=True, conf=0.2)





########################################################################
# # Train
# model = YOLO("yolov8n.pt") # pass any model type
# model.train(epochs=5)
#
# # Validation
# model = YOLO("model.pt")
# # It'll use the data yaml file in model.pt if you don't set data.
# model.val()
# # or you can set the data you want to val
# model.val(data="coco128.yaml")
#
# # Predict
# model = YOLO("model.pt")
# # accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
# results = model.predict(source="0")
# results = model.predict(source="folder", show=True) # Display preds. Accepts all YOLO predict arguments
#
# # from PIL
# im1 = Image.open("bus.jpg")
# results = model.predict(source=im1, save=True)  # save plotted images
#
# # from ndarray
# im2 = cv2.imread("bus.jpg")
# results = model.predict(source=im2, save=True, save_txt=True)  # save predictions as labels
#
# # from list of PIL/ndarray
# results = model.predict(source=[im1, im2])
#
#
#


















