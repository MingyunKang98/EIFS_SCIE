import albumentations as A
import cv2

transform = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])

image = cv2.imread("/path/to/image.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mask = cv2.imread("/path/to/mask.png")

mask_1 = cv2.imread("/path/to/mask_1.png")
mask_2 = cv2.imread("/path/to/mask_2.png")
mask_3 = cv2.imread("/path/to/mask_3.png")
masks = [mask_1, mask_2, mask_3]

transformed = transform(image=image, mask=mask)
'''
mask=mask 부분만 추가하면 똑같이 변형됩니다.
'''
transformed_image = transformed['image']
transformed_mask = transformed['mask']