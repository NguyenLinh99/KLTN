import io
import json
import cv2
import numpy as np
from PIL import Image
from flask import Flask, jsonify, request
import base64
import re
from flask_cors import CORS
from tool.config import Cfg
import torch
from tool.predictor import Predictor

app = Flask(__name__)
CORS(app)

config = Cfg.load_config_from_name('vgg_transformer')
# config = Cfg.load_config_from_name('vgg_seq2seq')
config['vocab'] = 'aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!"&\'()+,-./:;= '
config['weights'] = 'weights/transformerocr.pth'
# config['weights'] = '/home/v000354/Downloads/transformerocr_ben.pth'
config['device'] = 'cpu'
config['predictor']['beamsearch']=False
device = config['device']

detector = Predictor(config)
img = Image.open("0_mcocr_public_145014mpsud.jpg")
# img.show()
s = detector.predict(img)
print(s)

exit()
def convert_image(img):
    # comvert vector
    # img = cv2.imread(full_path)

    pixel_values = img.reshape((-1,3))
    pixel_values = np.float32(pixel_values)

    # kmean
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    ret,labels,center=cv2.kmeans(pixel_values,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    # convert to image
    labels = labels.flatten()
    res = center[labels]
    # print(res)
    res[labels == 1] = [0,0,0]

    if res[0][0] == 0:
        # print("aaaa",full_path)
        res[labels == 1] = [255,255,255]
        res[labels == 0] = [0,0,0]
    else:
        # print("hhhh",full_path)
        res[labels == 1] = [0,0,0]
        res[labels == 0] = [255,255,255]


  
    res2 = res.reshape((img.shape))
    # print("hhahaha", img.shape)
    cv2.imshow("hehehe", res2)
    cv2.waitKey()
    # print("hhhihhi", full_path)
    # cv2.imwrite("/home/v000354/Downloads/data_chu/data_dung/data_dung/result/"+full_path.split("/")[-1],res2)
    return res2

def template(image):
    # img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(image, cv2.IMREAD_GRAYSCALE)
    template = cv2.imread('/home/v000354/Documents/Linh/invoice_ocr/OCR/App/template.jpg',0)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        # print("hehhehee", pt)
        image[pt[1]:pt[1]+h,pt[0]:pt[0]+w,:] = 255
        # cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
    return image

def del_space(image):
    original = image.copy()
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(image, (25,25), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Perform morph operations, first open to remove noise, then close to combine
    noise_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, noise_kernel, iterations=2)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13,13))
    close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, close_kernel, iterations=3)

    # Find enclosing boundingbox and crop ROI
    coords = cv2.findNonZero(close)
    x,y,w,h = cv2.boundingRect(coords)
    # cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
    crop = original[y:y+h, x:x+w+3]
    return crop

def predict_crop(img):
    # img = Image.open(image_bytes)
    s = detector.predict(img)
    return s

img = Image.open("data/train/1_mcocr_warmup_956dfb1e5168a778958ffe90ba4df9f3_00171.jpg")
img.show()
print(predict_crop(img))


exit()

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        image_data = re.sub('^data:image/.+;base64,', '', str(request.form["data"]))

        result = predict_crop(image_bytes=base64.b64decode(image_data))
        print("linhlinh", result)
        return jsonify({'predicted': result})


if __name__ == '__main__':
    app.run(debug=True)

# pil_image = Image.open("/home/v000354/Documents/Linh/invoice_ocr/OCR/Model/data/train_data/noise_6.jpg")
# pil_image.show()
# img = np.array(pil_image)
# img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
# cv2.imshow("hihihi",img)
# for i in range(0,img.shape[0]):
#     for j in range(0,img.shape[1]):
#         k = img[i, j]
#         if k[2]>160 :
#         # k[1] = 255
#             k[0] = 255 
#             k[1] = 255 
#             k[2 ] = 255 
 
# cv2.imshow("haha",img)
# cv2.waitKey()

# # kernel = np.ones((5,5),np.uint8)
# # erosion = cv2.erode(img,kernel)
# # cv2.imshow("hehhehe", erosion)
# # # cv2.imwrite("huhuhuhu.jpg", img)
# # # img = del_space(img)
# # # print("hehhe", img.shape)


# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# # gray1 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,15,11)
# ret3,gray1 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# gray = cv2.blur(gray1,(3,3))
# image=Image.fromarray(gray)
# cv2.imshow("hehhehe", gray)
# cv2.waitKey()
# image=Image.fromarray(gray)
# s = detector.predict(pil_image) 
# print("jejhehe", s)