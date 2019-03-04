# coding:utf-8
from PIL import Image, ImageEnhance
import pytesseract
import numpy as np
# 上面都是导包，只需要下面这一行就能实现图片文字识别
im = Image.open('test.jpg')


enh_con = ImageEnhance.Contrast(im)
contrast = 1.5
image_contrasted = enh_con.enhance(contrast)
# image_contrasted.show()


# 增强亮度
enh_bri = ImageEnhance.Brightness(image_contrasted)
brightness = 1.5
image_brightened = enh_bri.enhance(brightness)
# image_brightened.show()


# 增强对比度
enh_col = ImageEnhance.Color(image_brightened)
color = 1.5
image_colored = enh_col.enhance(color)
# image_colored.show()

# 增强锐度
enh_sha = ImageEnhance.Sharpness(image_colored)
sharpness = 3.0
image_sharped = enh_sha.enhance(sharpness)
# image_sharped.show()


"""
his = im.histogram()
values = {}

for i in range(256):
    values[i] = his[i]

for j,k in sorted(values.items(),key=lambda x:x[1],reverse = True):
    print(j,k)
 
im2 = Image.new("P", im.size, 255)

for x in range(im.size[1]):
    for y in range(im.size[0]):
        pix = im.getpixel((y,x))
        if pix == 219:
            # these are the numbers to get
            im2.putpixel((y,x),0)
im2.show()           

"""

# 灰度处理部分
im = image_contrasted.convert("L")
text = pytesseract.image_to_string(
    im, lang='chi_sim').strip()  # 使用image_to_string识别验证码
print(text)
