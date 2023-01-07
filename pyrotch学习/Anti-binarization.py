from PIL import Image

# pixel_number=20
#
# weight = pixel_number
# height = pixel_number
#
# # file_in = "C:/Users/王瑞/Desktop/Python Saving/pyrotch学习/random/randomint.txt"
# # file_out = "C:/Users/王瑞/Desktop/Python Saving/pyrotch学习/randomint.png"
#
# for i in range(100):
#     file_in="C:/Users/王瑞/Desktop/Python Saving/pyrotch学习/random20-20-txt/randomint{}.txt".format(i)
#     file_out = "C:/Users/王瑞/Desktop/Python Saving/pyrotch学习/random20-20-jpg/randomint{}.jpg".format(i)
#     with open(file_in) as f:
#         content = f.read()
#         newIm = Image.new('RGB', (weight, height), 'white')
#         white = (255, 255, 255)
#         black = (0, 0, 0)
#
#         for i in range(height):
#             for j in range(weight):
#                 if (content[i * weight + j] == '1'):
#                     newIm.putpixel((i, j), black)
#                 else:
#                     newIm.putpixel((i, j), white)
#
#         newIm.save(file_out)

#单个
pixel_number=20

weight = pixel_number
height = pixel_number


file_in="C:/Users/王瑞/Desktop/Python Saving/pyrotch学习/wr模型预测形状.txt"
file_out = "C:/Users/王瑞/Desktop/Python Saving/pyrotch学习/wr模型预测形状.jpg"
with open(file_in) as f:
    content = f.read()
    newIm = Image.new('RGB', (weight, height), 'white')
    white = (255, 255, 255)
    black = (0, 0, 0)

    for i in range(height):
        for j in range(weight):
            if (content[i * weight + j] == '1'):
                newIm.putpixel((i, j), black)
            else:
                newIm.putpixel((i, j), white)

    newIm.save(file_out)
