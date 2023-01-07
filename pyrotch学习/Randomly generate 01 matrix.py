import random
import xlwt

random.seed()

pixel_number=20

#file_name="randomint.txt"
for i in range(100):
    f = open("1/randomint-with-n{}.txt".format(i), "w")
    s = open("1/randomint-without-n{}.txt".format(i), "w")
    # for i in range(8):
    #     f.write('\n')
    for i in range(pixel_number * pixel_number):
        number = random.randint(0, 1)
        f.write(str(number))
        s.write(str(number))
        if i % pixel_number == pixel_number-1:
            f.write('\n')

