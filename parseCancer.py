#!/usr/bin/env python

import re
import numpy as np
import cv2

def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))


if __name__ == "__main__":
    from matplotlib import pyplot
    ind = 1
    out = open("../output.txt","w+")
    out.truncate(0)

    outM = np.array([], dtype=np.str).reshape(0,8+(256*256))
    print(outM)
    with open('../miadata.txt','r+') as f:
        for line in f:
            old = line
            noCoords = re.sub(r'(?<!\S)\d+(?!\S)','',old)
            addXAfterNORM = re.sub(r'(NORM)', r'\1 X', noCoords)
            noNote = re.sub(r'(\*NOTE 3\*)','', addXAfterNORM)
            remvFat = re.sub(r'(\w+(?=\s+NORM)|\w+(?=\s+CALC)|\w+(?=\s+CIRC)|\w+(?=\s+SPIC)|\w+(?=\s+MISC)|\w+(?=\s+ARCH)|\w+(?=\s+ASYM))', '', noNote)
            remExtraSpaces = re.sub(' +',' ',remvFat)
            out.write(remExtraSpaces)
            row = np.array(map(str, remExtraSpaces.split()))

            p = re.compile(r'(\w+(?=\s+NORM)|\w+(?=\s+CALC)|\w+(?=\s+CIRC)|\w+(?=\s+SPIC)|\w+(?=\s+MISC)|\w+(?=\s+ARCH)|\w+(?=\s+ASYM))')
            imageName = p.search(remExtraSpaces)
            image = read_pgm('../all-mias/{}.pgm'.format(imageName.group(1)), byteorder='<')
            resized = cv2.resize(image, (256,256), interpolation = cv2.INTER_AREA)
            flat = resized.flatten()

            characteristics = np.zeros(8)

            if row[1] == 'CALC':
                characteristics[0] = 1
            elif row[1] == 'CIRC':
                characteristics[1] = 1
            elif row[1] == 'SPIC':
                characteristics[2] = 1
            elif row[1] == 'MISC':
                characteristics[3] = 1
            elif row[1] == 'ARCH':
                characteristics[4] = 1
            elif row[1] == 'ASYM':
                characteristics[5] = 1
            elif row[1] == 'NORM':
                characteristics[6] = 1
            else:
                print("Parse ERROR, row[1]: " + str(row[1]))


            if row[2] == 'B':
                characteristics[7] = 1

            row = np.insert(flat, 0, characteristics)

            outM = np.vstack([outM,row])

            print(imageName.group(1) + " DONE")
        f.close()

    np.savetxt('../matrix.txt', outM, delimiter=' ', fmt="%s") 

    pyplot.imshow(image, pyplot.cm.gray)
    pyplot.show()

