#!/usr/bin/python

import os,sys
import styleTransfer as st
from PIL import Image

style_list = [
#    'dh1.png',
#    'dh1.1.png',
#    'dh2.png',
#    'dh3.jpeg',
#    'dh3.1.jpg',
#    'dh4.jpg',
#    'dh4.1.jpg',
#    'dh5.jpg',
#    'dh6.jpg'
]

def main():

    for dirpath,dirnames,filenames in os.walk('style_image'):
        style_list = filenames
        break

    filenames = []
    for dirpath,dirnames,filenames in os.walk('content_image'):
        break  	

    cnt = 0
    s_id = 0
    # print filenames
    # sys.exit()
    for f in filenames:
        st.target_image_path = 'content_image/'+f
        st.style_reference_path = 'style_image/'+ style_list[s_id]

        tar_size = Image.open(st.target_image_path).size
	st.img_width = tar_size[0]
	st.img_height = tar_size[1]

        if s_id < style_list.__len__()-1:
            s_id += 1
        else :
            s_id = 0

        st.main()
        print cnt
        cnt += 1

if __name__=='__main__':
    main()
