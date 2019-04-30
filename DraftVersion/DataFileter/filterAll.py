import sys,os,shutil,re,cv2
from . import YcbCrFilter as mFileter
pj = os.path.join



ori_img_parent = '/home/deepcam/Data/categories/Tee'
new_img_parent = '/home/deepcam/Data/DF/Tee'

def main():
    cnt = 0

    if not os.path.exists(ori_img_parent):
        print ("ori path not exists : %s" % ori_img_parent)
        sys.exit()
    if not os.path.exists(new_img_parent):
        print ('new path not exists : %s\n create ' % new_img_parent)
        os.mkdir(new_img_parent)

    for dirpath, dirnames, filenames in os.walk(ori_img_parent):
        for file in filenames:
            if ( re.match('\w*.jpg$',file) ):
                fname = pj(dirpath,file)
                new_img,res = mFileter.main(fname,3)
                if res:
                    nname = pj(new_img_parent,str(cnt)+'.jpg')
                    cv2.imwrite(nname,new_img)
                    cnt += 1






if __name__ == '__main__':
    main()
