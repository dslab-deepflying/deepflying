#! /usr/bin/python

import os
import sys
import re
import pandas
pj = os.path.join

src_root_path = '/home/deepcam/Data/DF'
tar_path = '/home/deepcam/Data/'



def get_paths(src = '',txtname = ''):

    files = []

    for dirpath, dirnames, filenames in os.walk(src):
        for file in filenames:
            if ( re.match('\w*.jpg$',file) ):
		nj = pj (tar_path,dirpath,file)
		#os.rename( nj,(str)(nj).split('.jpg')[0]+'.png')
                files.append( pj (dirpath,file) )

    if not os.path.exists(tar_path) :
        print('Tar path not exists , create ')
        os.mkdir(tar_path)

    pandas.Series(files).to_csv(tar_path+'/' + txtname,index=None)


def main():

    cnt = 0

    for dirpath,dirnames,filenames in os.walk(src_root_path):
        break

    for src in dirnames:
        cnt += 1
        sys.stdout.write('\r[%.2f%%] %s' % (cnt * 100.0 / dirnames.__len__(), src ))
        sys.stdout.flush()

        if os.path.exists(dirpath+'/'+src+'.txt'):

            os.remove(dirpath+'/'+src+'.txt')
        get_paths(dirpath + '/'+ src, src+'.txt')

    print ('\t\n')

main()
