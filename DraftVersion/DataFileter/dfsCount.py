import numpy as np
import sys


# Depth of python recursion is 999 with default
sys.setrecursionlimit(100000000)


visit = np.array([])
img = np.array([])
ROWS = 0
COLS = 0
V = 0


'''
visit direction for dfs
'''
dire = [[0,1],[-1,0],[0,-1],[1,0]]


def dfs(x,y):
    for i in range(4):
        tx = x + dire[i][0]
        ty = y + dire[i][1]
        if tx <0 or tx > ROWS-1 or ty<0 or ty > COLS-1:
            continue


        if visit[tx][ty] == 0 and img[tx][ty] == V:
            visit[tx][ty] = 1
            dfs(tx,ty)



def main(_img,v):
    global ROWS, COLS, visit, V,img

    visit = np.array([])
    img = np.array([])
    ROWS = 0
    COLS = 0
    V = 0

    img = np.array(_img)
    ROWS = img.shape[0]
    COLS = img.shape[1]
    V = v

    visit = np.zeros((ROWS,COLS))

    cnt = 0
    for r in range(ROWS):
        for c in range(COLS):
            if visit[r][c] == 0 and img[r][c] == V:
                visit[r][c] = 1
                cnt += 1
                dfs(r,c)

    return cnt
