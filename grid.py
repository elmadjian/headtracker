import cv2
import numpy as np

class Grid():
    '''
    A boolean matrix indicating which element of the grid is
    being intersected by the head frame
    Default = 32 x 32 grid
    '''

    def __init__(self, width, height, blk_w=20, blk_h=15):
        self.col = width
        self.lin = height
        self.w  = blk_w
        self.h  = blk_h
        sw = int(self.col/blk_w)
        sh = int(self.lin/blk_h)
        self.grid = np.zeros((sh,sw), np.bool)

    
    def __convert_to_text(self, grid):
        grid = grid.astype(int)
        grid_text = ""
        for i in grid:
            grid_text += str(i)
        return grid_text


    def get_intersection(self, head_frame):
        self.grid.fill(0)
        x1,y1,x2,y2 = head_frame
        for i in range(y1,y2, self.h//2):
            for j in range(x1,x2, self.w//2):
                lin = int(np.round(i/self.h))
                col = int(np.round(j/self.w))
                self.grid[lin,col] = True
        grid = np.ravel(self.grid.copy())
        return grid
