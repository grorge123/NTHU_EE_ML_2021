
import numpy as np
import sys
 
# 设置矩阵
def set_matrix():
    # 设置系数矩阵A
    matrix_a =np.mat([
        [2.0, 1.0, 2.0],
        [5.0, -1.0, 1.0],
        [1.0, -3.0, -4.0]],dtype=float)
    matrix_b = np.mat([5, 8, -4],dtype=float).T
    return matrix_a, matrix_b
 
 
def gauss_shunxu(mat):
    print(mat.shape)
    for i in range(0,mat.shape[0]-1):
        if mat[i,i] == 0:
            continue
        else:
            print(mat[i+1:,:].shape, (mat[i+1:,i]/mat[i,i]).shape, mat[i,:].shape)
            mat[i+1:,:] = mat[i+1:,:] - (mat[i+1:,i]/mat[i,i])*mat[i,:]
    return mat
 
def huidai(mat):
    x = np.mat(np.zeros(mat.shape[0],dtype=float))
    n = x.shape[1]-1
    print(mat.shape, x[0,n].shape, (mat[n,n+1]/mat[n,n]).shape)
    x[0,n] = mat[n,n+1]/mat[n,n]
    for i in range(n):
        n -= 1
        x[0,n] = (mat[n,mat.shape[1]-1] - np.sum(np.multiply(x[0, n+1:], mat[n,n+1:mat.shape[1]-1])))/mat[n,n]
    return x
 
 
if __name__ == "__main__":
    # 增广矩阵m为系数矩阵A加上列矩阵b
    m = np.hstack(set_matrix())            #按列合并：vstack()   按行合并：hstack()
    print('原矩阵：')
    print(m)
 
    # 顺序消去过程
    m1 = gauss_shunxu(m)
    print ('\n上三角矩阵：')
    print (m1)
 
    # 回带过程
    x = huidai(m1)
    print (x.shape, x)