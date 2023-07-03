import numpy as np
import faiss
import time
d = 64                           # dimension
nb = 100000                      # database size
nq = 10000                       # nb of queries
np.random.seed(1234)             # make reproducible
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.
index = faiss.IndexFlatL2(d)   # build the index
print(index.is_trained)
index.add(xb)                  # add vectors to the index
print(index.ntotal)
k = 4                          # we want to see 4 nearest neighbors
D, I = index.search(xb[:5], k) # sanity check
print('Size of I:',np.shape(I))
print('Size of D:',np.shape(D))
print(I)
print(D)
D, I = index.search(xq, k)     # actual search
print('Size of I:',np.shape(I))
print('Size of D:',np.shape(D))
print(I[:5])                   # neighbors of the 5 first queries
print(I[-5:])                  # neighbors of the 5 last queries



# 矩阵向量l2距离计算
def dist(xq, xb):
    xq2 = np.sum(xq**2, axis=1, keepdims=True)
    xb2 = np.sum(xb**2, axis=1, keepdims=True)
    xqxb = np.dot(xq,xb.T)
    # faiss在计算l2距离的时候没有开方，加快速度
    # return np.sqrt(xq2 - 2 * xqxb + xb2.T)
    return xq2 - 2 * xqxb + xb2.T

# 获取结果
def get_result(dst,k = 5):
    D = np.sort(dst)[:,:k]
    I = np.argsort(dst)[:,:k]
    return D,I 

# 开始时间
start = time.time()
dst = dist(xq,xb)
D_, I_ = get_result(dst,k)
# 结束时间
end = time.time()
# 这里用的是一台很弱的电脑，速度慢正常。
# print("耗时{}s".format(end-start))

# 前五个查询向量的检索结果 
print(I[:5])        
print('---分割线---')
# 最后五个查询向量的检索结果
print(I[-5:])

