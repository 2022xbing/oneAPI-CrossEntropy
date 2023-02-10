# oneAPI-CrossEntropy
并行计算交叉熵
三维输入x[K][M][N]，K=128，M=32，N=8192，其中K为batchsize，M为category，N为feature，二维坐标数组mask[K]]N]，权重数组weight[K][N]。K维度相互独立，在每一个面做上述二维的cross entropy计算。
最终结果为二维数组loss[K][N]。分别使用GPU和CPU进行计算，并比较计算结果和运行时间。
然后简单移植使用交叉熵作为损失函数的Pytorch程序。
