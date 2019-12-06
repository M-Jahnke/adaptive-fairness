import numpy as np

def compareMatrixWithScalar(m, s):
    return [True if m[i] == s else False for i in range(0, np.size(m, 0))]