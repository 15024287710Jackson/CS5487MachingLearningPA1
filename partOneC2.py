import numpy as np
from cvxopt import matrix
from cvxopt import solvers
import matplotlib.pyplot as plt
import partOneA as partOne
import random

def xunhun(per):
    [sampleX1, sampleY1, polyx, polyy] = partOne.acquire_date()
    num_size = round(sampleX1.size * per)
    location = random.sample(range(1, sampleX1.size), num_size)
    selection_sampleX = []
    selection_sampleY = []
    for loca in location:
        selection_sampleX.append(sampleX1[loca])
        selection_sampleY.append(sampleY1[loca])
    sampleX = np.array(selection_sampleX).reshape(len(selection_sampleX), 1)
    sampleY = np.array(selection_sampleY).reshape(len(selection_sampleY), 1)
    korder = 5
    # least_squares(LS)
    aftermat_sampleX = partOne.theta_mat(sampleX, korder)
    LS_theta = partOne.least_squ_theta(aftermat_sampleX, sampleY)
    aftermat_polyx = partOne.theta_mat(polyx, korder)
    LS_result = partOne.least_squ_predi(aftermat_polyx, LS_theta)
    LS_error = partOne.error(LS_result, polyy)
    # print('LS_error:', LS_error)
    plt.figure(1)
    plt.title('least_squares(LS)')
    plt.xlabel('x-value')
    plt.ylabel('y-label')
    plt.legend()
    plt.plot(polyx, LS_result, '.-')
    plt.plot(sampleX, sampleY, '*')
    # Regularized LS(RLS)
    RLS_theta = partOne.RLS_theta(aftermat_sampleX, sampleY)
    RLS_result = partOne.RLS_predi(aftermat_polyx, RLS_theta)
    RLS_error = partOne.error(RLS_result, polyy)
    # print('RLS_error:', RLS_error)
    plt.figure(2)
    plt.title('Regularized LS(RLS)')
    plt.xlabel('x-value')
    plt.ylabel('y-label')
    plt.legend()
    plt.plot(polyx, RLS_result, '.-')
    plt.plot(sampleX, sampleY, '*')
    # L1-regularized LS(LASSO)
    LASSO_theta = partOne.LASSO_Theta(aftermat_sampleX, sampleY, 5)
    LASSO_result = partOne.LASSO_predi(aftermat_polyx, LASSO_theta)
    LASSO_error = partOne.error(LASSO_result, polyy)
    # print('LASSO_error:', LASSO_error)
    plt.figure(3)
    plt.title('L1-regularized LS(LASSO)')
    plt.xlabel('x-value')
    plt.ylabel('y-label')
    plt.legend()
    plt.plot(polyx, LASSO_result, '.-')
    plt.plot(sampleX, sampleY, '*')
    # Robust regression(RR)
    RR_theta = partOne.RR_Theta(sampleX, sampleY, aftermat_sampleX, korder)
    # print(RR_theta.shape)
    RR_result = partOne.RR_predi(aftermat_polyx, RR_theta)
    RR_error = partOne.error(RR_result, polyy)
    # print('RR_error:', RR_error)
    plt.figure(4)
    plt.title('Robust regression(RR)')
    plt.xlabel('x-value')
    plt.ylabel('y-label')
    plt.legend()
    plt.plot(polyx, RR_result, '.-')
    plt.plot(sampleX, sampleY, '*')
    # plt.show()
    # Bayesian regression (BR)
    BR_cov, BR_mean = partOne.posterior_BR(aftermat_sampleX, sampleY)
    average, fangcha = partOne.BR_predi(aftermat_polyx, BR_cov, BR_mean)
    BR_polyy = np.random.normal(average, np.sqrt(np.abs(fangcha)), size=None)
    # print('average.shape:', average.shape)
    # print('fangcha.shape:', fangcha.shape)
    # print('BR_pred.shape:', BR_polyy.shape)
    BR_error = partOne.error(BR_polyy, polyy)
    # print('BR_error:', BR_error)
    plt.figure(5)
    plt.title('Bayesian regression (BR)')
    plt.xlabel('x-value')
    plt.ylabel('y-label')
    plt.legend()
    plt.plot(polyx, BR_polyy, '.')
    plt.plot(sampleX, sampleY, '*')
    # plt.show()
    return LS_error,RLS_error,LASSO_error,RR_error,BR_error


if __name__=='__main__':
    LS_error_ex=[]
    RLS_error_ex = []
    LASSO_error_ex = []
    RR_error_ex = []
    BR_error_ex = []
    per=0.75
    for i in range(100):
        LS_error,RLS_error,LASSO_error,RR_error,BR_error=xunhun(per)
        LS_error_ex.append(LS_error)
        RLS_error_ex.append(LS_error)
        LASSO_error_ex.append(LASSO_error)
        RR_error_ex.append(RR_error)
        BR_error_ex.append(BR_error)

    print('LS_error_ex',np.array(LS_error_ex).mean())
    print('RLS_error_ex',np.array(RLS_error_ex).mean())
    print('LASSO_error_ex',np.array(LASSO_error_ex).mean())
    print('RR_error_ex',np.array(RR_error_ex).mean())
    print('BR_error_ex',np.array(BR_error_ex).mean())