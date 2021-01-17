import numpy as np
from cvxopt import matrix
from cvxopt import solvers
import matplotlib.pyplot as plt
import cmath
from scipy import stats

def acquire_date():

    sampX = np.loadtxt("./PA-1-data-text/PA-1-data-text/polydata_data_sampx.txt")
    sampY = np.loadtxt("./PA-1-data-text/PA-1-data-text/polydata_data_sampy.txt")
    polyX = np.loadtxt("./PA-1-data-text/PA-1-data-text/polydata_data_polyx.txt")
    polyY = np.loadtxt("./PA-1-data-text/PA-1-data-text/polydata_data_polyy.txt")
    thtrue = np.loadtxt("./PA-1-data-text/PA-1-data-text/polydata_data_thtrue.txt")

    sampleX = sampX.reshape(len(sampX),1)
    sampleY = sampY.reshape(len(sampY), 1)
    polyx = polyX.reshape(len(polyX), 1)
    polyy = polyY.reshape(len(polyY), 1)

    return sampleX,sampleY,polyx,polyy
#process the data
def theta_ver(sampleX,k):
    return np.array([sampleX**m for m in range(k+1)]).reshape(k+1,1)

def theta_arr(sampleX,k):
    return np.array([theta_ver(n,k) for n in sampleX]).transpose()

#matrix of samplyx
def theta_mat(data,k):
    return np.matrix(theta_arr(data,k))

#least_squares(LS)--theta
def least_squ_theta(sampleX,sampleY):
    return np.dot(np.matrix(np.dot(sampleX,sampleX.transpose())).I,sampleX).dot(sampleY)

#least_squares(LS)--prediction
def least_squ_predi(polyX,least_theta):
    return np.dot(polyX.transpose(),least_theta)

#Regularized LS(RLS)
def RLS_theta(aftermat_sampleX,sampleY): #k*1
    lamda=1
    return (np.matrix(np.dot(aftermat_sampleX, aftermat_sampleX.transpose()) + \
                      lamda * np.identity(len(aftermat_sampleX))).I).dot(aftermat_sampleX).dot(sampleY)
def RLS_predi(polyX,least_theta):
    return np.dot(polyX.transpose(),least_theta)

#L1-regularized LS(LASSO)
def LASSO_Theta(aftermat_sampleX,sampley,k):
    lamda2=1
    samllthate_mat_fang = np.dot(aftermat_sampleX, aftermat_sampleX.transpose())
    samllthate_mat_sampley = np.dot(aftermat_sampleX,sampley)
    samllthate_mat_yaugre = np.concatenate((samllthate_mat_sampley, -1 * samllthate_mat_sampley), axis=0)
    Matleft = np.concatenate((samllthate_mat_fang, -1 * samllthate_mat_fang), axis=0)
    Matright = np.concatenate((-1 * samllthate_mat_fang, samllthate_mat_fang), axis=0)
    Mat_H = np.concatenate((Matleft, Matright), axis=1)
    f = lamda2 * np.ones((len(samllthate_mat_yaugre), 1)) - samllthate_mat_yaugre
    G = -1 * np.identity((len(Mat_H)))
    value = np.zeros((len(Mat_H), 1))
    LASSO_theta = solvers.qp(matrix(Mat_H), matrix(f), matrix(G), matrix(value))['x']
    return np.matrix(
        [LASSO_theta[i] - LASSO_theta[i + k+1] for i in range(int(len(LASSO_theta) / 2))]).transpose()

def LASSO_predi(aftermat_polyx, LASSO_Theta):
    return np.dot(aftermat_polyx.transpose(), LASSO_Theta)

#Robust regression(RR)
def RR_Theta(samplex,sampley,aftermat_sampleX,kthorder):
    ko=kthorder+1
    Connect = np.concatenate((np.zeros((ko, 1)), np.ones((len(samplex), 1))), axis=0)
    Mat_left = np.concatenate((-1 * aftermat_sampleX.transpose(), aftermat_sampleX.transpose()), axis=0)
    Mat_right = np.concatenate((-1 * np.identity(len(samplex)), -1 * np.identity(len(samplex))), axis=0)
    Mat_A = np.concatenate((Mat_left, Mat_right), axis=1)
    Mat_B = np.concatenate((-1 * sampley, sampley), axis=0)
    return solvers.lp(matrix(Connect), matrix(Mat_A), matrix(Mat_B))['x'][0:ko]

def RR_predi(aftermat_polyx, RR_theta):
    return np.dot(aftermat_polyx.transpose(), RR_theta)

# Bayesian regression(BR)
def posterior_BR(aftermat_sampleX,sampleY):
    alpha=1
    BR_var=5
    BR_cov = np.matrix((1 / alpha) * np.identity(len(aftermat_sampleX))+ (1 / BR_var) * np.dot(aftermat_sampleX,aftermat_sampleX.transpose())).I
    BR_mean = (1 / BR_var) * BR_cov.dot(aftermat_sampleX).dot(sampleY)
    return BR_cov, BR_mean

def BR_predi(aftermat_polyx, BR_cov, BR_mean):
    average = np.dot(aftermat_polyx.transpose(), BR_mean)
    fangcha = np.dot(aftermat_polyx.transpose(),BR_cov).dot(aftermat_polyx)
    return average, fangcha

def BR_Gaussian(value,average,fangcha):
        left = (1 / (np.sqrt(2 * np.pi* fangcha)))
        exp_value = np.exp(-(np.power(value - average, 2))/(2*fangcha))
        pro=left*exp_value
        return pro

def error(polypredic_y,polyy):
    predic_result=np.array(polypredic_y)
    true_polyy=np.array(polyy)
    error_total = ((predic_result - true_polyy)**2).sum()/(polypredic_y.shape[0]*polypredic_y.shape[1])
    return  error_total

# if __name__=='__main__':
#     [sampleX,sampleY,polyx,polyy]=acquire_date()
#     korder=5
#     # least_squares(LS)
#     aftermat_sampleX=theta_mat(sampleX, korder)
#     print('aftermat_sampleX:',aftermat_sampleX.shape)
#     print('sampleY:', sampleY.shape)
#     LS_theta=least_squ_theta(aftermat_sampleX,sampleY)
#     print('LS_theta:',LS_theta.shape)
#     aftermat_polyx=theta_mat(polyx, korder)
#     print('aftermat_polyx:', aftermat_polyx.shape)
#     LS_result=least_squ_predi(aftermat_polyx,LS_theta)
#     print('LS_result:', LS_result.shape)
#     LS_error=error(LS_result,polyy)
#     print('LS_error:',LS_error)
#     plt.figure(1)
#     plt.title('least_squares(LS)')
#     plt.xlabel('x-value')
#     plt.ylabel('y-label')
#     plt.legend()
#     plt.plot(polyx,LS_result,'.-')
#     plt.plot(sampleX, sampleY,'*')
#     # Regularized LS(RLS)
#     RLS_theta=RLS_theta(aftermat_sampleX,sampleY)
#     RLS_result = RLS_predi(aftermat_polyx, RLS_theta)
#     RLS_error = error(RLS_result, polyy)
#     print('RLS_error:', RLS_error)
#     plt.figure(2)
#     plt.title('Regularized LS(RLS)')
#     plt.xlabel('x-value')
#     plt.ylabel('y-label')
#     plt.legend()
#     plt.plot(polyx,RLS_result,'.-')
#     plt.plot(sampleX, sampleY,'*')
#     #L1-regularized LS(LASSO)
#     LASSO_theta=LASSO_Theta(aftermat_sampleX,sampleY,5)
#     LASSO_result=LASSO_predi(aftermat_polyx, LASSO_theta)
#     LASSO_error = error(LASSO_result, polyy)
#     print('LASSO_error:', LASSO_error)
#     plt.figure(3)
#     plt.title('L1-regularized LS(LASSO)')
#     plt.xlabel('x-value')
#     plt.ylabel('y-label')
#     plt.legend()
#     plt.plot(polyx, LASSO_result, '.-')
#     plt.plot(sampleX, sampleY, '*')
#     #Robust regression(RR)
#     RR_theta=RR_Theta(sampleX,sampleY,aftermat_sampleX,korder)
#     # print(RR_theta.shape)
#     RR_result=RR_predi(aftermat_polyx, RR_theta)
#     RR_error = error(RR_result, polyy)
#     print('RR_error:', RR_error)
#     plt.figure(4)
#     plt.title('Robust regression(RR)')
#     plt.xlabel('x-value')
#     plt.ylabel('y-label')
#     plt.legend()
#     plt.plot(polyx, RR_result, '.-')
#     plt.plot(sampleX, sampleY, '*')
#     # plt.show()
#     # Bayesian regression (BR)
#     BR_cov, BR_mean=posterior_BR(aftermat_sampleX, sampleY)
#     average, fangcha=BR_predi(aftermat_polyx, BR_cov, BR_mean)
#     BR_polyy=np.random.normal(average,np.sqrt(np.abs(fangcha)),size=None)
#     # print('average.shape:', average.shape)
#     # print('fangcha.shape:', fangcha.shape)
#     # print('BR_pred.shape:',BR_polyy.shape[0]*BR_polyy.shape[1])
#     BR_error=error(BR_polyy, polyy)
#     # print('BR_error:',BR_error)
#     plt.figure(5)
#     plt.title('Bayesian regression (BR)')
#     plt.xlabel('x-value')
#     plt.ylabel('y-label')
#     plt.legend()
#     plt.plot(polyx, BR_polyy, '.')
#     plt.plot(sampleX, sampleY, '*')
#     plt.show()
