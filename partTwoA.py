import numpy as np
import partOneA as partOne
import matplotlib.pyplot as plt
def acquire_partTwodata():

    testX = np.loadtxt("./PA-1-data-text/PA-1-data-text/count_data_testx.txt")
    testY = np.loadtxt("./PA-1-data-text/PA-1-data-text/count_data_testy.txt")
    trainX = np.loadtxt("./PA-1-data-text/PA-1-data-text/count_data_trainx.txt")
    trainY = np.loadtxt("./PA-1-data-text/PA-1-data-text/count_data_trainy.txt")
    thtrue = np.loadtxt("./PA-1-data-text/PA-1-data-text/count_data_ym.txt")

    return testX,testY,trainX,trainY

def mae_error(polypredic_y,polyy):
    predic_result=np.array(polypredic_y)
    true_polyy=np.array(polyy)
    error_total = (abs(predic_result - true_polyy)).sum()/(polypredic_y.shape[0]*polypredic_y.shape[1])
    return  error_total

if __name__=='__main__':
    testX,tran_testY,trainX,tran_trainY=acquire_partTwodata();
    korder=8
    testY = tran_testY.reshape(len(tran_testY), 1)
    trainY = tran_trainY.reshape(len(tran_trainY), 1)
    # print(trainX.shape)
    # print(trainY.shape)
    # print(testX.shape)
    # print(testY.shape)
    # least_squares(LS)
    LS_theta = partOne.least_squ_theta(trainX, trainY)
    print('testX',testX.shape)
    print('LS_theta', LS_theta.shape)
    LS_result = partOne.least_squ_predi(testX, LS_theta)
    LS_error = partOne.error(LS_result, testY)
    LS_mae_error = mae_error(LS_result, testY)
    print('LS_result',LS_result.shape)
    print('LS_MSE_error:', LS_error)
    print('LS_MAE_error:', LS_mae_error)
    plt.figure(1)
    plt.title('least_squares(LS)')
    plt.xlabel('x-value')
    plt.ylabel('y-label')
    plt.plot(testX[0],LS_result,'.-')
    plt.plot(testX[0], testY, '.-')
    # plt.plot(trainX[0], trainY,'*')
    plt.legend(["test", "true"])
    # Regularized LS(RLS)
    RLS_theta = partOne.RLS_theta(trainX, trainY)
    RLS_result = partOne.RLS_predi(testX, RLS_theta)
    RLS_error = partOne.error(RLS_result, testY)
    RLS_mae_error = mae_error(RLS_result, testY)
    print('RLS_MSE_error:', RLS_error)
    print('RLS_MAE_error:', RLS_mae_error)
    plt.figure(2)
    plt.title('Regularized LS(RLS)')
    plt.xlabel('x-value')
    plt.ylabel('y-label')
    plt.plot(testX[0], RLS_result, '.-')
    plt.plot(testX[0], testY, '.-')
    # plt.plot(trainX[0], trainY,'*')
    plt.legend(["test", "true"])
    # L1-regularized LS(LASSO)
    LASSO_theta = partOne.LASSO_Theta(trainX, trainY, 8)
    LASSO_result = partOne.LASSO_predi(testX, LASSO_theta)
    LASSO_error = partOne.error(LASSO_result, testY)
    LASSO_mae_error = mae_error(LASSO_result, testY)
    print('LASSO_MSE_error:', LASSO_error)
    print('LASSO_MAE_error:', LASSO_mae_error)
    plt.figure(3)
    plt.title('L1-regularized LS(LASSO)')
    plt.xlabel('x-value')
    plt.ylabel('y-label')
    plt.plot(testX[0], LASSO_result, '.-')
    plt.plot(testX[0], testY, '.-')
    # plt.plot(trainX[0], trainY,'*')
    plt.legend(["test", "true"])
    # Robust regression(RR)
    RR_theta = partOne.RR_Theta(trainX[1], trainY, trainX, korder)
    # print(RR_theta.shape)
    RR_result = partOne.RR_predi(testX, RR_theta)
    RR_error = partOne.error(RR_result, testY)
    RR_mae_error = mae_error(RR_result, testY)
    print('RR_MSE_error:', RR_error)
    print('RR_MAR_error:', RR_mae_error)
    plt.figure(4)
    plt.title('Robust regression(RR)')
    plt.xlabel('x-value')
    plt.ylabel('y-label')
    plt.plot(testX[0], RR_result, '.-')
    plt.plot(testX[0], testY, '.-')
    # plt.plot(trainX[0], trainY,'*')
    plt.legend(["test", "true"])
    # Bayesian regression (BR)
    BR_cov, BR_mean = partOne.posterior_BR(trainX, trainY)
    average, fangcha = partOne.BR_predi(testX, BR_cov, BR_mean)
    BR_polyy = np.random.normal(average, np.sqrt(np.abs(fangcha)), size=None)
    # print('average.shape:', average.shape)
    # print('fangcha.shape:', fangcha.shape)
    # print('BR_pred.shape:', BR_polyy.shape)
    BR_error = partOne.error(BR_polyy, testY)
    BR_mae_error = mae_error(BR_polyy, testY)
    print('BR_MSE_error', BR_error)
    print('BR_MAE_error:', RR_mae_error)
    plt.figure(5)
    plt.title('Bayesian regression (BR)')
    plt.xlabel('x-value')
    plt.ylabel('y-label')
    plt.plot(testX[0], BR_polyy, '.-')
    plt.plot(testX[0], testY, '.-')
    # plt.plot(trainX[0], trainY, '*')
    plt.legend(["test", "true"])
    plt.show()