import numpy as np
import random


class FFRLS_FF:
    def __init__(self, matrix_dim, lam, gamma, epsilon, beta, sigma, Ts):
        self.Ts = Ts        #采样时间
        self.lam = lam      #遗忘因子
        self.gamma = gamma      #鲁棒系数（性能边界）
        self.epsilon = epsilon      #误差能量窗口
        self.beta = beta            #误差放大系数
        self.sigma = sigma          #误差能量临界阈值
        self.matrix_dim = matrix_dim        #矩阵维数
        self.eye = np.eye(matrix_dim)
        self.theta_R = np.array([[1, 1, 1, 1, 1, 1, 1]]).reshape(-1, 1)         #RLS辨识参数结果θ_r
        self.theta_H = np.array([[1, 1, 1, 1, 1, 1, 1]]).reshape(-1, 1)         #HIF辨识参数结果θ_h
        self.theta_f = np.array([[1, 1, 1, 1, 1, 1, 1]]).reshape(-1, 1)         #融合辨识参数结果θ_final
        self.Po_R = np.eye(matrix_dim) * 1e6        #RLS信息自逆矩阵
        self.Po_H = np.eye(matrix_dim)              #HIF协方差矩阵
        self.V_k = 1e-4                             #HIF建模随机漂移噪声（过程噪声）
        self.W_k = 1e-4 * np.eye(matrix_dim)       #HIF辨识结果噪声（测量噪声/辨识噪声）
        self.e_rls: float = 0.0                     #RLS辨识误差
        self.J_k: float = 0.0                       #RLS辨识误差平方的指数滑动平均



    def rls_update(self, phi, volt):
        Ko = self.Po_R @ phi / (self.lam + phi.reshape(1, -1) @ self.Po_R @ phi)
        self.Po_R = (self.eye - Ko @ phi.reshape(1, -1)) @ self.Po_R / self.lam
        # if np.trace(self.Po_R) > 1e12:
        #     self.Po_R = np.eye(7) * 1e11
        self.e_rls = volt - phi.reshape(1, -1) @ self.theta_R
        self.theta_R = self.theta_R + Ko * self.e_rls


        return self.theta_R.copy(), self.Po_R.copy()


    def hif_update(self, phi, volt):
        intermid = self.lam * self.eye - self.gamma * self.Po_H + phi @ phi.reshape(1, -1) @ self.Po_H / self.V_k
        Ko = self.Po_H @ np.linalg.pinv(intermid) @ phi / self.V_k
        self.theta_H = self.theta_H + Ko * (volt - phi.reshape(1, -1) @ self.theta_H)
        self.Po_H = self.Po_H @ np.linalg.pinv(intermid) + self.W_k


        return self.theta_H.copy()


    def weight_update(self, volt):
        self.e_rls = self.e_rls / volt  # 误差归一化处理，用于权重分配，保证临界阈值不受工况数值大小影响
        self.J_k = self.epsilon * self.J_k + (1 - self.epsilon) * self.e_rls**2
        alpha = 1 / (1 + np.exp(-self.beta * (self.J_k - self.sigma)))
        self.theta_f = alpha * self.theta_R + (1 - alpha) * self.theta_H
        if self.J_k - self.sigma > 0.0:
            weight = 1           #hif
        else:
            weight = 0         #rls

        return self.theta_f.copy(), weight



    def params_id(self, theta):
        self.theta = theta
        a = (self.theta[3] - self.theta[4] + self.theta[5] - self.theta[6]) / (1 + self.theta[0] - self.theta[1] + self.theta[2])
        b = self.Ts**3 * (1 + self.theta[0] - self.theta[1] + self.theta[2]) / 8 / (1 - self.theta[0] - self.theta[1] - self.theta[2])
        c = self.Ts**2 * (3 + self.theta[0] + self.theta[1] - 3*self.theta[2]) / 4 / (1 - self.theta[0] - self.theta[1] - self.theta[2])
        d = self.Ts * (3 - self.theta[0] + self.theta[1] + 3 * self.theta[2]) / 2 / (1 - self.theta[0] - self.theta[1] - self.theta[2])
        e = self.Ts**2 * (3 * self.theta[3] - self.theta[4] - self.theta[5] + 3 * self.theta[6]) / 4 / (1 - self.theta[0] - self.theta[1] - self.theta[2])
        f = self.Ts * (3 * self.theta[3] + self.theta[4] - self.theta[5] - 3 * self.theta[6]) / 2 / (1 - self.theta[0] - self.theta[1] - self.theta[2])
        g = (self.theta[3] + self.theta[4] + self.theta[5] + self.theta[6]) / (1 - self.theta[0] - self.theta[1] - self.theta[2])



        b = b.item()
        c = c.item()
        d = d.item()


        x0 = np.roots([1, -d, c, -b])
        #避免方程出现虚数解
        x0 = [root.real for root in x0 if abs(root.imag) < 1]
        x0.sort()

        t1 = x0[0]  #高频
        t2 = x0[1]  #中频
        t3 = x0[2]  #低频

        m1 = t1 - t2
        m2 = t1 - t3
        m3 = t2 - t3

        # eps = 1e-5
        #
        # if -eps < m1 < eps:
        #     m1 = np.sign(m1)*eps if m1 != 0 else eps
        # if -eps < m2 < eps:
        #     m2 = np.sign(m2)*eps if m2 != 0 else eps
        # if -eps < m3 < eps:
        #     m3 = np.sign(m3)*eps if m3 != 0 else eps
        
        #参数解算
        r0 = a
        r1 = -(-e + f * t1 - g * t1**2 + r0 * t2 * t3)/(m1 * m2)
        r2 = -(e - f * t2 + g * t2**2 - r0 * t1 * t3)/(m1 * m3)
        r3 = (-e + r0 * t1 * t2 + f * t3 - g * t3**2)/(m2 * (-m3))
        c1 = t1 / r1
        c2 = t2 / r2
        c3 = t3 / r3


        models = [r0.item(), r1.item(), c1.item(), r2.item(), c2.item(), r3.item(), c3.item()]
        return models.copy(), t1, t2, t3






def load_waveforms(curr_file, volt_file):
    curr = np.loadtxt(curr_file)
    volt = np.loadtxt(volt_file)
    assert len(curr) == len(volt)
    return curr, volt


def main():

    ffrls = FFRLS_FF(matrix_dim=7, lam=0.99, gamma=0.05, epsilon=0.95, beta=100, sigma=0.1, Ts=1e-3)
    curr, volt = load_waveforms('is.txt', 'us.txt')


    num = curr.shape[0]
    models = np.zeros((num, 7))
    weight_t = np.zeros((num, 1))

    w: float = 1

    for i in range(3, num):
        #RLS参数辨识更新
        phi = np.hstack([[volt[i-1], volt[i-2], volt[i-3], curr[i], curr[i-1], curr[i-2], curr[i-3]]]).reshape(-1, 1)

        # theta = ffrls.hif_update(phi, volt[i])
        # ffrls.rls_update(phi, volt[i])
        # ffrls.hif_update(phi, volt[i])
        theta, weight_t[i] = ffrls.weight_update(volt[i])
        models[i, :], t1, t2, t3 = ffrls.params_id(theta)



        #参数限幅，限参数>0
        if np.any(models[i, :] < 0):
            j = np.where(models[i, :] < 0)[0]
            for k in j:
                models[i, k] = models[i-1, k]



        # R0限幅
        if models[i, 0] > 10:
            models[i, 0] = models[i - 1, 0]

        # R1限幅
        if models[i, 1] > 10:
            models[i, 1] = models[i - 1, 1]

        # C1限幅
        if models[i, 2] > 10:
            models[i, 2] = models[i - 1, 2]

        # R2限幅
        if models[i, 3] > 10:
            models[i, 3] = models[i - 1, 3]

        # C2限幅
        if models[i, 4] > 10:
            models[i, 4] = models[i - 1, 4]

        # R3限幅
        if models[i, 5] > 10:
            models[i, 5] = models[i - 1, 5]

        # C3限幅
        if models[i, 6] > 10:
            models[i, 6] = models[i - 1, 6]



    print(t1, t2, t3)



    np.savetxt("FFRLS_HIF_identification_results.csv", models, delimiter=",", fmt="%.12f", header="R0,R1,C1,R2,C2,R3,C3",
               comments="")
    np.savetxt("Po.csv", weight_t, delimiter=",", fmt="%.12f", header="Po",
               comments="")




if __name__ == '__main__':
    main()