import numpy as np
from scipy.linalg import eigh


class DPP:
    def __init__(self, D, V):
        self.D_const = D
        self.V_const = V

    def elem_sympoly(self, k):
        ''' Algorithm 7 '''
        N = len(self.D)
        E = np.zeros((k+1,N+1))
        E[0,:] = 1
        for l in range(1, k+1):
            for n in range(1, N+1):
                E[l, n] = E[l, n-1] + self.D[n-1]*E[l-1,n-1];
        return E

    # def k_dpp_phase1(self, k, E=None):
    #     ''' Algorithm 8 '''
    #     J, l = [], k
    #     E = E if E is not None else self.elem_sympoly(k) 
    #     for n in range(self.N, 0, -1):# n = N,.., 2, 1
    #         # print("len(D):",str(len(D)),", shapeE:",str(E.shape),", n=",str(n),", l=",str(l))
    #         if l==0: break
    #         if np.random.random() < self.D[n-1] * E[l-1, n-1] / E[l, n]:
    #             J.append(n-1)
    #             l -= 1
    #     # print(J)
    #     return np.array(np.sort(J))

    def k_dpp_phase1(self, k, E=None):
        E = E if E is not None else self.elem_sympoly(k) 
#        print('E = ', str(E))
        i = len(self.D)-1
        remaining = k-1
        S = np.zeros(k)
        while remaining >= 0:
            # compute marginal of i given that we choose remaining values from 1:i
            if i == remaining:
                marg = 1
            else:
                marg = self.D[i] * E[remaining, i] / E[remaining+1, i+1]
            
            # sample marginal
            if np.random.random() < marg:
                S[remaining] = i
                remaining = remaining - 1
            i -= 1
        return list(map(int, np.sort(S)))

    def sample(self, k=None, E=None):
        self.D, self.V = self.D_const.copy(), self.V_const.copy()
        self.N = self.D.shape[0]
        # PHASE 1
        if k is None:# general dpp
            self.D /= 1 + self.D
            self.V = self.V[:,np.random.rand(self.N) < self.D]
            k = self.V.shape[1]
        else:# k-dpp
            self.V = self.V[:,self.k_dpp_phase1(k, E)]

        # PHASE 2
        Y = []
        for vi in range(k-1,-1,-1):
            # choose vector index, with prob proportional to K_ii = v_i^T v_i, lambda==1??
            P = np.sum(np.power(self.V, 2), axis=1)
            i = np.random.choice(range(self.N), p=P/np.sum(P))

            Y.append(i)

            # Update K to condition on event i ∈ Y
            col_idx = np.nonzero(self.V[i])[0][0]# Select our eigenvector to remove
            V_j = np.copy(self.V[:,col_idx])# save the first eigenvector
            # print('\n1: i = '+str(i))#+', col_idx = '+str(col_idx)+'\n')
            # [print(''.join(['{0:.2f},'.format(vi) if vi<0 else ' {0:.2f},'.format(vi) for vi in v])) for v in self.V]
            # Inference conditioning: P(B in Y|A in Y) = P(AUB in Y)/P(A in Y) = det(K_B - K_BA*K_A^-1*K_AB)
            # vB - vBA/vA * vAB
            # Remove a c*V_j from each vector, where c is a factor choosen so that the i element of each vector is set to 0
            self.V -= np.outer(V_j, self.V[i]/V_j[i])# the first vector in V is now zeros
            # print('\n2')
            # [print(''.join(['{0:.2f},'.format(vi) if vi<0 else ' {0:.2f},'.format(vi) for vi in v])) for v in self.V]
            self.V[:,col_idx] = self.V[:,vi]# set the first vector in V to the last vector in V
            # print('\n3')
            # [print(''.join(['{0:.2f},'.format(vi) if vi<0 else ' {0:.2f},'.format(vi) for vi in v])) for v in self.V]
            self.V  = self.V[:,:vi]# remove the last vector
            # print('\n4')
            # [print(''.join(['{0:.2f},'.format(vi) if vi<0 else ' {0:.2f},'.format(vi) for vi in v])) for v in self.V]

            # Orthogonalize
            # if vi > 0: self.V = self.V.dot(inv(np.real(sqrtm(self.V.T.dot(self.V)))))
            if vi > 0: #self.gramSchmidt(vi)
                for a in range(vi):
                    for b in range(a):
                        self.V[:, a] -= np.dot(self.V[:, a], self.V[:, b])*self.V[:, b]
                    self.V[:, a] /= np.linalg.norm(self.V[:, a])
        return np.sort(Y)

    def gramSchmidt(self, i):
        for a in range(i):
            for b in range(a):
                self.V[:, a] -= np.dot(self.V[:, a], self.V[:, b])*self.V[:, b]
            self.V[:, a] /= np.linalg.norm(self.V[:, a])
