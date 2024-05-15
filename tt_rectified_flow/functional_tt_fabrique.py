import torch
torch.set_default_dtype(torch.float64)

from numpy.polynomial.legendre import Legendre, legder, leg2poly
from numpy.polynomial.chebyshev import Chebyshev, chebder
from numpy.polynomial.polynomial import Polynomial, polyder
from numpy.polynomial.hermite_e import HermiteE
from math import exp, sqrt, pi, sqrt

from colorama import Fore, Style

from copy import deepcopy

from modelclass import Custom_Polynom_Arithmetic, Custom_Polynomial
from tt_fabrique import TensorTrain
from tictoc import TicToc


def coeffs_grad(coeffs):

    grad_coeffs = []
    for coeff in coeffs:
        grad_coeff = deepcopy(coeff)
        for j in range(coeff.shape[1]-1):
            grad_coeff[:,j] = (j+1)*grad_coeff[:,j+1]
        grad_coeff[:,-1] = 0*grad_coeff[:,-1]
        grad_coeffs.append(grad_coeff)
    return grad_coeffs


def cov(m, rowvar=False):
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()


class orthpoly(object):   

    # TODO: allow different domains in different dimensions
    def __init__(self, degrees,domain):
        """
            Modified to permit now lists of domains for the different dimensions,
            as well as lists of regularization norms for the different dimensions.
            this means 'domain' should now either be a tuple [float,float] or a list
            [[float,float],...,[float,float]] of tuples. 'norm' should be either a string or 
            a list of strings.
        """
        # assert norm in ['L2','H1','H2']
        # assert (len(domain) == 2)

        self.d = len(degrees)
        self.degs = degrees

        self.cpa = Custom_Polynom_Arithmetic(degrees,domain)

        self.coeffs = [self.cpa.to_monomial_mats[mu].T for mu in range(self.d)]
        self.coeffs_grad = coeffs_grad(self.coeffs)
        self.coeffs_lap = coeffs_grad(self.coeffs_grad)

        self.a = [domain[k][0] for k in range(self.d)]
        self.b = [domain[k][1] for k in range(self.d)]

        self.domain = domain


    def __call__(self, x):
        """lifts the inputs to feature space.

        Parameters
        ----------
        x : lb.tensor
            batched inputs of size (batch_size,input_dim)

        Returns
        -------
        embedded_data : list of lb.tensor
            inputs lifted to feature space defined by the feature and
            basis_coeffs attributes. 
            Query [i][j,k] is the k-th basis function evaluated at the j-th sample's
            i-th component.

        """
        assert x.shape[1] == self.d
        embedded_data = []

        for k in range(self.d):
            exponents = torch.arange(0,self.degs[k]+1,1, dtype=torch.float64) 
            embedded_data.append(x[:, k, None] ** exponents)
            if self.coeffs is not None:
                embedded_data[k] = torch.einsum('oi, bi -> bo', self.coeffs[k], embedded_data[k])
        return embedded_data

    def grad(self, x):
        """lifts the inputs to feature-derivative space.

        Parameters
        ----------
        input_data : lb.tensor
            batched inputs of size (batch_size,input_dim)

        Returns
        -------
        embedded_data : list of lb.tensor
            inputs lifted to feature-derivative space defined by the feature and
            grad_coeffs attributes. 
            Query Query [i][j,k] is the first derivative of the k-th basis function evaluated 
            at the j-th sample's i-th component.

        """
        assert x.shape[1] == self.d
        embedded_data = []

        for k in range(self.d):
            exponents = torch.arange(0,self.degs[k]+1,1, dtype=torch.float64)
            embedded_data.append(x[:, k, None] ** exponents)
            if self.coeffs is not None:
                embedded_data[k] = torch.einsum('oi, bi -> bo', self.coeffs_grad[k], embedded_data[k])
        return embedded_data

    def laplace(self, x):
        """lifts the inputs to feature-second-derivative space.

        Parameters
        ----------
        input_data : lb.tensor
            batched inputs of size (batch_size,input_dim)

        Returns
        -------
        embedded_data : list of lb.tensor
            inputs lifted to feature-derivative-derivative space defined by the feature and
            grad_coeffs attributes. 
            Query Query [i][jk] is the second derivative of the k-th basis function evaluated 
            at the j-th sample's i-th component.

        """
        assert x.shape[1] == self.d
        embedded_data = []

        for k in range(self.d):
            exponents = torch.arange(0,self.degs[k]+1,1, dtype=torch.float64)
            embedded_data.append(x[:, k, None] ** exponents)
            if self.coeffs is not None:
                embedded_data[k] = torch.einsum('oi, bi -> bo', self.coeffs_lap[k], embedded_data[k])
        return embedded_data

    
class Extended_TensorTrain(object):

    def __init__(self, tfeatures, ranks, comps=None):
        """
            tfeatures should be a function returning evaluations of feature functions if given a data batch as argument,
            i.e. tfeatures(x), where x is an lb.array of size (batch_size, n_comps),
            is a list of lb.arrays such that tfeatures(x)[i][j,k] is the k-th feature function (in that dimension) 
            evaluated at the j-th samples i-th component
        """

        self.tfeatures = tfeatures
        self.d = self.tfeatures.d

        # TODO also allow ranks len = d + 1  with [1] [...] + [1] shape 
        assert(len(ranks) == self.tfeatures.d+1)
        self.rank = ranks

        self.tt = TensorTrain([deg+1 for deg in tfeatures.degs])
        if comps is None:
            self.tt.fill_random(ranks,1.)
        else:
            # TODO allow ranks len d+1
            for pos in range(self.tfeatures.d-1):
                assert(comps[pos].shape[2] == ranks[pos+1])
            self.tt.set_components(comps)
        self.tt.rank = self.rank

    def __call__(self, x):
        assert(x.shape[1] == self.d)
        u = self.tfeatures(x)
        return self.tt.dot_rank_one(u)

    def evaluate_density(self,x):
        assert(x.shape[1] == self.d)
        u = self.tfeatures(x)
        potential = self.tt.dot_rank_one(u)
        res = torch.exp(-potential)
        return res

    def evaluate_marginals(self,x,d1,d2):
        assert x.shape[1] == 2
        data_slice = torch.zeros((x.shape[0],self.d))
        x_vals = x[:,0]
        y_vals = x[:,1]
        data_slice[:,d1] = x_vals
        data_slice[:,d2] = y_vals
        res = self.evaluate_density(data_slice)
        return res, x_vals, y_vals

    def marginal_density(self,x,y,d1=0,d2=1):
        # TODO enable different dimensions as input (does not work with graphics methods right now)
        res = torch.zeros_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                u = torch.zeros((1,self.d))
                u[0,d1] = x[i,j]
                u[0,d2] = y[i,j]
                res[i,j] = self.evaluate_density(u).squeeze().item()
        return res

    def set_ranks(self, ranks):
        self.tt.retract(self, ranks, verbose=False)

    def grad(self, x=None, grad_data=None, sample_size=None):
        """computes the analytical gradient of the forward pass. 
        Parameters
        ----------
        x : lb.tensor
            input of shape (batch_size,input_dim)

        Returns
        -------
        gradient : lb.tensor
            gradient of the forward pass. Shape (batch_size,input_dim)

        """
        assert (x is not None or (grad_data is not None and isinstance(sample_size,int)))

        if x is not None:
            assert(x.shape[1] == self.d)
            # initialize gradient
            gradient = torch.zeros((x.shape[0], self.d))

            # lift data to feature space and feature-derivative space
            embedded_data = self.tfeatures(x)
            embedded_data_grad = self.tfeatures.grad(x)

            for mu in range(0,self.d):
                data = embedded_data[:mu] + [embedded_data_grad[mu]] + embedded_data[mu+1:]
                gradient[:, mu] = torch.squeeze(self.tt.dot_rank_one(data))
        else:
            gradient = torch.zeros((sample_size, self.d))
            for mu in range(0,self.d):
                gradient[:, mu] = torch.squeeze(self.tt.dot_rank_one(grad_data[mu]))
        return gradient

    def laplace(self, x=None, lap_data=None, sample_size=None):
        """computes the analytical laplacian of the forward pass. 

        Parameters
        ----------
        x : lb.tensor
            input of shape (batch_size,input_dim)

        Returns
        -------
        gradient : lb.tensor
            laplacian of the forward pass. Shape (batch_size,1)
        """
        assert (x is not None or (lap_data is not None and isinstance(sample_size,int)))

        if x is not None:
            assert(x.shape[1] == self.d)
            # initialize laplace
            lap = torch.zeros((x.shape[0],))

            # lift data to feature space and feature-derivative space
            embedded_data = self.tfeatures(x)
            embedded_data_lap = self.tfeatures.laplace(x)

            for mu in range(0,self.d):
                data = embedded_data[:mu] + [embedded_data_lap[mu]] + embedded_data[mu+1:]
                lap += torch.squeeze(self.tt.dot_rank_one(data))
        else:
            lap = torch.zeros((sample_size,))
            for mu in range(0,self.d):
                lap += torch.squeeze(self.tt.dot_rank_one(lap_data[mu]))

        return lap


    def fit(self, x, y, iterations, rule = None, tol = 8e-6, verboselevel = 0, reg_param=None):
        """
            Fits the Extended Tensortrain to the given data (x,y) of some target function 
                     f : K\subset IR^d to IR^m 
                                     x -> f(x) = y.

            @param x : input parameter of the training data set : x with shape (b,d)   b \in \mathbb{N}
            @param y : output data with shape (b,m)
        """

        # assert(x.shape[1] == self.d)
        solver = self.ALS_Regression

        with TicToc(key=" o ALS total ", do_print=False, accumulate=True, sec_key="ALS: "):
            solver(x,y,iterations,tol,verboselevel, rule, reg_param)
        # self.tt.set_components(res.comps)

    def ALS_Regression(self, x, y, iterations, tol, verboselevel, rule = None, reg_param=None):
        
        """
            @param loc_solver : 'normal', 'least_square',  
            x shape (batch_size, input_dim)
            y shape (batch_size, 1)
        """
        
        # size of the data batch
        b = y.shape[0]

        # feature evaluation on input data
        u = self.tfeatures(x)

        # 0 - orthogonalize, s.t. sweeping starts on first component
        self.tt.set_core(mu = 0)

        # TODO: name stack instead of list
        # initialize lists for left and right contractions
        R_stack = [torch.ones((b, 1))]
        L_stack = [torch.ones((b, 1))]

        d = self.tt.n_comps

        def add_contraction(mu, list, side='left'):

            assert ((side == 'left' or side == 'right') or (side == +1 or side == -1))

            with TicToc(key=" o left/right contractions ", do_print=False, accumulate=True, sec_key="ALS: "):     
                core_tensor = self.tt.comps[mu]
                data_tensor = u[mu]
                contracted_core = torch.einsum('idr, bd -> bir', core_tensor, data_tensor)
                if (side == 'left' or side == -1):
                    list.append(torch.einsum('bir, bi -> br', contracted_core, list[-1]))
                else: 
                    list.append(torch.einsum('bir, br -> bi', contracted_core, list[-1]))


        def solve_local(mu,L,R):

            with TicToc(key=" o least square matrix allocation ", do_print=False, accumulate=True, sec_key="ALS: "):
                A = torch.einsum('bi,bj,br->bijr', L, u[mu], R)
                A = A.reshape(A.shape[0], A.shape[1]*A.shape[2]*A.shape[3])

                if reg_param is not None:
                    assert isinstance(reg_param,float)

                
            with TicToc(key=" o local solve ", do_print=False, accumulate=True, sec_key="ALS: "):

                # c, res, rank, sigma = lb.linalg.lstsq(A, y, rcond = None)  
                ATA, ATy = A.T@A, A.T@y

                if reg_param is not None:
                    assert isinstance(reg_param,float)
                    ATA += reg_param * torch.eye(ATA.shape[0])

                c = torch.linalg.solve(ATA,ATy)

                rel_err = torch.linalg.norm(A@c - y)/torch.linalg.norm(y)
                if rel_err > 1e-4:
                    with TicToc(key=" o local solve via lstsq ", do_print=False, accumulate=True, sec_key="ALS: "):
                        if reg_param is not None:
                            Ahat = torch.cat([A,sqrt(reg_param)*torch.eye(A.shape[1])],0)
                            yhat = torch.cat([y,torch.zeros((A.shape[1],1))],0)
                            c, res, rank, sigma = torch.linalg.lstsq(Ahat, yhat, rcond = None) 
                        else:
                            c, res, rank, sigma = torch.linalg.lstsq(A, y, rcond = None)  

                s = self.tt.comps[mu].shape
                self.tt.comps[mu] = c.reshape(s[0],s[1],s[2])


        # initialize residual
        #TODO rename to rel res
        curr_res = torch.linalg.norm(self(x) - y)**2/torch.linalg.norm(y)**2 # quadratic norm
        if verboselevel > 0: print("START residuum : ", curr_res)

        # initialize stop condition
        niter = 0
        stop_condition = niter > iterations or curr_res < tol

        # loc_solver =  solve_local_iterativeCG
        loc_solver = solve_local

        # before the first forward sweep we need to build the list of right contractions
        for mu in range(d-1,0,-1):
            add_contraction(mu, R_stack, side='right')

        history = []

        while not stop_condition:
            # forward half-sweep
            for mu in range(d-1):
                self.tt.set_core(mu)
                if mu > 0:
                    add_contraction(mu-1, L_stack, side='left')
                    del R_stack[-1]
                loc_solver(mu,L_stack[-1],R_stack[-1])

            # before back sweep
            self.tt.set_core(d-1)
            add_contraction(d-2, L_stack, side='left')
            del R_stack[-1]

            # backward half sweep
            for mu in range(d-1,0,-1):
                self.tt.set_core(mu)
                if mu < d-1:
                    add_contraction(mu+1, R_stack, side='right')
                    del L_stack[-1]
                loc_solver(mu,L_stack[-1],R_stack[-1])


            # before forward sweep
            self.tt.set_core(0)
            add_contraction(1, R_stack, side='right')
            del L_stack[-1]


            # update stop condition
            niter += 1
            curr_res = torch.linalg.norm(self(x) - y)**2/torch.linalg.norm(y)**2
            # update reg_param
            reg_param = reg_param*max(curr_res.item(),0.9)
            stop_condition = niter > iterations or  curr_res < tol
            if verboselevel > 0: # and  niter % 10 == 0: 
                print("{c}{k:<5}. iteration. {r} Data residuum : {c2}{res}{r}".format(c=Fore.GREEN, c2=Fore.RED, r=Style.RESET_ALL, k = niter, res = curr_res))


            history.append(curr_res)
            Hlength = 5
            rateTol = 1e-5

            if len(history) > Hlength:
                latestH = torch.tensor(history[-Hlength:])
                relative_history_rate = cov(latestH) / torch.mean(latestH)

                if relative_history_rate < rateTol:
                    if verboselevel > 0:
                        print("===== Attempting rank update ====")
                    if rule is not None:
                        self.tt.modify_ranks(rule)
                        # set core to 0 and re-initialize lists
                        self.tt.set_core(mu=0)
                        R_stack = [torch.ones((b, 1))]
                        L_stack = [torch.ones((b, 1))]
                        for mu in range(d-1,0,-1):
                            add_contraction(mu, R_stack, side='right')

                        history = []
                
        # return self.tt
