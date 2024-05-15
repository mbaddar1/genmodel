import torch 
torch.set_default_dtype(torch.float64)
# import numpy as np
from copy import copy, deepcopy
from tictoc import TicToc
from itertools import product
#    

from colorama import Fore, Style

import time

# TODO
# - embedd into TT.py tensor train class for features like core position, core shift,...
# - test cases


def common_field_dtype(tensor_list1, tensor_list2):
    is_complex = False
    for t in tensor_list1:
        assert t.dtype in [torch.cfloat, torch.get_default_dtype()]
        if t.dtype == torch.cfloat:
            return torch.cfloat
        
    for t in tensor_list2:
        assert t.dtype in [torch.cfloat, torch.get_default_dtype()]
        if t.dtype == torch.cfloat:
            return torch.cfloat
        
    return torch.get_default_dtype() 



def get_ranks(TT_list):
    return [1] + [C.shape[2] for C in TT_list[:-1]] + [1]

def laplace_like_sum(TT_list, TT2_list):
    # @pre each item of the TT_list is given as :

    #  LaplaceLikeOperator = sum_mu : I o I ... o L_mu o... I    with L_mu at mu-th position
    # in the implementation LLO = [L_1, ..., L_d]
    
    # the resulting sum of d tensor trains with this operator action, then defines a tensor train of rank 2r. 
    assert len(TT_list) == len(TT2_list)
    for mu in range(len(TT_list)):
        assert TT_list[mu].shape == TT2_list[mu].shape

    comps = []

    # Set first component
    L = TT2_list[0]
    C = TT_list[0]
    data = torch.cat([deepcopy(L[0,:,:]), C[0,:,:]], axis = 1)
    data = data.reshape(1,data.shape[0],data.shape[1])
    comps.append(data)

    # set middle components
    for C, L in zip(TT_list[1:-1], TT2_list[1:-1]):
        rp1, _, rpp1 =  C.shape
        rp2, _, rpp2 =  L.shape

        
        ldtype = common_field_dtype([C], [L])

        data = torch.zeros((rp1+rp2, C.shape[1], rpp1+rpp2), dtype = ldtype)
        data[:rp1, :, :rpp1] = deepcopy(C)
        data[rp1:, :, :rpp1] = deepcopy(L)
        data[rp1:, :, rpp1:] = deepcopy(C)
        comps.append(data)

    C, L = TT_list[-1], TT2_list[-1]
    data = torch.cat([deepcopy(C)[:,:,0], L[:,:,0]], axis = 0)
    data = data.reshape(data.shape[0],data.shape[1],1)
    comps.append(data)

    return comps


def TT_add(TT_1,TT_2, a= 1., b = 1.):
    TT1 = TT_1.comps
    TT2 = TT_2.comps
    assert len(TT1) == len(TT2)
    assert TT_1.dims == TT_2.dims
    for mu in range(len(TT1)):
        assert TT1[mu].shape[1] == TT2[mu].shape[1]
    TT = []
    # Set first component
    data = torch.cat([deepcopy(TT1[0][0,:,:]), deepcopy(TT2[0][0,:,:])], axis = 1)
    data = data.reshape(1,data.shape[0],data.shape[1])
    TT.append(data)

    # set middle components
    for p in range(1, len(TT1)-1):
        # r_{i}^1  di  r_{i+1}^1
        c1, c2 = TT1[p], TT2[p]
        rp1, _, rpp1 =  c1.shape
        # r_{i}^2  di  r_{i+1}^2
        rp2, _, rpp2 =  c2.shape
        data = torch.zeros((rp1+rp2, c1.shape[1], rpp1+rpp2))
        data[:rp1, :, :rpp1] = deepcopy(c1)
        data[rp1:, :, rpp1:] = deepcopy(c2)
        TT.append(data)
    
    # set last component
    data = torch.cat([deepcopy(a*TT1[-1][:,:,0]), deepcopy(b*TT2[-1][:,:,0])], axis = 0)
    data = data.reshape(data.shape[0],data.shape[1],1)
    TT.append(data)

    Added_TT = TensorTrain(TT_1.dims,TT)

    return Added_TT



def prod(list):
    res = list[0]
    for i in range(1,len(list)):
        res = res * list[i]
    return res

# upper bound for ranks
def max_ranks(degrees):
    dofs = [degree+1 for degree in degrees]
    max_rank = [1] \
                    +[min(prod(dofs[:k+1]), prod(dofs[k+1:])) for k in range(len(dofs)-1)] \
                    + [1]
    return max_rank

class Threshold(object):
    def __init__(self,delta):
        self.delta = delta
    def __call__(self, u, sigma,v,  pos):
        return max([torch.sum(sigma > self.delta),1])
    
def rankstepsizecontrol(ETT, F, tau, truncation_rank, rtol, max_rank, verbose = False, bisection_steps = 5):

    def get_Atrun_A(dt):
        A_trun = TT_add(ETT,F, a=1., b=dt)
        A = deepcopy(A_trun)
        A_trun.rank_truncation(truncation_rank)
        return A_trun, A

    def relative_truncation_error(dt):
        A_trun, A = get_Atrun_A(dt)
        diff = TT_add(A_trun,A, a=1., b=-1)
        diff.rank_truncation(max_rank)
        return  TensorTrain.frob_norm(diff).item() / TensorTrain.frob_norm(A).item()

    # init the tau_rank test
    tau_rank = tau 
    
    # perform halfing tau_rank until relative error requirement is met
    print(relative_truncation_error(tau_rank),  rtol)
    while relative_truncation_error(tau_rank) > rtol: 
        tau_rank /= 2.        
        if verbose: 
            print("tau_rank halfed")

    # if no half step is performed take the proposed tau as candidate
    if tau_rank == tau: 
        return tau
    
    # else perform bisection search to potentially increase tau_rank
    tau_rank_bounds = [tau_rank, 2*tau_rank]
    for k in range(bisection_steps):
        tau_bound_candidate = 0.5*sum(tau_rank_bounds)
        if relative_truncation_error( tau_bound_candidate ) < rtol:
            tau_rank_bounds[0] = tau_bound_candidate 
        else:
            tau_rank_bounds[1] = tau_bound_candidate

        if verbose: 
            print("{k+1}th bisection step : ", tau_rank_bounds)
    
    # the lower tau_rank bound by will lower bound for optimal tau_rank
    tau_rank =  tau_rank_bounds[0]
    return tau_rank

    

    


class TT_Fabrique(object):
    def __init__(self, degrees, B, B_inv, ONB, ONB_inv):
        """
            Class for handling algebraic TT operations defined by HJB right hand side.
            Input:
                - B (list): list of basis change matrices from Legendre polynomials to monomials 1, x, x^2,...
                            B[i] must be a torch.tensor of shape AT LEAST (2*degrees[i]+2,2degrees[i]+2) entrywise  (NOTE THE 2!!!)
                - B_inv (list): list of inverses of B (change from monomials back to Legendre)
        """
        self.d = len(degrees)
        assert len(B) == self.d
        assert len(B_inv) == self.d

        self.degrees = degrees
        self.degrees_times_2 =  [2*deg for deg in degrees]
        self.d = len(self.degrees)
        self.B = B
        self.B_inv = B_inv
        self.ONB = ONB
        self.ONB_inv = ONB_inv

        self.L_list = []
        self.D_list = []

        
        for mu in range(self.d):
            
            N_mu = self.degrees[mu]
            
            # Operator \partial^2_x + x\partial_x for linear part on Monomials
            L_mu = torch.diag(torch.tensor([k for k in range(N_mu+1)], dtype = torch.float64))
            L_mu += torch.diag(torch.tensor([(k+2.0)*(k+1.0) for k in range(N_mu-1)]),+2)
            
            # Operator \partial_x for nonlinear part on Monomial
            D_mu = torch.diag(torch.tensor([(k+1.) for k in range(N_mu)]),+1)
            
            # apply basis back and forth transformation to define operator in Legendre polynomials
            B_mu = self.B[mu][:N_mu+1,:N_mu+1]
            B_mu_inv = self.B_inv[mu][:N_mu+1,:N_mu+1]
            L_mu = B_mu_inv @ L_mu @ B_mu
            D_mu = B_mu_inv @ D_mu @ B_mu
            self.L_list.append(L_mu)
            self.D_list.append(D_mu)
            

    def _B_transform(self,TT_list):
        """Gets a List of TT compoments corresponding to a function in Legendre polynomials and transforms
            them to components of the same function expressed in monomials

        Args:
            TT_list (list): list of TT components (each a torch.tensor)

        Returns:
            BTT_List (list): lost of transformed copmonents
        """        
        assert len(TT_list) == len(self.B)
        d  = [TT_list[mu].shape[1] for mu in range(len(TT_list))]

        ldtype = common_field_dtype(self.B, TT_list)

        BTT_list = [ torch.einsum('ij, kjl -> kil', (self.B[mu][:d[mu],:d[mu]]).type(ldtype), TT_list[mu].type(ldtype)) for mu in range(len(self.B))]
        return BTT_list

    def _B_inv_transform(self,TT_list):
        """Gets a List of TT compoments corresponding to a function in monomials and transforms
            them to components of the same function expressed in Legendre polynomials

        Args:
            TT_list (list): list of TT components (each a torch.tensor)

        Returns:
            BTT_List (list): lost of transformed copmonents
        """  
        assert len(TT_list) == len(self.B_inv)
        d  = [TT_list[mu].shape[1] for mu in range(len(TT_list))]
        ldtype = common_field_dtype(self.B_inv, TT_list)
        BTT_list = [ torch.einsum('ij, kjl -> kil', (self.B_inv[mu][:d[mu],:d[mu]]).type(ldtype), TT_list[mu].type(ldtype)) for mu in range(len(self.B_inv))]
        return BTT_list

    def _ONB_transform(self,TT_list):
        """Gets a List of TT compoments corresponding to a function in Legendre polynomials and transforms
            them to components of the same function expressed in ORTHONORMAL Legendre polynomials

        Args:
            TT_list (list): list of TT components (each a torch.tensor)

        Returns:
            BTT_List (list): lost of transformed copmonents
        """        
        assert len(TT_list) == len(self.ONB)
        d  = [TT_list[mu].shape[1] for mu in range(len(TT_list))]

        ldtype = common_field_dtype(self.ONB, TT_list)

        BTT_list = [ torch.einsum('ij, kjl -> kil', (self.ONB[mu][:d[mu],:d[mu]]).type(ldtype), TT_list[mu].type(ldtype)) for mu in range(len(self.ONB))]
        return BTT_list

    def _ONB_inv_transform(self,TT_list):
        """Gets a List of TT compoments corresponding to a function in ORTHONORMAL Legendre polynomials and transforms
            them to components of the same function expressed in Legendre polynomials

        Args:
            TT_list (list): list of TT components (each a torch.tensor)

        Returns:
            BTT_List (list): lost of transformed copmonents
        """  
        assert len(TT_list) == len(self.ONB_inv)
        d  = [TT_list[mu].shape[1] for mu in range(len(TT_list))]
        ldtype = common_field_dtype(self.ONB_inv, TT_list)
        BTT_list = [ torch.einsum('ij, kjl -> kil', (self.ONB_inv[mu][:d[mu],:d[mu]]).type(ldtype), TT_list[mu].type(ldtype)) for mu in range(len(self.ONB_inv))]
        return BTT_list
    
    def linear_HJB_part(self, TT):
        """Returns an evaluation of the linear part of the HJB rhs on a TT, and returns the resulting TT.

        Args:
            TT (TensorTrain): TensorTrain object with ranks
                                    [1,r_2,...,r_{d-1},1]

        Returns:
            LTT (TensorTrain): TensorTrain object with ranks 
                                    [1,min{max_rank_1,2r_1},...,min{max_rank_{d-1},2r_{d-1}},1]
                                where the max_ranks are determined by the dimensions.
        """        
        assert TT.n_comps == self.d
        LTT = deepcopy(TT)
        # operator_list = [lambda core: torch.einsum('ij, kjl -> kil', self.L_list[mu], core) for mu in range(self.d)]
        # LTT.comps =  laplace_like_sum(LTT.comps, operator_list)


        ldtype = common_field_dtype(self.L_list, LTT.comps)
        temp_comps = [torch.einsum('ij, kjl -> kil', self.L_list[mu].type(ldtype), LTT.comps[mu].type(ldtype)) for mu in range(self.d)]
        LTT.comps = laplace_like_sum(LTT.comps, temp_comps)

        ranks = max_ranks(self.degrees)
        LTT.rank_truncation(ranks)

        return LTT
    
    @staticmethod
    def TT_from_monomial_sparse_dict( dict, delta = 1e-10, max_degrees = None): 

        "@param delta : treshold value for the underlying rounding procedure of sum of TTs"
        
        # define the dimension, all indices must be given in same length
        # additional define maximum degree in each dimension
        d = -1
        for idx in dict.keys():
            if d == -1:
                d = len(idx)
                degrees = [i for i in idx]
            assert len(idx) == d
            degrees = [max(deg, i) for deg, i in zip(degrees, idx)]
        
        if max_degrees is not None : 
            assert len(max_degrees) == len(degrees)
            degrees = [max(deg, mdeg) for deg, mdeg in zip(degrees, max_degrees)]

        dims = [deg +1 for deg in degrees]
        init_comps = [ torch.zeros(1,dd,1) for dd in dims]
        TT = TensorTrain(dims, init_comps)

        for  idx, v in dict.items():
            add = TensorTrain(dims, init_comps)

            for mu, ii in enumerate(idx): 
                add.comps[mu][0,ii,0] = abs(v)**(1./d)

                if mu == 0 : 
                    add.comps[mu][0,ii,0] *= torch.sign(torch.tensor(v))

            TT = TT_add(TT, add)
            TT.round(delta)
            rank = [1] + [comp.shape[2] for comp in TT.comps[0:-1]] + [1]
            TT.rank = rank 

        return TT

        
        

    def __component_square_operation(self, C):
        """
        Realization of 
        
        """
        s = C.shape
        new_C = torch.zeros((s[0],s[0], 2*s[1], s[2], s[2]))
        for offset in range(s[1]):
            new_C[:,:, slice(offset, offset + s[1], None), :,:] += torch.einsum('idj, kl-> ikdjl', C, C[:,offset,:])
        new_C = new_C.reshape(s[0]*s[0], 2*s[1], s[2]* s[2])
        
        return new_C
    
    def __component_fg_operation(self, C1, C2):
        """
        Realization of 
        
        """
        s1 = C1.shape
        s2 = C2.shape
        
        ldtype = common_field_dtype([C1], [C2])

        new_C = torch.zeros((s1[0],s2[0], s1[1]+s2[1], s1[2], s2[2]), dtype = ldtype)
        for offset in range(s1[1]):
            new_C[:,:, slice(offset, offset + s2[1], None), :,:] += torch.einsum('idj, kl-> ikdjl', C1.type(ldtype), C2[:,offset,:].type(ldtype))
        new_C = new_C.reshape(s1[0]*s2[0], s1[1]+s2[1], s1[2]* s2[2])
        
        return new_C
    
    def non_linear_HJB_part(self, TT):
        """_summary_

        Args:
            TT (_type_): _description_

        Returns:
            _type_: _description_
        """        

        NTT = deepcopy(TT)
        # print(NTT.comps[0].shape)
        
        D_TT_list = [torch.einsum('ij, rjs-> ris', self.D_list[mu], C) for mu, C in enumerate(NTT.comps)]
        # print(D_TT_list[0].shape)
       
        D_TT_list_mon = self._B_transform(D_TT_list)
        TT_list_mon   = self._B_transform(NTT.comps)

        D_TT_list_mon_square = [self.__component_square_operation(C) for C in D_TT_list_mon]
        TT_list_mon_square   = [self.__component_square_operation(C) for C in TT_list_mon]

        D_TT_list_square   = self._B_inv_transform(D_TT_list_mon_square)
        TT_list_square     = self._B_inv_transform(TT_list_mon_square)

        # def L(mu):
        #     return D_TT_list_square[mu]
        # operator_list = [lambda C: L(mu) for mu in range(NTT.n_comps)]
        # res =  laplace_like_sum(TT_list_square, operator_list)

        temp_comps = [D_TT_list_square[mu] for mu in range(self.d)]
        res = laplace_like_sum(TT_list_square, temp_comps)

        # NTT.comps = res
        dims = [comp.shape[1] for comp in res]
        rank = [1] + [comp.shape[2] for comp in res[0:-1]] + [1]
        NTT = TensorTrain(dims=dims,comp_list=res)
        NTT.rank = rank

        # print('rang precut:', [1] + [comp.shape[2] for comp in NTT.comps[:-1]] + [1])
        # Optional: rank truncation to maximal ranks

        ranks = max_ranks([deg+1 for deg in self.degrees_times_2])
        NTT.rank_truncation(ranks)

        return NTT
     
    def projection(self, TT, degrees):
        """ 
        Applies the degree reduction projection 
        and updates the TT ranks with reduction if current rank larger than maximum possible ranks

        @pre the TT is in orthogonal poly class coefficient representation
        """
        assert TT.n_comps == len(degrees)
        PTT = deepcopy(TT)
        for C, deg in zip(PTT.comps, degrees):
            assert deg + 1 <= C.shape[1]
        TT_proj =  [ deepcopy(C[:,:deg+1,:]) for C, deg in zip(PTT.comps, degrees)]
        
        # PTT.comps = TT_proj
        dims = [comp.shape[1] for comp in TT_proj]
        rank = [1] + [comp.shape[2] for comp in TT_proj[0:-1]] + [1]
        PTT = TensorTrain(dims=dims,comp_list=TT_proj)
        PTT.rank = rank

        # short ranks to maximum possible : 
        ranks = max_ranks(self.degrees)
        PTT.rank_truncation(ranks)

        return PTT

                
    def HJB_RHS(self, TT, compute_proj_error = True):

        # @PRE TT_list is right orthogonal
        assert TT.n_comps == self.d
        for mu in range(self.d):
            assert TT.dims[mu] == self.degrees[mu] + 1
        HJB_TT = deepcopy(TT)
        if HJB_TT.core_position != HJB_TT.n_comps-1:
            HJB_TT.set_core(HJB_TT.n_comps-1)
        
        TT_lin = self.linear_HJB_part(HJB_TT)                      # right orthogonal 
        TT_nonlin = self.non_linear_HJB_part(HJB_TT)               # right orthogonal 
        TT_nonlin_proj = self.projection( TT_nonlin, self.degrees)  # right orthogonal 

        res =  TT_add(  TT_lin, TT_nonlin_proj, a=1, b= -1)
        # short ranks to maximum possible : 
        res.rank_truncation(max_ranks(self.degrees))
        
        if not compute_proj_error:
            return res
        
        # embedd the projection back into higher space to compute the difference
        if compute_proj_error: 

            TT_nonlin_proj_embedd = deepcopy(TT_nonlin)
            TT_nonlin_proj_embedd.fill_random(TT_nonlin.rank,0.)
            for mu in range(self.d):
                comp = TT_nonlin_proj.comps[mu]
                TT_nonlin_proj_embedd.comps[mu][:comp.shape[0],:comp.shape[1],:comp.shape[2]] = comp
        
            # Tensors need to be transformed to orthonormal basis to use Parseval
            TT_nonlin.comps = self._ONB_transform(TT_nonlin.comps)
            TT_nonlin_proj_embedd.comps = self._ONB_transform(TT_nonlin_proj_embedd.comps)

            TT_proj_error =   TT_add(TT_nonlin, TT_nonlin_proj_embedd,1.,-1.)
            TT_proj_error.rank_truncation(max_ranks([2*d for d in self.degrees]))
            
            # proj_error = TensorTrain.frob_norm(TT_proj_error).item()
            proj_error = TensorTrain.frob_norm(TT_proj_error).item() / TensorTrain.frob_norm(TT_nonlin).item() # relative projection error 

            return res, proj_error

        

    def stiffness_operator(self, TT):
        
        D_TT_list = [torch.einsum('ij, rjs-> ris', self.D_list[mu], C) for mu, C in enumerate(TT.comps)]
        D_TT_list_mon = self._B_transform(D_TT_list)
        TT_list_mon   = self._B_transform(TT.comps)


        def linearized_non_linear_HJB_part(x):
            
            ldtype = common_field_dtype(self.D_list, x.comps)

            D_x_list = [torch.einsum('ij, rjs-> ris', self.D_list[mu].type(ldtype), C.type(ldtype)) for mu, C in enumerate(x.comps)]
            D_x_list_mon = self._B_transform(D_x_list)
            x_list_mon   = self._B_transform(x.comps)

            D_TT_list_mon_fg = [self.__component_fg_operation(C[0], C[1]) for C in zip(D_TT_list_mon, D_x_list_mon)]
            TT_list_mon_fg   = [self.__component_fg_operation(C[0], C[1]) for C in zip(TT_list_mon, x_list_mon)]

            D_TT_list_fg   = self._B_inv_transform(D_TT_list_mon_fg)
            TT_list_fg     = self._B_inv_transform(TT_list_mon_fg)

            res = laplace_like_sum(TT_list_fg, D_TT_list_fg)

            # NTT.comps = res
            dims = [comp.shape[1] for comp in res]
            rank = [1] + [comp.shape[2] for comp in res[0:-1]] + [1]
            

            Linearised_NTT_x =      TensorTrain(dims=dims,comp_list=res) 
                                
            Linearised_NTT_x.rank = rank

            #print('rang precut:', [1] + [comp.shape[2] for comp in NTT.comps[:-1]] + [1])
            # Optional: rank truncation to maximal ranks

            ranks = max_ranks([deg+1 for deg in self.degrees_times_2])
            Linearised_NTT_x.rank_truncation(ranks)

            return Linearised_NTT_x

        def Linearized_HJB(x):
            x1 = self.linear_HJB_part(x)                   # right orthogonal 
            x2 = linearized_non_linear_HJB_part(x)               # right orthogonal 
            x2_proj = self.projection( x2, self.degrees)  # right orthogonal 
            res =  TT_add(  x1, x2_proj, a=1, b= -2)

            #res = x1
            # short ranks to maximum possible : 
            #res.rank_truncation(max_ranks(self.degrees_times_2))
            #res = self.projection(res, self.degrees)
            res.rank_truncation(max_ranks(self.degrees))
        
            return res
        return Linearized_HJB
    

    def explicit_Euler_step(self, TT, tau, trun_rtol, ranks=None, HJB_rhs=None, compute_trunc_error=True,  verbose = False):
        ETT = deepcopy(TT)
        # ranks = get_ranks(ETT.comps)
        F = self.HJB_RHS(ETT,compute_proj_error=False) if HJB_rhs is None else HJB_rhs

        if ranks is None:
            # perform at least a consistency rank truncation to max possible ranks
            ranks = max_ranks(self.degrees)
        else:
            ranks = get_ranks(ETT.comps)

        if trun_rtol is not torch.inf: 
            tau_2 = rankstepsizecontrol(ETT, F, tau, truncation_rank = ranks, rtol = trun_rtol, 
                                         max_rank = max_ranks([2*d for d in self.degrees]), verbose = verbose)
            if tau_2 != tau: 
                print(f" {Fore.BLUE} Time step decreased due to rel. rank truncation criteria: \
                                {Fore.BLUE} {tau}  ->  {tau_2}{Style.RESET_ALL}")
                time.sleep(0.1) 

            res = TT_add(ETT,F, a=1., b=tau_2)
        else:
            res = TT_add(ETT,F, a=1., b=tau)


        if compute_trunc_error:
            # store TT before truncation for reference
            res_before_trunc = deepcopy(res)


        # perform rank truncation
        res.rank_truncation(ranks)

        # error of rank truncation
        if compute_trunc_error: 
            TT_trunc_error =   TT_add(res_before_trunc, res,1.,-1.)
            # print(max_ranks([2*d for d in self.degrees]))
            # print([TT_trunc_error.comps[mu].shape[2] for mu in range(len(TT_trunc_error.comps)-1)])
            TT_trunc_error.rank_truncation(max_ranks([2*d for d in self.degrees]))
            trunc_error = TensorTrain.frob_norm(TT_trunc_error).item() / TensorTrain.frob_norm(res_before_trunc).item() # relative projection error 

            return res, trunc_error

        else:
            return res


    def explicit_euler_step(self, TT, tau, ranks=None, HJB_rhs=None, compute_trunc_error=True,  verbose = False):
        ETT = deepcopy(TT)
        # ranks = get_ranks(ETT.comps)
        F = self.HJB_RHS(ETT,compute_proj_error=False) if HJB_rhs is None else HJB_rhs

        if ranks is None:
            # perform at least a consistency rank truncation to max possible ranks
            ranks = max_ranks(self.degrees)
        # else:
        #     ranks = get_ranks(ETT.comps)


        res = TT_add(ETT,F, a=1., b=tau)


        if compute_trunc_error:
            # store TT before truncation for reference
            res_before_trunc = deepcopy(res)


        # perform rank truncation
        res.rank_truncation(ranks)

        # error of rank truncation
        if compute_trunc_error: 
            TT_trunc_error =   TT_add(res_before_trunc, res,1.,-1.)
            # print(max_ranks([2*d for d in self.degrees]))
            # print([TT_trunc_error.comps[mu].shape[2] for mu in range(len(TT_trunc_error.comps)-1)])
            TT_trunc_error.rank_truncation(max_ranks([2*d for d in self.degrees]))
            trunc_error = TensorTrain.frob_norm(TT_trunc_error).item() / TensorTrain.frob_norm(res_before_trunc).item() # relative projection error 

            return res, trunc_error

        else:
            return res





def left_unfolding(order3tensor):
    s = order3tensor.shape
    return order3tensor.reshape(s[0]*s[1], s[2])

def right_unfolding(order3tensor):
    s = order3tensor.shape
    return order3tensor.reshape(s[0], s[1]*s[2])

class TensorTrain(object):
    def __init__(self, dims, comp_list = None):
        
        self.n_comps = len(dims)
        self.dims = dims
        self.comps = [None] * self.n_comps

        self.rank = None
        self.core_position = None

        # upper bound for ranks
        self.uranks = [1] + [min(prod(dims[:k+1]), prod(dims[k+1:])) for k in range(len(dims)-1)] + [1]

        if comp_list is not None:
            self.set_components(comp_list)

    @staticmethod
    def hadamard_product(A,B):
        """ Computes <A,B> = AoB  with o beeing the hadamard product"""

        if isinstance(A,TensorTrain) and isinstance(B, TensorTrain):
            assert len(A.dims) == len(B.dims)
            for d in range(len(A.dims)):
                assert A.dims[d] == B.dims[d]

            n_comps = len(A.dims) 
            d = A.dims[0]

            v = sum(torch.kron(A.comps[0][:,i,:], B.comps[0][:,i,:]) for i in range(d))
            

            for pos in range(1, n_comps):
                d = A.dims[pos]
                rA = A.comps[pos].shape[0]
                rB = B.comps[pos].shape[0]
                v = sum (v @ torch.einsum('ij,kl -> ikjl', A.comps[pos][:,i,:], B.comps[pos][:,i,:]).reshape(rA*rB,-1) for i in range(d))
                #v = sum(v @ torch.kron(A.comps[pos][:,i,:].contiguous(), B.comps[pos][:,i,:].contiguous()) for i in range(d))  # annoying contiguous
            return v

        elif isinstance(A, list) and isinstance(B, list):
            assert len(A) == len(B) 
            for c_A, c_B in zip(A, B):
                assert c_A.shape[1] == c_B.shape[1]

            n_comps = len(A) 
            d = A[0].shape[1]
            v = sum(torch.kron(A[0][:,i,:], B[0][:,i,:]) for i in range(d))
            

            for mu in range(1, n_comps):
                d = A[mu].shape[1]
                rA = A[mu].shape[0]
                rB = B[mu].shape[0]
                v = sum (v @ torch.einsum('ij,kl -> ikjl', A[mu][:,i,:], B[mu][:,i,:]).reshape(rA*rB,-1) for i in range(d))
                #v = sum(v @ torch.kron(A.comps[pos][:,i,:].contiguous(), B.comps[pos][:,i,:].contiguous()) for i in range(d))  # annoying contiguous
            return v


        else: 
            raise NotImplementedError("Only TensorTrain/TensorTrain  or component_list/component_list implemented.")
    

    # TODO: rename
    @staticmethod
    def skp(A,B):
        return TensorTrain.hadamard_product(A,B)

    @staticmethod
    def frob_norm(A):
        return torch.sqrt(TensorTrain.skp(A,A))

    @staticmethod
    def hsvd(A_full, ranks = None):
        """
        Obtains a TensorTrain from a full tensor via high order svd
        """
        d = len(A_full.shape)
        shapes = A_full.shape
        A_mat = A_full

        # if no ranks are provided, choose maximum possible ranks
        if ranks is None: 
            ranks = [1] + [min(prod(shapes[:mu+1]), prod(shapes[mu+1:])) for mu in range(d-1)] + [1]

        comps = []
        for mu in range(d-1):
            # A_mat = A_mat.reshape((prod(shapes[:mu+1]),-1))  # matrification
            A_mat = A_mat.reshape((ranks[mu]*shapes[mu],-1))
            u, sigma, vt = torch.linalg.svd(A_mat)
            # truncatioN: 
            u, sigma, vt = u[:,:ranks[mu+1]], sigma[:ranks[mu+1]], vt[:ranks[mu+1],:]

            u_comp = u.reshape(ranks[mu], shapes[mu], ranks[mu+1])
            comps.append(u_comp)

            A_mat = torch.diag(sigma) @ vt

        comps.append(A_mat.unsqueeze(2))
        return comps



    def set_components(self, comp_list):
        """ 
           @param comp_list: List of order 3 tensors representing the component tensors
                            = [C1, ..., Cd] with shape
                            Ci.shape = (ri, self.dims[i], ri+1)
                            
                            with convention r0 = rd = 1

        """
        # the length of the component list has to match 
        assert(len(comp_list) == self.n_comps)
    
        # each component must be a order 3 tensor object
        for pos in range(self.n_comps):
            assert(len(comp_list[pos].shape)==3)
        
        # the given components inner dimension must match the predefined fixed dimensions
        for pos in range(self.n_comps):
            assert(comp_list[pos].shape[1] == self.dims[pos])
            
        # neibourhood communication via rank size must match
        for pos in range(self.n_comps-1):
            assert(comp_list[pos].shape[2] == comp_list[pos+1].shape[0])

        # setting the components
        for pos in range(self.n_comps):
            self.comps[pos] = deepcopy(comp_list[pos])

    def fill_random(self, ranks, eps):
        """
            Fills the TensorTrain with random elements for a given structure of ranks.
            If entries in the TensorTrain object have been setted priviously, they are overwritten 
            regardless of the existing rank structure.

            @param ranks #type list
        """
        self.rank = ranks
        
        for pos in range(self.n_comps):
            self.comps[pos] = eps * torch.rand(self.rank[pos], self.dims[pos], self.rank[pos+1])
        
    def full(self):
        """
            Obtain the underlying full tensor. 

            WARNING: This can become abitrary slow and may exceed memory.
        """
        res = torch.zeros((self.dims))
        for idx in product(*[list(range(d)) for d in self.dims]):  
            val = torch.tensor([1.]) 
            for k, c in enumerate(self.comps):
                val = torch.matmul(val, c[:,idx[k],:].reshape(c.shape[0],-1))
            res[idx] = val

        return res

    def __shift_to_right(self,pos, variant):
        with TicToc(key=" o right shifts", do_print=False, accumulate=True, sec_key="Core Moves:"):
            c = self.comps[pos]
            s = c.shape
            c = left_unfolding(c)
            if variant == 'qr': 
                q, r = torch.linalg.qr(c) 
                self.comps[pos] = q.reshape(s[0],s[1],q.shape[1])
                self.comps[pos+1] = torch.einsum('ij, jkl->ikl ', r, self.comps[pos+1] )
            else : # variant == 'svd'
                u, S, vh = torch.linalg.svd(c,  full_matrices=False)
                u, S, vh = u[:,:len(S)], S[:len(S)], vh[:len(S),:]

                # store orthonormal part at current position
                self.comps[pos] = u.reshape(s[0],s[1],u.shape[1])
                self.comps[pos+1] = torch.einsum('ij, jkl->ikl ', torch.diag(S)@vh, self.comps[pos+1] )
            

    def __shift_to_left(self, pos, variant):
        with TicToc(key=" o left shifts", do_print=False, accumulate=True, sec_key="Core Moves:"):
            c = self.comps[pos]
        
            s = c.shape
            c = right_unfolding(c)
            if variant == 'qr':
                q, r = torch.linalg.qr(torch.transpose(c,1,0)) 
                qT = torch.transpose(q,1,0)
                self.comps[pos] = qT.reshape(qT.shape[0],s[1],s[2]) # refolding
                self.comps[pos-1] = torch.einsum('ijk, kl->ijl ', self.comps[pos-1], torch.transpose(r,1,0))

            else: # perform svd
                u, S, vh = torch.linalg.svd(c, full_matrices = False)
                # store orthonormal part at current position
                self.comps[pos] = vh.reshape(vh.shape[0], s[1],s[2])
                self.comps[pos-1] = torch.einsum('ijk, kl->ijl ', self.comps[pos-1], u@torch.diag(S) )

    def set_core(self, mu, variant = 'qr'):

        cc = [] # changes components

        if self.core_position is None:
            assert(variant in ['qr', 'svd'])
            self.core_position = mu
            # from left to right shift of the non-orthogonal component
            for pos in range(0, mu):
                self.__shift_to_right(pos, variant)
            # right to left shift of the non-orthogonal component          
            for pos in range(self.n_comps-1, mu, -1):
                self.__shift_to_left(pos, variant)
            #self.rank[mu+1] = self.comps[mu].shape[2]

            cc= list(range(self.n_comps))

        else:
            while self.core_position > mu:
                cc.append(self.core_position)
                self.shift_core('left')
            while self.core_position < mu:
                cc.append(self.core_position)
                self.shift_core('right')

            cc.append(mu)

        assert(self.comps[0].shape[0] == 1 and self.comps[-1].shape[2] == 1)

        self.rank = [1] + [self.comps[pos].shape[2] for pos in range(self.n_comps)] 
        return cc
   
    def shift_core(self, direction, variant = 'qr'):
        assert( direction in [-1,1,'left','right'])
        assert(self.core_position is not None)

        if direction == 'left':    shift = -1
        elif direction == 'right': shift = 1
        else:                      shift = direction
        # current core position
        mu = self.core_position
        if shift == 1:
            self.__shift_to_right(mu, variant)
        else:
            self.__shift_to_left(mu, variant)
        
        self.core_position += shift
  
    def dot_rank_one(self, rank1obj):
        """ 
          Implements the multidimensional contraction of the underlying Tensor Train object
          with a rank 1 object being product of vectors of sizes di 
          @param rank1obj: a list of vectors [vi i = 0, ..., modes-1] with len(vi)=di
                           vi is of shape (b,di) with bi > 0
        """
        with TicToc(key=" o dot rank one ", do_print=False, accumulate=True, sec_key="TT application:"):
            # the number of vectors must match the component number
            assert(len(rank1obj) == self.n_comps)
            for pos in range(0, self.n_comps):
                # match of inner dimension with respective vector size
                assert(self.comps[pos].shape[1] == rank1obj[pos].shape[1])
                # vectors must be 2d objects 
                assert(len(rank1obj[pos].shape) == 2)
            
            G = [ torch.einsum('ijk, bj->ibk', c, v)  for  c,v in zip(self.comps, rank1obj) ]  
            #print(G)
            res = G[-1]
            # contract from right to left # TODO here we assume row-wise memory allocation of matrices in G
            for pos in range(self.n_comps-2, -1,-1):
                # contraction w.r.t. the 3d coordinate of G[pos]
                #res = lb.dot(G[pos], res)
                res = torch.einsum('ibj, jbk -> ibk', G[pos], res) # k = 1 only
            # res is of shape b x 1
            return res.reshape(res.shape[1], res.shape[2])
    

    def rank_truncation(self, max_ranks):

        if self.core_position != 0:
           self.set_core(0)

        for pos in range(self.n_comps-1):
            c = self.comps[pos]
            s = c.shape

            c = c.reshape(s[0]*s[1], s[2])
            u, sigma, vt = torch.linalg.svd(c, full_matrices=False)
            new_rank = max_ranks[pos+1]
            k = u.shape[1]

            # update informations
            u, sigma, vt = u[:,:new_rank], sigma[:new_rank], vt[:new_rank,:]

            new_shape = (s[0], s[1], min(new_rank,k))

            self.comps[pos] = u.reshape(new_shape)

            self.comps[pos+1] = torch.einsum('ir, rkl->ikl ', torch.matmul(torch.diag(sigma),vt), self.comps[pos+1] ) # Stimmt das noch ?

        self.core_position = self.n_comps-1
        assert(self.comps[0].shape[0] == 1 and self.comps[-1].shape[2] == 1)
        self.rank = [1] + [self.comps[pos].shape[2] for pos in range(self.n_comps-1)] + [1] 


    def round(self, delta, verbose = False):

        rank_changed = False

        self.set_core(0)
        rule = Threshold(delta)
        for pos in range(self.n_comps-1):
            c = self.comps[pos]
            s = c.shape
            c = c.reshape(s[0]*s[1], s[2])
            u, sigma, vt = torch.linalg.svd(c, full_matrices=False) 
            new_rank =  rule(u, sigma, vt, pos) 

            # update informations
            u, sigma, vt = u[:,:new_rank], sigma[:new_rank], vt[:new_rank,:]
            new_shape = (s[0], s[1], min(new_rank,s[2]))
            self.comps[pos] = u.reshape(new_shape)

            ldtype = common_field_dtype([self.comps[pos+1]], [sigma, vt])
            self.comps[pos+1] = torch.einsum('ir, rkl->ikl ', torch.diag(sigma).type(ldtype) @ vt.type(ldtype), self.comps[pos+1].type(ldtype) ) # Stimmt das noch ?

        self.core_position = self.n_comps-1
        assert(self.comps[0].shape[0] == 1 and self.comps[-1].shape[2] == 1)

        if verbose and self.rank is not None:
            for mu, c in enumerate(self.comps[:-1]):
                if self.rank[mu+1] > c.shape[2]: 
                    print(f" {Fore.GREEN} A rank changed : {Style.RESET_ALL}  \
                                {Fore.BLUE} r_{mu} :  {self.rank[mu+1]} -> {c.shape[2]}{Style.RESET_ALL}")
                    rank_changed = True
                    time.sleep(1)  
        
        # update the rank
        self.rank = [1] + [self.comps[pos].shape[2] for pos in range(self.n_comps-1)] + [1]
        if verbose:
            print('New rank is ', self.rank)
            # time.sleep(1) 

        return rank_changed



def generate_TT_from_gauss(d,Sigma=None):
    Sigma = torch.linalg.inv(Sigma) # get inverse of the covariance matrix
    if Sigma == None:
        Sigma = torch.randn(d,d)
        Sigma = 0.5* (Sigma+Sigma.T)
    assert Sigma.shape[0] == d
    assert Sigma.shape[1] == d
    #lam = torch.linalg.eigvals(Sigma)
    eps = 1e-6
    Sigma += eps * torch.eye(d)

    full_A = torch.zeros([3]*d)

    for i in range(d):
        for j in range(d):
            idx = [0]*d 
            idx[i] +=1
            idx[j] +=1 
            full_A[tuple(idx)] += Sigma[i,j]

    TT = TensorTrain.hsvd(full_A)

    return TT, Sigma

def get_CovMatrix_from_TT(TT):

    comps = TT.comps

    d = len(comps)
    Sigma = torch.zeros(d,d)

    for i in range(d):
        for j in range(i, d):
            idx = [0]*d 
            idx[i] +=1
            idx[j] +=1 
            # TODO: Avoid Full Tensor computation, since the input is a TT 
            val = torch.tensor([1.])
            for ii, c in zip(idx, comps):
                val = val @ c[:,ii,:]
            Sigma[i,j] = val[0]
    
    Sigma = 0.5*(Sigma + Sigma.T)
        
    return Sigma


def get_CovMatrix_from_TT_list(TT_list):
    dims = [c.shape[1] for c in TT_list]
    TT = TensorTrain(dims, TT_list)
    return get_CovMatrix_from_TT(TT)




def power_iteration(LinOpt, x_init, K = 1000, ACC = 1, verbose = False):

    def break_criteria(lams, ACC=ACC):
        """ Returns True if the first ACC+1 nonzero values are equal between last two lams."""
        
        if len(lams) > 2: 
            lam1, lam2 = torch.abs(lams[-2]), torch.abs(lams[-1])
            k1, k2     = torch.ceil( -torch.log10(lam1) ) , torch.ceil( -torch.log10(lam2) )      

            if k1 == k2: 
                if torch.round(lam1 * 10**k1, decimals = ACC) - torch.round(lam2 * 10**k2, decimals = ACC) == torch.tensor(0.) : 
                    return True, k1 
        return False


    xk = x_init
    lams = []

    for k in range(K):


        xk.comps[0] /= TensorTrain.frob_norm(xk)
        xkp1 = LinOpt(xk)
        xkp1.rank_truncation(x_init.rank)

        lamk = TensorTrain.skp(xk, xkp1)
        xk = xkp1
        lams.append(lamk)

        if not break_criteria(lams, ACC):
            continue
        else:
            _, k_power = break_criteria(lams, ACC)
            lamk = abs(lamk) + 1. / 10**(k_power+ACC)
            break


    xkp1 = LinOpt(xk)
    xkp1.rank_truncation(x_init.rank)

    if verbose:
        print(f"Eigenvalue: {Fore.BLUE} {lamk.item()} {Style.RESET_ALL}  found at iteration {Fore.BLUE} {k} {Style.RESET_ALL}")
    
    if k == K-1:
        converged = False
    else:
        converged = True
    
    return xk, lamk.item(), converged





































