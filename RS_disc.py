import torch as th
import numpy as np

from torch.autograd import Variable
from torch.nn.parameter import Parameter

from torch.optim.lr_scheduler import ReduceLROnPlateau

if th.cuda.is_available():
    dtype = th.cuda.DoubleTensor
else:
    dtype = th.DoubleTensor
    
    
##---Neural Network architecture------#####
class TurbNN(th.nn.Module):

    def __init__(self, D_in, H, D_out):
        """
        Architecture of the turbulence deep neural net
        Args:
            D_in (Int) = Number of input parameters
            H (Int) = Number of hidden paramters
            D_out (Int) = Number of output parameters
        """
        super(TurbNN, self).__init__()
        self.linear1 = th.nn.Linear(D_in, H)
        self.f1 = th.nn.LeakyReLU()
        self.linear2 = th.nn.Linear(H, H)
        self.f2 = th.nn.LeakyReLU()
        self.linear3 = th.nn.Linear(H, H)
        self.f3 = th.nn.LeakyReLU()
        self.linear4 = th.nn.Linear(H, H)
        self.f4 = th.nn.LeakyReLU()
        self.linear5 = th.nn.Linear(H, int(H/5))
        self.f5 = th.nn.LeakyReLU()
        self.linear6 = th.nn.Linear(int(H/5), int(H/10))
        self.f6 = th.nn.LeakyReLU()
        self.linear7 = th.nn.Linear(int(H/10), D_out)

    def forward(self, x):
        """
        Forward pass of the neural network
        Args:
            x (th.DoubleTensor): [N x D_in] column matrix of training inputs
        Returns:
            out (th.DoubleTensor): [N x D_out] matrix of neural network outputs
        """
        lin1 = self.f1(self.linear1(x))
        lin2 = self.f2(self.linear2(lin1))
        lin3 = self.f3(self.linear3(lin2))
        lin4 = self.f4(self.linear4(lin3))
        lin5 = self.f5(self.linear5(lin4))
        lin6 = self.f6(self.linear6(lin5))
        out = self.linear7(lin6)

        return out # these would the the 10 G value which acts acts coeff to the 10 tensor invarients
        
        

class Invariant():
    """
    Class used for invariant calculations
    """
    def getInvariants(self, s0, r0):
        """
        Calculates the invariant neural network inputs
        Args:
            s0 (DoubleTensor): [nCellx3x3] Rate-of-strain tensor -> 0.5*(k/e)(du + du')
            r0 (DoubleTensor): [nCellx3x3] Rotational tensor -> 0.5*(k/e)(du - du')
        Returns:
            invar (DoubleTensor): [nCellx5x1] Tensor containing the 5 invariant NN inputs
        """
        # Invariant Training inputs
        # For equations see Eq. 14 in paper
        # SB Pope 1975 (http://doi.org/10.1017/S0022112075003382)
        # Or Section 11.9.2 (page 453) of Turbulent Flows by SB Pope
        nCells = s0.size()[0]
        invar = th.DoubleTensor(nCells, 5).type(dtype)

        s2 = s0.bmm(s0)
        r2 = r0.bmm(r0)
        s3 = s2.bmm(s0)
        r2s = r2.bmm(s0)
        r2s2 = r2.bmm(s2)

        invar[:,0] = (s2[:,0,0]+s2[:,1,1]+s2[:,2,2]) #Tr(s2)
        invar[:,1] = (r2[:,0,0]+r2[:,1,1]+r2[:,2,2]) #Tr(r2)
        invar[:,2] = (s3[:,0,0]+s3[:,1,1]+s3[:,2,2]) #Tr(s3)
        invar[:,3] = (r2s[:,0,0]+r2s[:,1,1]+r2s[:,2,2]) #Tr(r2s)
        invar[:,4] = (r2s2[:,0,0]+r2s2[:,1,1]+r2s2[:,2,2]) #Tr(r2s2)

        # Scale invariants by sigmoid function
        # Can use other scalings here
        invar_sig = (1.0 - th.exp(-invar))/(1.0 + th.exp(-invar))
        invar_sig[invar_sig != invar_sig] = 0
        
        return invar_sig

    def getTensorFunctions(self, s0, r0):
        """
        Calculates the linear independent tensor functions for calculating the
        deviatoric  component of the Reynolds stress. Ref: S. Pope 1975 in JFM
        Args:
            s0 (DoubleTensor): [nCellsx3x3] Rate-of-strain tensor -> 0.5*(k/e)(du + du')
            r0 (DoubleTensor): [nCellsx3x3] Rotational tensor -> 0.5*(k/e)(du - du')
        Returns:
            invar (DoubleTensor): [nCellsx10x[3x3]] Tensor 10 linear independent functions
        """
        # Invariant Training inputs
        # For equations see Eq. 15 in paper
        # Or SB Pope 1975 (http://doi.org/10.1017/S0022112075003382)
        # Or Section 11.9.2 (page 453) of Turbulent Flows by SB Pope
        nCells = s0.size()[0]
        invar_func = th.DoubleTensor(nCells,10,3,3).type(dtype)

        s2 = s0.bmm(s0)
        r2 = r0.bmm(r0)
        sr = s0.bmm(r0)
        rs = r0.bmm(s0)

        invar_func[:,0] = s0
        invar_func[:,1] = sr - rs
        invar_func[:,2] = s2 - (1.0/3.0)*th.eye(3).type(dtype)*(s2[:,0,0]+s2[:,1,1]+s2[:,2,2]).unsqueeze(1).unsqueeze(1)
        invar_func[:,3] = r2 - (1.0/3.0)*th.eye(3).type(dtype)*(r2[:,0,0]+r2[:,1,1]+r2[:,2,2]).unsqueeze(1).unsqueeze(1)
        invar_func[:,4] = r0.bmm(s2) - s2.bmm(r0)
        t0 = s0.bmm(r2)
        invar_func[:,5] = r2.bmm(s0) + s0.bmm(r2) - (2.0/3.0)*th.eye(3).type(dtype)*(t0[:,0,0]+t0[:,1,1]+t0[:,2,2]).unsqueeze(1).unsqueeze(1)
        invar_func[:,6] = rs.bmm(r2) - r2.bmm(sr)
        invar_func[:,7] = sr.bmm(s2) - s2.bmm(rs)
        t0 = s2.bmm(r2)
        invar_func[:,8] = r2.bmm(s2) + s2.bmm(r2) - (2.0/3.0)*th.eye(3).type(dtype)*(t0[:,0,0]+t0[:,1,1]+t0[:,2,2]).unsqueeze(1).unsqueeze(1)
        invar_func[:,9] = r0.bmm(s2).bmm(r2) + r2.bmm(s2).bmm(r0)

        # Scale the tensor basis functions by the L2 norm
        l2_norm = th.DoubleTensor(invar_func.size(0), 10)
        l2_norm = 0
        for (i, j), x in np.ndenumerate(np.zeros((3,3))):
            l2_norm += th.pow(invar_func[:,:,i,j],2)
        invar_func = invar_func/th.sqrt(l2_norm).unsqueeze(2).unsqueeze(3)

        return invar_func

class InvarientNN():
    
    """
    Class for formulating Bayesian posterior, training and prediction
    """
    def __init__(self,S,R,k,b_avg):
        self.S=S # Strain rate tensor
        self.R=R # Rotation rate tensor
        self.k=k # turbulent kinetic energy
        self.b_avg=b_avg # RS term anisotropic tensor
        
        nData=S.size()[0]
        self.x_train = th.Tensor(nData,5).type(dtype) # Invariant inputs
        self.t_train = th.Tensor(nData,10,3,3).type(dtype) # Tensor basis
        self.k_train = th.Tensor(nData).type(dtype) # RANS TKE
        self.y_train = th.Tensor(nData,3,3).type(dtype) # Target output
        
        Invar=Invariant()
        self.x_train=Invar.getInvariants(S,R)
        self.t_train=Invar.getTensorFunctions(S,R)   
        self.k_train=k
        self.y_train=b_avg.view(-1,3,3)  ##[7096, 9]
        
    
        self.turb_nn = TurbNN(D_in=5, H=200, D_out=10).double() #Construct neural network        
        # Get invarients
        
        # Student's t-distribution: w ~ St(w | mu=0, lambda=shape/rate, nu=2*shape)
        # See PRML by Bishop Page 103
        self.prior_w_shape = 1.0
        self.prior_w_rate = 0.025
        # noise variance: beta ~ Gamma(beta | shape, rate)
        self.prior_beta_shape = 100
        self.prior_beta_rate = 2e-4
        
        
        # Now set up parameters for the noise hyper-prior
        #beta_size = (self.n_samples, 1)
        beta_size=1
        log_beta = np.log(np.random.gamma(self.prior_beta_shape,
                        1. / self.prior_beta_rate, size=beta_size))
        # Log of the additive output-wise noise (beta)
      
        self.turb_nn.log_beta = Parameter(th.Tensor(log_beta).type(dtype)) ## AA: have to see

        # Network weights learning weight
        lr = 1e-3 #Original 1e-05
        # Output-wise noise learning weight
        lr_noise=0.001
     
        # Pre-pend output-wise noise to model parameter list
        parameters = [{'params': [self.turb_nn.log_beta], 'lr': lr_noise},  ## AA: have to see
                  {'params': [p for n, p in self.turb_nn.named_parameters() if n!='log_beta']}]
        # ADAM optimizer (minor weight decay)
        self.optim = th.optim.Adam(parameters, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
        # Decay learning weight on plateau, can adjust these parameters depending on data
        self.scheduler = ReduceLROnPlateau(self.optim, mode='min', factor=0.75, patience=3,
            verbose=True, threshold=0.05, threshold_mode='rel', cooldown=5, min_lr=0, eps=1e-07)

    def forward(self, input, t_data):
        """
        Computes all the `n_samples` NN output
        Args: 
            input (Tensor): [nx5] tensor of input invariants
            t_data (Tensor): [nx10x3x3] tensor of linear independent tensore basis functions
        Return: out (Tensor): [nx3x3] tensor of predicted scaled anisotropic terms
        """
        out_size = (3, 3)
        output = Variable(th.Tensor(input.size(0), *out_size).type(dtype))
        #output=self.turb_nn.forward(input)
        g_pred = self.turb_nn.forward(input)
        g_pred0 = th.unsqueeze(th.unsqueeze(g_pred, 2), 3)  ## AA: have to see, also is the n from SVGD or the no cells
        output = th.sum(g_pred0*t_data,1)
        return output
      
    def BayesianPosterior(self,output,target):
        """
        Computer posterior value and its gradients
        Args:
            output: [Nx3x3] Forward pass of the Invarient Neural Network (b_pred)
            Target: [Nx3x3] LES b values
        Return:
            posterior: (scalar). Value of the posterior
            grads:list containing grad of each parameter set {beta, w_1,w_2...}
        """
        log_likelihood= (-0.5 * self.turb_nn.log_beta.exp()              ## N_total/N_batch can be added at the start to accomodate for mini batching. The wieght of the likelihood needs to be adjusted as per the mini batch
                                * (target - output).pow(2).sum()
                                + 0.5 * target.numel()
                                * self.turb_nn.log_beta)
         # Log Gaussian weight prior
            # See Eq. 17 in paper
        prior_ws = Variable(th.Tensor([0]).type(dtype))
        for param in self.turb_nn.parameters():
            prior_ws += th.log1p(0.5 / self.prior_w_rate * param.pow(2)).sum()
        prior_ws *= -(self.prior_w_shape + 0.5)

        # Log Gamma Output-wise noise prior
        # See Eq. 20 in paper
        prior_log_beta = (self.prior_beta_shape * self.turb_nn.log_beta \
                              - self.turb_nn.log_beta.exp() * self.prior_beta_rate).sum()
        posterior=log_likelihood + prior_ws + prior_log_beta
        #posterior.backward()
        #grads = []
        #for param in self.turb_nn.parameters():
        #    grads.append(param.grad)
        #grads = th.cat(grads)
        #return posterior, grads
        return posterior, log_likelihood.data.item()
    
    def loh_h_MCMC(self):
        b_pred=self.forward(self.x_train,self.t_train)
        posterior=self.BayesianPosterior(b_pred,self.y_train)
        posterior.backward()
        grads = []
        for param in self.turb_nn.parameters():
            grads.append(param.grad)
        return posterior,grads
        
    
    def MAP_train(self, n_epoch=200, gpu=True):
        """
        for epoch in range(n_epoch):
            training_loss=0
            training_MSE=0
            N_total=self.x_train.size()[0]
            perm=np.random.permutation(N_total)
            
            batch_size=N_total/100
            for it in range(0,N_total,batch_size)
                idx=perm[np.arange(it,it+batch_size)]
                x_batch=self.x_train[idx,:,:]
                y_batch=self.y_train[idx,:,:]
                t_batch=self.t_train[idx,:,:]
                self.optim.zero_grad()
                b_pred=self.forward(x_batch,t_batch)
                posterior,log_likelihood=self.BayesianPosterior(b_pred,y_batch)
                J=-posterior
                J.backward()
                self.optim.step()
            
                training_loss += posterior.data.item()
                # Scaled MSE
                training_MSE += (1/b_pred.size()[0])((self.y_train - b_pred) ** 2).sum().data.item()
                print("===> Epoch: {}, Current loss: {:.6f} Log Beta: {:.6f} Scaled-MSE: {:.6f}".format(
                    epoch + 1, training_loss, self.turb_nn.log_beta.data.item(), training_MSE))
        """
        for epoch in range(n_epoch):
                self.optim.zero_grad()
                b_pred=self.forward(self.x_train,self.t_train)
                posterior,log_likelihood=self.BayesianPosterior(b_pred,self.y_train)
                J=-posterior
                J.backward()
                self.optim.step()
                
                training_loss = posterior.data.item()
                # Scaled MSE
                loss=th.nn.MSELoss()
                #training_MSE = loss(self.y_train,b_pred)/(self.y_train**2).mean() #||b_hat-b_LES||^2/||b_LES||^2
                training_MSE = (((self.y_train-b_pred)**2).mean())/(self.y_train**2).mean()
                if epoch%10==0:
                    print("===> Epoch: {}, Posterior: {:.6f} Log Beta: {:.6f} Scaled-MSE: {:.6f}".format(
                    epoch + 1, posterior.data.item(), self.turb_nn.log_beta.data.item(), training_MSE))
    
    def predict(self,ransS,ransR):
        """
        After training does a forward pass 
        Args:
            ransS (Tensor): [nCellx3x3] Baseline RANS rate-of-strain tensor
            ransR (Tensor): [nCellx3x3] Baseline RANS rotation tensor
        return:
            b_pred: The predicted value of the b    
        """
        temp=Invariant()
        x_train=temp.getInvariants(ransS,ransR)
        t_train=temp.getTensorFunctions(ransS,ransR)
        
        b_pred=self.forward(x_train,t_train)
        # error computation, need to input DNS/LES only for predicted domain. full LES imported for the time being
        loss=th.nn.MSELoss()
        error_pred=loss(self.y_train,b_pred)
        
        return b_pred,error_pred
        
if __name__ == "__main__":
    dtype = th.DoubleTensor
    S=th.load('training-data/periodic-hills/RANS/90/S-torch.th')
    R=th.load('training-data/periodic-hills/RANS/90/R-torch.th')
    k0=th.load('training-data/periodic-hills/RANS/90/k-torch.th')
    RS_avg=th.load('training-data/periodic-hills/LES/1000/UPrime2Mean-torch.th')
     
    RS_avg=RS_avg.view(RS_avg.size()[0],3,-1)
    k = k0.unsqueeze(0).unsqueeze(0).expand(3,3,k0.size()[0]) ## Two unsqueeze made it from 7096 to 1,1,7096
    k = k.permute(2, 0, 1) ## makes it 7096x3x3x from 3x3x7096
    b_avg = RS_avg/2*k - (2.0/3.0)*th.eye(3).type(dtype)  ## torch.Size([7096, 3, 3])
    
    Invar=InvarientNN(S,R,k0,b_avg)
    
    Invar.MAP_train()
    
    b_pred,error_prediction=Invar.predict(S,R) # NOTE: for simplicity, precition and testing dataset is same for the time being


