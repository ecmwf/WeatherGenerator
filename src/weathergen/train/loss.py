
import code

import numpy as np
import torch


####################################################################################################
def Gaussian( x, mu=0., std_dev=1.) :
  # unnormalized Gaussian where maximum is one
  return torch.exp( -0.5 * (x-mu)*(x-mu) / (std_dev*std_dev))

####################################################################################################
def normalized_Gaussian( x, mu=0., std_dev=1.) :
  return (1 / (std_dev*np.sqrt(2.*np.pi))) * torch.exp( -0.5 * (x-mu)*(x-mu) / (std_dev*std_dev))

def erf( x, mu=0., std_dev=1.) :
  c1 = torch.sqrt( torch.tensor(0.5 * np.pi) )
  c2 = torch.sqrt( 1. / torch.tensor(std_dev * std_dev))
  c3 = torch.sqrt( torch.tensor( 2.) )
  val = c1 * ( 1./c2 - std_dev * torch.special.erf( (mu - x) / (c3 * std_dev) ) )
  return val

####################################################################################################
def gaussian_crps( target, ens, mu, stddev) :
  # see Eq. A2 in S. Rasp and S. Lerch. Neural networks for postprocessing ensemble weather
  # forecasts. Monthly Weather Review, 146(11):3885 – 3900, 2018.
  c1 = np.sqrt(1./np.pi)
  t1 = 2. * erf( (target-mu) / stddev) - 1.
  t2 = 2. * normalized_Gaussian( (target-mu) / stddev)
  val = stddev * ( (target-mu)/stddev * t1 + t2 - c1 )
  return torch.mean(val) # + torch.mean( torch.sqrt( stddev) )

####################################################################################################
def stats( target, ens, mu, stddev) :
  diff = Gaussian( target, mu, stddev) - 1.
  return torch.mean( diff * diff) + torch.mean( torch.sqrt( stddev) )
          
####################################################################################################
def stats_normalized( target, ens, mu, stddev) :
  a = normalized_Gaussian( target, mu, stddev)
  max = 1 / (np.sqrt(2 * np.pi) * stddev)
  d = a - max
  return torch.mean( d*d) + torch.mean( torch.sqrt( stddev) )

####################################################################################################
def stats_normalized_erf( target, ens, mu, stddev) :
  delta = -torch.abs(target - mu)
  d = 0.5 + torch.special.erf(delta / (np.sqrt(2.) * stddev))
  return torch.mean( d*d) #+ torch.mean( torch.sqrt( stddev) )

####################################################################################################
def mse( target, ens, mu, *kwargs) :
  return torch.nn.functional.mse_loss( target, mu)

####################################################################################################
def mse_ens( target, ens, mu, stddev) :
  mse_loss = torch.nn.functional.mse_loss
  return torch.stack( [mse_loss( target, mem) for mem in ens], 0).mean()

####################################################################################################
def kernel_crps( target, ens, mu, stddev, fair = True) :

  ens_size = ens.shape[0]
  mae = torch.stack( [(target - mem).abs().mean() for mem in ens], 0).mean()

  if ens_size == 1:
    return mae

  coef = -1.0 / (2.0 * ens_size * (ens_size - 1)) if fair else -1.0 / (2.0 * ens_size**2)
  ens_var = coef * torch.tensor( [(p1 - p2).abs().sum() for p1 in ens for p2 in ens]).sum()
  ens_var /= ens.shape[1]

  return mae + ens_var

