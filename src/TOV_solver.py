#!/usr/bin/env python
# coding: utf-8

# In[4]:


from joblib import Parallel, delayed
from scipy.integrate import solve_ivp
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import sympy as sp
import matplotlib.pyplot as plt
from scipy.optimize import root
import random
from scipy.stats import norm, uniform
from PIL import Image
import statistics
import os
import time


# In[5]:


#constants
c_light = 2.997*1e10
hbar = 6.582*1e-22
MeV = 1.602*1e-6
Kp = MeV / (hbar**3 * c_light**3)
Krho = MeV/(hbar**3 * c_light**5)
fm = (1e-13)**(-3)
n0 = 0.16 * fm
m = 1.675 * 1e-24
M_sun = 1.988 * 1e33
G_const = 6.67*1e-11*1e6/1e3
Lu = (1 / M_sun) * (c_light ** 2) * (1/G_const)
Pu = (1/M_sun) / (Lu ** 3) / (c_light **2)
rhou = (1/ M_sun ) / (Lu **3)
KK = ((2.997*1e5)**2)/(6.67*1e-20 * 1.988 *1e30)


# In[6]:


rho_t = 2*n0*m*rhou
rho_fin = 12*n0*m*rhou
pc = 200**4 * Kp * Pu
r0 = 1e-5
a0=1
f0=1
h0=1
H0 = a0 * r0**2
beta0 = 2 * a0 * r0
rspan = (r0, 200)
Lamb_arr = [-(194.0) * Kp * Pu, -(150.0) * Kp * Pu, -(120.0) * Kp * Pu, -(95.0) * Kp * Pu,
    -(50.0) * Kp * Pu, 0, (50.0)
            * Kp * Pu, (95.0) * Kp * Pu, (120.0) * Kp * Pu,
    (194.0) * Kp * Pu]

M_norm = 3
R_norm = 20


# In[7]:


ap4 = pd.read_csv("Rescaledap4.dat", sep = r'\s+', header=None)
sly = pd.read_csv("Rescaledsly.dat", sep = r'\s+', header=None)
ap4.columns=['p', 'eps', 'rho']
sly.columns=['p', 'eps', 'rho']

#training data
rho_train = pd.read_csv("matrixrho2.dat", sep = r'\s+', header=None)
cs_train = pd.read_csv("matrixcs2.dat", sep = r'\s+', header=None)
rho_train.columns=['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7']
cs_train.columns=['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7']

#test data
rho_test = pd.read_csv("matrixrhoTest.dat", sep = r'\s+', header=None)
cs_test = pd.read_csv("matrixcsTest.dat", sep = r'\s+', header=None)
rho_test.columns=['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7']
cs_test.columns=['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7']

sly = sly.values
ap4 = ap4.values

rho_train = rho_train.values
cs_train = cs_train.values
rho_test = rho_test.values
cs_test = cs_test.values


# In[8]:


def find_pos(matrix, val, col):
  """
  Find element index in a matrix column closest to a specific value.
  
  Parameters:
      matrix: ndarray
          2D array with data
      val: float         
          Value to find in column.
      col: int
          Column index of matrix where search is done.
      
  Returns:
      int
          Row index of element which value in column is closest to target.
  """
  col_data = matrix[:, col]
  for i in range(len(col_data) - 1):
    dx1 = abs(col_data[i] - val)
    dx2 = abs(col_data[i+1] - val)

    if col_data[i+1] >= val:
      if dx1<dx2:
        return i
      else:
        return i+1

def deriv(eos, max_i):
  """Compute the derivative of the EOS up to an index max_i.
  
  Parameters:
      eos: ndarray
          2D array with EOS data, with columns being parameters.
      max_i: int           
          Index for which we compute derivative up to.

  Returns:
      s: float
          Derivative of pressure with respect to the energy density.
  """
  s = 0
  for i in range(max_i):
    dp = eos[i+1,0] - eos[i,0]
    deps = eos[i+1,1] - eos[i,1]
    s = dp / deps
  return s

def cs_interpolate(eos, max_i, p1, matrix_rho, matrix_cs, ds):
  """ 
  Create linear interpolation for speed of sound c_s for 7 points.
  
  Parameters:
      eos: ndarray
          Equation of state.
      max_i: int
          Index for which EOS interpolation starts.
      p1: float
          pressure at first interpolation point
      matrix_rho: ndarray
          Matrix with density data.
      matrix_cs: ndarray
          Matrix with speed of sound data.
      ds: int
          Row index which selects interval of rho and speed of sound values.

  Returns
      cs_interpolation
          Function which interpolates the speed of sound given datasets.
  """
  segment_mat = []
  segment_mat.append([eos[max_i, 2], np.sqrt(p1)])

  for i in range(7):
    segment_mat.append([matrix_rho[ds, i], matrix_cs[ds, i]])

  segment_mat = np.array(segment_mat)
  cs_interpolation = interp1d(segment_mat[:,0], segment_mat[:,1], kind='linear', fill_value='extrapolate')

  return cs_interpolation

def EOS_HE(he_eos, eos, max_i, cs_interpolation, rho_final):
  """
  Extend the EOS table to higher densities by interpolation speed of sound. Starting from the 
  maximum index, increase the density and update energy density and pressure, using the 
  speed of sound interpolations. These new rows are appended to the dataset until we arrive 
  at the final density node.

  Parameters
      he_eos: ndarray
          EOS array which we will increase with data.
      eos: ndarray
          EOS table which initial data are obtained from.
      max_i: int
          Index from which we start in the EOS row.
      cs_interpolation: function
          Function returning speed of sound with respect to density.
      rho_final: float
          Final density up to which we determine EOS.

  Returns
      he_eos: ndarray
          EOS array which is enlargened.
  """
  drho = 1e-5

  M = int((1e-3 - eos[max_i, 2])/1e-5) + int((1e-2 - 1e-3)/1e-4) + int((rho_final - 1e-2)/1e-3)

  p_last, eps_last, rho_last = eos[max_i,0], eos[max_i,1], eos[max_i,2]

  for i in range(M-1):
    p_next = p_last + cs_interpolation(rho_last)**2 * drho * (eps_last + p_last)/rho_last
    eps_next = eps_last + drho * (eps_last + p_last) / rho_last
    rho_next = rho_last + drho

    if rho_next > rho_final:
      break

    #add row
    he_eos = np.vstack([he_eos, [p_next, eps_next, rho_next]])

    p_last, eps_last, rho_last = p_next, eps_next, rho_next

    if rho_next < 1e-3:
      drho = 1e-5
    elif rho_next < 1e-2:
      drho = 1e-4
    elif rho_next < 1e-1:
      drho = 1e-3
    else:
      drho = 1e-2

  return he_eos

def mergeEOS(eos_mat, he_eos, eos, max_i):
  """
  Add new data to original EOS matrix.

  Parameters
      eos_mat: ndarray
          Array which stores the total EOS.
      he_eos: ndarray
          Extension of EOS with new rows.
      eos: ndarray
          Initial EOS data table.
      max_i: int
          Index up to which we include rows from EOS

  Returns
      eos_mat: ndarray
          Merged EOS array.
  """
  eos_mat = np.vstack([eos_mat, eos[:max_i+1,:]])#vstack is better than the loops
  eos_mat = np.vstack([eos_mat,he_eos])
  return eos_mat


# We combine this in the following function. here we build the equation of state matrix before the lambda transition

# In[9]:


def build(eos_mat, eos, matrix_rho, matrix_cs, ds, rho_treshold, rho_final):
  """
  Create an EOS table by including higher densities to our original model. Here we find the
  transition point, apply a speed of sound interpolation and generate the EOS extension.
  Subsequently, we merge this with our original EOS data

  Parameters
      eos_mat: ndarray
          Array storing final EOS.
      eos: ndarray
          baseline EOS data.
      matrix_rho: ndarray
          Matrix with base density values used for interpolation.
      matrix_cs: ndarray
          Matrix with speed of sound values used for interpolation.
      ds: int 
          Row index for which chooses which interpolation interval to use.
      rho_treshold: float
          Density value for which we start extending EOS.
      rho_final: float
          Density up to which EOS is extended.

  Returns
      eos_matrix: ndarray
          Final EOS table including extended data.
  """
  i = find_pos(eos, rho_treshold, 2)
  p1 = deriv(eos, i)
  cs_interpolation = cs_interpolate(eos, i, p1, matrix_rho, matrix_cs, ds)

  he_eos = np.empty((0,3))
  he_eos = EOS_HE(he_eos, eos, i, cs_interpolation, rho_final)

  eos_matrix = np.empty((0,3))
  eos_matrix = mergeEOS(eos_matrix, he_eos, eos, i)

  return eos_matrix


# In[10]:


Lambda=0
def eos_interpolate(x):
  """
  Evaluate interpolated EOS energy density.

  Parameters
      x: float
          Value of the pressure.
          
  Returns
      float
      Energy density for pressure x
  """
  if x < pc:
    return eps_fluid(x)
  else:
    return eps_fluid(x+Lambda)+Lambda

def eos_prime_interpolate(x):
  """
  Find the derivative of the interpolated the EOS energy density.

  Parameters
      x: float
          Value of the pressure.
          
  Returns
      float
      Derivative of the energy density for pressure x
  """
  if x<pc:
    return eps_prime(x)
  else:
    return eps_prime(x+Lambda)


# In[11]:


def tov_equations(r,u):
  """
  Compute Tolman-Oppenheimer-Volkoff (TOV) equations. We calculate how the elements
  of u change with the radius, applying the equation of state (EOS).

  Parameters
      r: float
          Radius.
      u: list - [f,h,P,H,beta]
          List containing parameters.

  Returns
      u_array: list
          List containing the derivatives of u elements with respect to the radius.
  """
  f, h, P, H, beta = u

  eps = eos_interpolate(P)
  deps = eos_prime_interpolate(P)

  du1 = (1 - f - 8*np.pi * (r ** 2) * eps) / r
  du2 = -(h * (-1 + f - 8 * np.pi * (r ** 2) * P)) / (r * f)
  du3 = ((-1 + f - 8 * np.pi * (r**2)*P) * (P+eps)) / (2*r*f)
  du4 = beta
  du5 = (H * (- (f**3)
              + (1 + 8 * np.pi * (r**2) * P)**3
              - f * (1 + 8 * np.pi * (r**2) * P) * (-3+60*np.pi * (r**2)*P + 20 * np.pi * (r**2) * eps)
              + (f**2) * (-3
                          + 60 * np.pi * (r**2) * P
                          + 8 * np.pi * (r**3) * deps * (-1 + f - 8 * np.pi * (r**2)*P) * (P+eps)/(2 * r * f)
                          + 20 * np.pi * (r**2) * eps
          )) + r * f * (-1 + f - 8 * np.pi * (r**2) * P) * (1 + f + 4 * np.pi * (r**2) * P - 4 * np.pi * (r**2) * eps) * beta
  ) / ((r**2) * (f**2) * (1 - f + 8 * np.pi * (r**2) * P))

  return [du1, du2, du3, du4, du5]


# In[12]:


def integrator(P0):
  """
  Integrate the TOV equations from the start center to the outer core. We start from an 
  initial pressure P0 and integrate outward.

  Parameters
      P0: float
          Core neutron star pressure
          
  Returns
      solve: function
          Solution of TOV equations using solve_ivp with the Runge-Kutta 45 method.
  """
  def stop_surface(r,u):
    return u[2] / P0 - 1e-12

  stop_surface.terminal = True
  stop_surface.direction = -1

  u0 = [f0, h0, P0, H0, beta0]

  #using runge kutta method
  solve = solve_ivp(tov_equations, rspan, u0, method="RK45", max_step = 0.05,rtol = 1e-05, atol=1e-7, events=stop_surface)
  return solve


# In[13]:


def cycle_tov(data_matrix, P0, Pf):
  """
  Solve the TOV equations for different initial neutron star core pressures. Here, we run the
  TOV solver for pressure values between P0 and Pf. Subsequently, we determine our global parameters,
  which correspond to the mass, radius and love number. We store them in data_matrix.

  Parameters
      data_matrix: ndarray
          Storage array for our global parameters.
      P0: float
          Initial neutron star core pressure.
      Pf: float
          Final neutron star core pressure.
          
  Returns
      data_matrix: ndarray
          Storage array for our global parameters.
  """
  if Pf > 1e-2:
    N = int(np.floor( (1e-4 - P0) / (2.5e-6) + (1e-3 - 1e-4)/(2.5e-5) + (1e-2 - 1e-3)/(2.5e-4)))
  elif Pf > 1e-3:
    N = int(np.floor( (1e-4 - P0) / (2.5e-6) + (1e-3 - 1e-4)/(2.5e-5) + (Pf - 1e-3)/(2.5e-4)))
  elif Pf> 1e-4:
    N = int(np.floor((1e-4 - P0)/(2.5e-6) + (Pf - 1e-4)/(2.5e-5)))
  elif Pf > 1e-5:
    N = int(np.floor((Pf - P0)/(2.5e-6)))
  else:
    N = 0

  for i in range(N):
    solution = integrator(P0)
    Radius = solution.t[-1]#radius
    M = Radius / 2*(1-solution.y[0,-1])#mass
    y = Radius * solution.y[4,-1] / solution.y[3,-1]
    C = M / (Radius / Lu * 1e-5) #compactness

    #love number
    k2 = (8 * ((1 - 2 * C)**2) * (C**5) * (2 + 2 * C * (-1 + y) - y)
          ) / ( 5 * ( 2 * C * (6
                               + (C**2) * (26 - 22 * y)
                               - 3 * y
                               + 4 * (C**4) * (1 + y)
                               + 3 * C * (-8 + 5 * y)
                               + (C**3) * (-4 + 6 * y)
                               ) + 3 * ((1 - 2 * C)**2) * (2 + 2 * C * (-1 + y) - y) * np.log(1 - 2 * C)
                               ))
    lamb = (2/3) * k2 * ((Radius / Lu * 1e-5)**5) * (KK**5) / (M**5)
    row_next = np.array([[P0 / rhou, M, Radius / Lu * 1e-5, lamb]])
    data_matrix = np.vstack([data_matrix, row_next])

    if P0 < 1e-4:
      P0 += 2.5e-6
    elif P0 < 1e-3:
      P0 += 2.5e-5
    elif P0 < 1e-2:
      P0 += 2.5e-4
    else:
      P0 += 2.5e-3

  return data_matrix


# In[14]:


eps_fluid = np.array([])
def eps_prime(x):
  return np.array([])


# In[15]:


def process_one_j(j, eos_name, eos_base, rho_matrix, cs_matrix, out_dir):
    """
    Generate and evaluate EOS candidates. Subsequently, save accepted TOV results. For the row j
    in the candidate parameters, the function creates an EOS table from eos_base. Also, rho_matrix 
    and cs_matrix are used to build the interpolations, solves the TOV equations for various density
    nodes and we determine whether this data is accepted or not. Accepted results are saved in out_dir.

    Parameters
        j: int
            Index deciding which row of our rho_matrix and cs_matrix are applied.
        eos_name: str
            Name of EOS used in file name.
        eos_base: ndarray
            baseline EOS data table
        rho_matrix: ndarray
            Matrix containing density values.
        cs_matrix: ndarray
            Matrix containing speed of sound values.
        out_dir: str
            Directory where we save result files in.

        Returns
        dict
            Dictionary containing amount of values tested, amount of accepted models and list of saved file paths.
    """
    global Lambda, eps_fluid, eps_prime

    z = j
    print(f"starting j = {j+1}")

    local_total = 0
    local_accepted = 0
    saved = []

    eos_matrix = np.empty((0, 3))
    eos_matrix = build(eos_matrix, eos_base, rho_matrix, cs_matrix, z, rho_t, rho_fin)

    p = eos_matrix[:, 0]
    eps = eos_matrix[:, 1]

    eps_fluid_base = interp1d(p, eps, kind='linear', fill_value="extrapolate")
    deps_dp = np.gradient(eps, p)
    eps_prime_base = interp1d(p, deps_dp, kind='linear', fill_value='extrapolate')

    eos_end_p = eos_matrix[-1, 0]

    for i in range(len(Lamb_arr)):
        Lambda = Lamb_arr[i]
        local_total += 1

        eps_fluid = eps_fluid_base
        eps_prime = eps_prime_base

        P0 = 2.5e-5
        if Lambda > 0 and eos_end_p > pc:
            Pf = eos_end_p - Lambda
        else:
            Pf = eos_end_p

        data_matrix = np.empty((0, 4))
        data_matrix = cycle_tov(data_matrix, P0, Pf)

        if data_matrix.size == 0:
            continue

        M_max = np.max(data_matrix[:, 1])

        if Lambda == 0:
            temp = 0
        else:
            temp = int(np.floor((abs(Lambda / (Kp * Pu)) ** 0.25) / np.sign(Lambda)))

        if 2.18 < M_max < 2.52:
            local_accepted += 1
            out_path = f"{out_dir}/TOV_{eos_name}_{temp}_{z+1}.csv"
            np.savetxt(out_path, data_matrix)
            saved.append(out_path)

    print(f"done j = {j+1}")
    return {"total": local_total, "accepted": local_accepted, "saved": saved}


# In[16]:


def generate_tovs(eos_name, eos_base, rho_matrix, cs_matrix, out_dir, n_jobs=6):
    """
    Generate TOV solutions for all EOS parameter data. Here, we paralellize over n_jobs amount of CPU cores, for a more efficient
    data generation. We run process_one_j for each row of rho_matrix. When data is accepted, these are saved in out_dir.
    We also print information like acceptance rate to keep track of the data generation.

    Parameters
        eos_name: str
            Name of EOS used in file names.
        eos_base: ndarray
            Baseline EOS data.
        rho_matrix: ndarray
            Matrix containing density values.
        cs_matrix: ndarray
            Matrix containing speed of sound values.
        out_dir: str
            Directory where we save result files in.
        n_jobs : int
            Number of CPU cores parallelized used to generate data (Check how much cores you have has before running!)
    """
    os.makedirs(out_dir, exist_ok=True)

    for f in os.listdir(out_dir):
        os.remove(os.path.join(out_dir, f))

    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
        delayed(process_one_j)(j, eos_name, eos_base, rho_matrix, cs_matrix, out_dir)
        for j in range(rho_matrix.shape[0])
    )

    total = sum(r["total"] for r in results)
    accepted = sum(r["accepted"] for r in results)

    print("eos tested:", total)
    print("eos accepted:", accepted)
    print("acceptance rate:", accepted / total if total > 0 else np.nan)


# Below is for (M,R) data only

# In[21]:


def data_generator(R_rand, M_rand, data_matrix, idx_max, sigma_M, sigma_R, TOT):
    """
    Generate radius and mass samples from our EOS curve. Here, we interpolate the mass-radius curve from our data. Masses are sampled and 
    radii are computed using the RM curve. Subsequently, Gaussian noise is added to those values with a standard deviation of sigma_M and sigma_R respectively.
    We normalize these values.

    Parameters
        R_rand: ndarray
            Array storing our radius values.
        M_rand: ndarray
            Array storing our mass values.
        data_matrix: ndarray
            Array containing EOS including mass and radius.
        max_i: int
            Index up to which we use data to interpolate.
        sigma_M: float
            Mass noise standard deviation.
        sigma_R: float
            Radius noise standard deviation
        TOT: int
            Number of samples generated.

    Returns
        R_rand, M_rand: ndarray, ndarray
            Arrays of samples generated.
    
    """
    M_arr = data_matrix[:, 1]
    R_arr = data_matrix[:, 2]

    M_unique, idx_unique = np.unique(M_arr[:idx_max+1], return_index=True)
    R_unique = R_arr[:idx_max+1][idx_unique]

    RM_curve = interp1d(M_unique, R_unique, kind='linear', fill_value='extrapolate')
    M_rand = uniform.rvs(loc=M_unique[0], scale=M_unique[-1] - M_unique[0], size=TOT)
    R_rand = RM_curve(M_rand)

    M_rand = norm.rvs(loc=M_rand, scale=sigma_M) / M_norm
    R_rand = norm.rvs(loc=R_rand, scale=sigma_R) / R_norm

    return R_rand, M_rand


# In[22]:


def generate_mr_dataset(source_dir, dataset_path, rho_matrix, cs_matrix, ns=300, TOT=30):
    """
    This function generates the mass-radius dataset from the data containing the solved TOV equations.
    Here, the function reads the TOV files and generates random mass-radius samples for all files. 
    Subsequently, these files are combined with the EOS parameters and then saved.

    Parameters
    source_dir: str
        Directory of saved TOV files.
    dataset_path: str
        Path of final dataset.
    rho_matrix: ndarray
        Matrix containing density values.
    cs_matrix: ndarray
        Matrix containing speed of sound values.
    ns: int
        Number of samples per TOV file.
    TOT: int
        Amount of mass-radius points per sample.

    Returns
        None
    """
    sigma_M = 0.1
    sigma_R = 0.5
    M_norm = 3
    R_norm = 20

    if os.path.exists(dataset_path):
        os.remove(dataset_path)

    listfile = sorted(os.listdir(source_dir))

    for file_idx, filename in enumerate(listfile, start=1):
        print(f"{file_idx}/{len(listfile)} : {filename}")

        parts = filename.split(".")[0].split("_")
        eos_name = parts[1]
        Lambda_temp = int(parts[2])
        z = int(parts[3])

        TOV_matrix = np.loadtxt(os.path.join(source_dir, filename))

        rows_to_write = []

        for i in range(ns):
            R_rand = np.array([])
            M_rand = np.array([])

            idx_max = np.argmax(TOV_matrix[:, 1])
            R_rand, M_rand = data_generator(
                R_rand, M_rand, TOV_matrix, idx_max, sigma_M, sigma_R, TOT
            )

            row = np.concatenate([
                np.array([eos_name, Lambda_temp], dtype=object),
                rho_matrix[z - 1, :],
                cs_matrix[z - 1, :],
                M_rand,
                R_rand
            ])

            rows_to_write.append(row)

        rows_to_write = np.array(rows_to_write, dtype=object)

        with open(dataset_path, "ab") as f:
            np.savetxt(f, rows_to_write, fmt="%s")


# In[24]:


def data_generator_k2(R_rand, M_rand, k2_rand, data_matrix, idx_max, TOT, sigma_M, sigma_R, sigma_k2):
    """
    Generate radius and mass samples from our EOS curve. Here, we interpolate the mass-radius curve from our data. Masses are sampled and 
    radii are computed using the RM curve. Subsequently, Gaussian noise is added to those values with a standard deviation of sigma_M and sigma_R respectively.
    We normalize these values.

    Parameters
        R_rand: ndarray
            Array storing our radius values.
        M_rand: ndarray
            Array storing our mass values.
        data_matrix: ndarray - [P0, M, R, lambda]
            Array containing EOS including mass and radius.
        max_i: int
            Index up to which we use data to interpolate.
        sigma_M: float
            Mass noise standard deviation.
        sigma_R: float
            Radius noise standard deviation.
        TOT: int
            Number of samples generated.
        sigma_M: float
            Mass noise standard deviation.
        sigma_R: float
            Radius noise standard deviation.
        sigma_k2: float
            k2 noise standard deviation.

    Returns
        R_rand, M_rand: ndarray, ndarray
            Arrays of samples generated.
    """
    M_arr = data_matrix[:, 1]
    R_arr = data_matrix[:, 2]
    lamb_arr = data_matrix[:, 3]

    #compute k2 from lambda
    k2_arr = (3.0 / 2.0) * lamb_arr * (M_arr ** 5) / (R_arr ** 5) / (KK ** 5)

    #keep stable branch up to maximum mass
    M_unique, idx_unique = np.unique(M_arr[:idx_max+1], return_index=True)
    R_unique = R_arr[:idx_max+1][idx_unique]
    k2_unique = k2_arr[:idx_max+1][idx_unique]

    RM_curve = interp1d(M_unique, R_unique, kind="linear", fill_value="extrapolate")
    k2M_curve = interp1d(M_unique, k2_unique, kind="linear", fill_value="extrapolate")

    #sample masses uniformly
    M_rand = uniform.rvs(loc=M_unique[0], scale=M_unique[-1] - M_unique[0], size=TOT)
    R_rand = RM_curve(M_rand)
    k2_rand = k2M_curve(M_rand)

    #Gaussian noise injection
    M_rand = norm.rvs(loc=M_rand, scale=sigma_M) / M_norm
    R_rand = norm.rvs(loc=R_rand, scale=sigma_R) / R_norm
    k2_rand = norm.rvs(loc=k2_rand, scale=sigma_k2)

    return R_rand, M_rand, k2_rand


# In[25]:


def generate_mrk2_dataset(source_dir, dataset_k2_path, rho_matrix, cs_matrix, ns=100, TOT=30):
    """
    This function generates the mass-radius-k2 dataset from the data containing the solved TOV equations.
    Here, the function reads the TOV files and generates random mass-radius-k2 samples for all files. 
    Subsequently, these files are combined with the EOS parameters and then saved.

    Parameters
    source_dir: str
        Directory of saved TOV files.
    dataset_k2_path: str
        Path of final dataset.
    rho_matrix: ndarray
        Matrix containing density values.
    cs_matrix: ndarray
        Matrix containing speed of sound values.
    ns: int
        Number of samples per TOV file.
    TOT: int
        Amount of mass-radius-k2 points per sample.

    Returns
        None
    """
    sigma_k2 = 0.05
    sigma_M = 0.1
    sigma_R = 0.5
    M_norm = 3
    R_norm = 20

    if os.path.exists(dataset_k2_path):
        os.remove(dataset_k2_path)

    listfile = sorted(os.listdir(source_dir))

    for file_idx, filename in enumerate(listfile, start=1):
        print(f"{file_idx}/{len(listfile)} : {filename}")

        parts = filename.split(".")[0].split("_")
        eos_name = parts[1]
        Lambda_temp = int(parts[2])
        z = int(parts[3])

        TOV_matrix = np.loadtxt(os.path.join(source_dir, filename))

        rows_to_write = []

        idx_max = np.argmax(TOV_matrix[:, 1])

        for i in range(ns):
            R_rand = np.array([])
            M_rand = np.array([])
            k2_rand = np.array([])

            R_rand, M_rand, k2_rand = data_generator_k2(
                R_rand, M_rand, k2_rand,
                TOV_matrix, idx_max, TOT,
                sigma_M, sigma_R, sigma_k2
            )

            row = np.concatenate([
                np.array([eos_name, Lambda_temp], dtype=object),
                rho_matrix[z - 1, :],
                cs_matrix[z-1, :],
                M_rand,
                R_rand,
                k2_rand
            ])

            rows_to_write.append(row)

        rows_to_write = np.array(rows_to_write, dtype=object)

        with open(dataset_k2_path, "ab") as f:
            np.savetxt(f, rows_to_write, fmt="%s")

