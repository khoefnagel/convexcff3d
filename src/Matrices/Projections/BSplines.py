"""Module to obtain B-spline basis functions and their polynomial coefficients."""

import numpy as np
import sympy as sym


x = sym.symbols('x')

#The BSplines1D dictionary defines B-spline basis functions for various degrees in
#one dimension. Each entry corresponds to a specific degree and contains a list of
#polynomial expressions for the basis functions of that degree. These polynomials
#are defined in terms of the variable x and are valid over unit intervals that span
#from 0 to degree + 1.
#source: https://en.wikipedia.org/wiki/Irwin%E2%80%93Hall_distribution#Special_cases
BSplines1D = {
    
    #Degree 0 B-spline basis functions, defined on the interval [0, 1].
    0 : [sym.Poly(1, x)],
    
    #Degree 1 B-spline basis functions, defined on the interval [0, 2].
    1 : [x, 2 - x],

    #Degree 2 B-spline basis functions, defined on the interval [0, 3].
    2 : [0.5*x**2, 0.5*(-2*x**2+6*x-3), 0.5*(3-x)**2],

    #Degree 3 B-spline basis functions, defined on the interval [0, 4].
    3 : [1/6*x**3, 1/6*(-3*x**3 + 12*x**2 - 12*x + 4),
         1/6*(3*x**3 - 24*x**2 + 60*x - 44), 1/6*(4-x)**3],

    #Degree 4 B-spline basis functions, defined on the interval [0, 5].
    4 : [1/24*x**4, 1/24*(-4*x**4 + 20*x**3 - 30*x**2 + 20*x - 5),
         1/24*(6*x**4 - 60*x**3 + 210*x**2 - 300*x + 155),
         1/24*(-4*x**4 + 60*x**3 - 330*x**2 + 780*x - 655),
         1/24*(5-x)**4]
}


#The CCBSplines1D dictionary is structured the same as the BSplines1D dictionary,
#but each spline piece is shifted to the interval [0, 1].
CCBSplines1DExpr = {}

#The CCBSplines1DPoly dictionary is similar to the CCBSplines1D dictionary but
#instead of holding lists of polynomial expressions, it holds a 2D array for each
#key. The first axis goes over the basis functions, the second over polynomial
#coefficients, these are ordered from low (x**0) to high (x**deg).
CCBSplines1D = {}
for deg, splineList in BSplines1D.items():
    CCBSplines1DExpr[deg] = []
    polyList = []
    for i, spline in enumerate(splineList):
        CCBSplines1DExpr[deg].append(spline.subs(x, x + i))
        
        polyCoeffs = np.array(CCBSplines1DExpr[deg][-1].as_poly().all_coeffs(),
                              dtype=float)[::-1]
        polyList.append([polyCoeffs])

    CCBSplines1D[deg] = np.concatenate(polyList)
                    

#The CCBSplines2D dictionary is similarly structured as CCBSplines1D, but for 2D
#splines. Each key corresponds to a (degx, degy) tuple and the value is a 4D array
#where the first two axes correspond to polynomial index in x and y. The last two
#axes correspond to polynomial coefficient in x and y.
CCBSplines2D = {}
for degx, polyCoeffsx in CCBSplines1D.items():    
    for degy, polyCoeffsy in CCBSplines1D.items():
        CCBSplines2D[(degx, degy)] = np.einsum('ij,kl->ikjl', polyCoeffsx, polyCoeffsy)


CCBSplines3D = {}
for degz, polyCoeffsz in CCBSplines1D.items():    
    for degy, polyCoeffsy in CCBSplines1D.items():    
        for degx, polyCoeffsx in CCBSplines1D.items():
            CCBSplines3D[(degx, degy, degz)] = np.einsum('ij,kl,mn->ikmljn',
                                                        polyCoeffsx,
                                                        polyCoeffsy,
                                                        polyCoeffsz)


if __name__ == "__main__":
    #Tests

    #Test 1: CCBSplines1DPoly

    #Test 1.1: order 0
    if np.linalg.norm(CCBSplines1D[0][0,0] - 1) < 1e-10:
        print('CCBSplines1DPoly: order 0 test passed')
    else:
        raise Exception('CCBSplines1DPoly: order 0 test failed')

    #Test 1.2: order 1
    if np.linalg.norm(CCBSplines1D[1] - np.array([[0, 1], [1, -1]])) < 1e-10:
        print('CCBSplines1DPoly: order 1 test passed')
    else:
        raise Exception('CCBSplines1DPoly: order 1 test failed')

    #Test 1.3: order 2
    if np.linalg.norm(CCBSplines1D[2] - np.array([[1/8, 1/2, 1/2],
                                                      [3/4, 0, -1],
                                                      [1/8, -1/2, 1/2]])) < 1e-10:
        print('CCBSplines1DPoly: order 2 test passed')
    else:
        raise Exception('CCBSplines1DPoly: order 2 test failed')

    #Test 2: CCBSplines2D

    #Test 2.1: order (0, 0)
    if np.linalg.norm(CCBSplines2D[(0, 0)] - 1) < 1e-10:
        print('CCBSplines2D: order (0, 0) test passed')
    else:
        raise Exception('CCBSplines2D: order (0, 0) test failed')

    #Test 2.2: order (0, 1)
    if np.linalg.norm(CCBSplines2D[(0, 1)][0,0] - np.array([0.5, 1])) < 1e-10 and\
       np.linalg.norm(CCBSplines2D[(0, 1)][0,1] - np.array([0.5, -1])) < 1e-10:
        print('CCBSplines2D: order (0, 1) test passed')
    else:
        print('CCBSplines2D: order (0, 1) test failed')

    #Test 2.3: order (1, 0)
    if np.linalg.norm(CCBSplines2D[(1, 0)][0,0] - np.array([[0.5], [1]])) < 1e-10 and\
       np.linalg.norm(CCBSplines2D[(1, 0)][1,0] - np.array([[0.5], [-1]])) < 1e-10:
        print('CCBSplines2D: order (1, 0) test passed')
    else:
        print('CCBSplines2D: order (1, 0) test failed')

    #Test 2.4: order (1, 1)
    if np.linalg.norm(CCBSplines2D[(1, 1)][0,0] - np.array([[0.25, 0.5],
                                                            [0.5, 1]])) < 1e-10 and\
       np.linalg.norm(CCBSplines2D[(1, 1)][0,1] - np.array([[0.25, -0.5],
                                                            [0.5, -1]])) < 1e-10 and\
       np.linalg.norm(CCBSplines2D[(1, 1)][1,0] - np.array([[0.25, 0.5],
                                                            [-0.5, -1]])) < 1e-10 and\
       np.linalg.norm(CCBSplines2D[(1, 1)][1, 1] - np.array([[0.25, -0.5],
                                                             [-0.5, 1]])) < 1e-10:
        print('CCBSplines2D: order (1, 1) test passed')
    else:
        raise Exception('CCBSplines2D: order (1, 1) test failed')

    #Plot 1D splines if the program is run directly
    import matplotlib.pyplot as plt
    for deg, splineArr in CCBSplines1D.items():
        plt.figure(num=deg)

        X = np.linspace(-0.5, 0.5, 101)
        for spline in splineArr:
            plt.plot(X, np.sum([spline[i]*X**i for i in range(deg+1)], axis=0))

        plt.title(f'deg = {deg}')
    plt.show()

    for i, deg in enumerate([(1,1), (2,2), (3,3)]):

        X = np.linspace(-0.5, 0.5, 21)
        Y = np.linspace(-0.5, 0.5, 21)
        XX, YY = np.meshgrid(X, Y)

        fig, axs = plt.subplots(num=i, nrows=deg[0]+1, ncols=deg[1]+1, figsize=(8,8))

        splineArr = CCBSplines2D[deg]
        for i in range(deg[0]+1):
            for j in range(deg[1]+1):
                f = np.zeros(XX.shape)
                for k in range(deg[0]+1):
                    for l in range(deg[1]+1):
                        f += splineArr[i,j,k,l]*XX**k*YY**l

                axs[deg[1]-j,i].imshow(f, extent=(-0.5, 0.5, -0.5, 0.5), vmin=0, vmax=1, origin='lower')
                axs[deg[1]-j,i].set_title(f'i={i}, j={j}')

        fig.suptitle(f'deg = {deg}')
    plt.show()
