# cython: language_level=3
import cython
cimport cython
from cython.parallel cimport prange, parallel
from libc.stdlib cimport abort, malloc, free
from libc.math cimport sqrt
cimport openmp
from libc.stdio cimport printf



@cython.boundscheck(False) # turn off boundscheck for this function
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef void calculation_of_TGI_and_GLI(double [::1] R, double [::1] G, double [::1] B, double [::1] TGI, double [::1] GLI, double max_const):

    cdef Py_ssize_t n_element, i

    n_element = R.size

    for i in prange(n_element, nogil=True, schedule='dynamic'):
        TGI[i] = (G[i] - (0.39 * R[i]) - (0.61 * B[i])) / max_const
        GLI[i]  = ((G[i] - R[i]) + (G[i] - B[i])) / (2.0 * G[i] + R[i] + B[i] + 1.0)





@cython.boundscheck(False) # turn off boundscheck for this function
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef void calculation_of_ExG_and_CIVE(double [::1] R, double [::1] G, double [::1] B, double [::1] ExG, double [::1] CIVE):

    cdef Py_ssize_t n_element, i

    n_element = R.size

    for i in prange(n_element, nogil=True, schedule='dynamic'):
        ExG[i] = 2.0 * G[i] - R[i] - B[i]
        CIVE[i] = 0.441 * R[i] - 0.811 * G[i] + 0.385 * B[i] + 18.78745




########################################################################################################################
@cython.boundscheck(False) # turn off boundscheck for this function
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef void calculation_of_WDRVI_and_CCCI_and_NDVI_and_CIrededge(double [::1] R, double [::1] NIR, double [::1] RedEdge,double alpha, double [::1] NDVI, double [::1] WDRVI, double [::1] CCCI, double [::1] CIrededge):

    cdef Py_ssize_t n_element, i

    n_element = R.size

    for i in prange(n_element, nogil=True, schedule='dynamic'):

        NDVI[i] = (NIR[i]-R[i])/(NIR[i]+R[i]+1.0)
        CCCI[i] =  ((NIR[i] - RedEdge[i])/(NIR[i]+RedEdge[i]+1.0))/(((NIR[i]-R[i])/(NIR[i]+R[i]+1.0))+1.0)
        WDRVI[i] = (alpha*NIR[i] - R[i])/(alpha*NIR[i]+R[i]+1.0)
        CIrededge[i] = (NIR[i]/(RedEdge[i]+1.0))-1.0

########################################################################################################################
@cython.boundscheck(False) # turn off boundscheck for this function
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef void calculation_of_NDVI(double [::1] R, double [::1] NIR, double [::1] NDVI):

    cdef Py_ssize_t n_element, i

    n_element = R.size

    for i in prange(n_element, nogil=True, schedule='dynamic'):

        NDVI[i] = (NIR[i]-R[i])/(NIR[i]+R[i]+1.0)


########################################################################################################################
@cython.boundscheck(False) # turn off boundscheck for this function
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef void calculation_of_MGRV_and_MPRI_and_RGBVI(double [::1] R, double [::1] G, double [::1] B, double [::1] MGRV, double [::1] MPRI, double [::1] RGBVI):

    cdef Py_ssize_t n_element, i

    n_element = R.size

    for i in prange(n_element, nogil=True, schedule='dynamic'):

        MGRV[i] = ((G[i]*G[i])-(R[i]*R[i]))/((G[i]*G[i])+(R[i]*R[i])+1.0)
        MPRI[i] =  (G[i] - R[i])/(G[i] + R[i]+1.0)
        RGBVI[i] = (G[i] - (B[i]*R[i]))/((G[i]*G[i]) + (B[i]*R[i]) + 1.0)


########################################################################################################################
@cython.boundscheck(False) # turn off boundscheck for this function
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef void calculation_of_veg_indices3(double [::1] R, double [::1] G, double [::1] B, double [::1] NIR, double [::1] RE, double [::1] GNDVI, double [::1] NDRE, double [::1] RDVI):


    cdef Py_ssize_t n_element, i

    n_element = R.size

    for i in prange(n_element, nogil=True, schedule='dynamic'):
        #printf("Evi %f\n")
        # EVI[i] = 2.5*(NIR[i]-R[i])/(((NIR[i]+6*R[i]-7.5*B[i])+1.0)+1.0)
        #printf("GNDVI %f \n")
        GNDVI[i] =  (NIR[i]-G[i])/(NIR[i]+G[i]+1.0)
        #printf("NDRE %f\n")
        NDRE[i] =  (NIR[i]-RE[i])/(NIR[i]+RE[i]+1.0)
        #printf("RDVI%f\n")
        RDVI[i] = (NIR[i]-R[i])/(sqrt(NIR[i]+R[i])+1.0)

########################################################################################################################
@cython.boundscheck(False) # turn off boundscheck for this function
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef void calculation_of_veg_indices4(double [::1] R, double [::1] G, double [::1] NIR, double [::1] OSAVI, double [::1] MSR, double [::1] MCARI1, double [::1] MCARI2):


    cdef Py_ssize_t n_element, i

    n_element = R.size

    for i in prange(n_element, nogil=True, schedule='dynamic'):

        OSAVI[i] = 1.6*((NIR[i]-R[i])/(NIR[i]+R[i]+0.16))
        MSR[i] =  ((NIR[i]/(R[i]+1.0))-1.0)/(sqrt((NIR[i]/(R[i]+1.0))+1.0)+1.0)
        MCARI1[i] = 1.2*(2.5*(NIR[i]-G[i])-1.3*(R[i]-G[i]))
        # mcari2_num = 1.5*((2.5*(NIR[i]-R[i]))-(1.3*(NIR[i]-G[i])))
        # mcari2_den = sqrt((2*NIR[i]+1.0)*(2*NIR[i]+1.0) + (6*NIR[i]-5*sqrt(R[i]))-0.5)
        MCARI2[i] = (1.5*((2.5*(NIR[i]-R[i]))-(1.3*(NIR[i]-G[i]))))/(sqrt((2*NIR[i]+1.0)*(2*NIR[i]+1.0) + (6*NIR[i]-5*sqrt(R[i]))-0.5)+1.0)

########################################################################################################################
@cython.boundscheck(False) # turn off boundscheck for this function
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef void calculation_of_veg_indices5(double [::1] R, double [::1] G, double [::1] NIR, double [::1] MTVI1, double [::1] MTVI2, double [::1] PSSRA, double [::1] EVI2):


    cdef Py_ssize_t n_element, i

    n_element = R.size

    for i in prange(n_element, nogil=True, schedule='dynamic'):


        MTVI1[i] = 1.2*(1.2*(NIR[i]-G[i])-2.5*(R[i]-G[i]))
        # mtvi2_num = 1.5*(1.2*(NIR[i]-G[i])-2.5*(R[i]-G[i]))
        # mtvi2_den = sqrt((2*NIR[i]+1.0)*(2*NIR[i]+1.0) + (6*NIR[i]-5*sqrt(R[i]))-0.5)
        MTVI2[i] = (1.5*(1.2*(NIR[i]-G[i])-2.5*(R[i]-G[i])))/(sqrt((2*NIR[i]+1.0)*(2*NIR[i]+1.0) + (6*NIR[i]-5*sqrt(R[i]))-0.5)+1.0)
        PSSRA[i] = NIR[i]/(R[i]+1.0)
        EVI2[i] =  2.5*(NIR[i] - R[i])/(1.0+NIR[i]+(2.5*R[i]))


########################################################################################################################
@cython.boundscheck(False) # turn off boundscheck for this function
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef void calculation_of_veg_indices6(double [::1] R, double [::1] G, double [::1] B, double [::1] NIR, double [::1] CIgreen, double [::1] NDWI, double [::1] RGBVI_CORRECT):


    cdef Py_ssize_t n_element, i

    n_element = R.size

    for i in prange(n_element, nogil=True, schedule='dynamic'):


        CIgreen[i] = (NIR[i]/(G[i]+1.0))-1.0
        NDWI[i] = (G[i] - NIR[i])/(G[i] + NIR[i]+1.0)
        RGBVI_CORRECT[i] = (G[i]*G[i] -  (B[i]*R[i]))/(G[i]*G[i] +  (B[i]*R[i]) + 1.0)



########################################################################################################################
@cython.boundscheck(False) # turn off boundscheck for this function
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef void prob_map_calc(double [:,::1] prob_map, double [:,::1] integral_image,int step):

    cdef int rows, cols, r,c, i, j, n

    r = prob_map.shape[0]
    c = prob_map.shape[1]

    for i in prange(r, nogil=True, schedule='dynamic'):
        for j in range(c):
            rows =  i + step
            cols =  j + step
            for n in range(step):
                prob_map[i,j] += <double>(integral_image[rows + n + 1,cols + n + 1] + integral_image[rows - n - 1, cols - n - 1] - integral_image[rows - n - 1, cols + n + 1] - integral_image[rows + n + 1, cols - n - 1])
