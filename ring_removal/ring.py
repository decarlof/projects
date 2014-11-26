#!/usr/bin/env python

"""ring.py: Code for ring suppresion."""

__author__     = "Eduardo X Miqueles"
__copyright__  = "Copyright 2014, CNPEM/LNLS" 
__credits__    = "Juan V.Bermudez & Hugo H.Slepicka"
__maintainer__ = "Eduardo X Miqueles"
__email__      = "edu.miqueles@gmail.com" 
__date__       = "14.Jan.2014" 

import numpy

##########################################################
def kernel(m,n):
    v = [
         [numpy.array([1,-1]), numpy.array([-3/2,2,-1/2]), numpy.array([-11/6,3,-3/2,1/3])],
         [numpy.array([-1,2,-1]), numpy.array([2,-5,4,-1])],
         [numpy.array([-1,3,-3,1])]
        ]
   
    return v[m-1][n-1]  

##########################################################
def ringMatXvec(h,x):
    s = numpy.convolve(x,numpy.flipud(h))

    u = s[numpy.size(h)-1:numpy.size(x)]

    y = numpy.convolve(u,h)
    
    return y

##########################################################
def ringCGM(h,alpha,f):

	x0 = numpy.zeros(numpy.size(f))

	r = f - (ringMatXvec(h,x0) + alpha*x0)

	w = -r

	z = ringMatXvec(h,w) + alpha*w

	a = numpy.dot(r,w)/numpy.dot(w,z) 

	x = x0 + numpy.dot(a,w)

	B = 0
		  
	for i in range(1000000):
		r = r - numpy.dot(a,z);
		if( numpy.linalg.norm(r) < 0.0000001 ):
			break
		B = numpy.dot(r,z)/numpy.dot(w,z) 
		w = -r + numpy.dot (B,w) 
		z = ringMatXvec(h,w) + alpha*w; 
		a = numpy.dot(r,w)/numpy.dot(w,z);
		x = x + numpy.dot(a,w);
                
	return x

##########################################################
def ring(data,m,n): 

	mydata = numpy.transpose(data)

	R = numpy.size(mydata,0)
	N = numpy.size(mydata,1)
    	
	#### Removing NaN !!!!!!!!   :-D

	pos = numpy.where( numpy.isnan(mydata) == True )
	
	mydata[pos] = 0

	#### Parameter
        
	alpha =  1 / (2*(mydata.sum(0).max() - mydata.sum(0).min()))

        #print alpha
	
	#### mathematical correction

	pp = mydata.mean(1)
		
	h = kernel(m,n)             

	#########

	f = -ringMatXvec(h,pp)

        q = ringCGM(h,alpha,f);

	### update sinogram

	q.shape = (R,1)
	K = numpy.kron(q,numpy.ones((1,N)))
	new = numpy.add(mydata,K)

	newsino = new.astype(numpy.float32)

	return numpy.transpose(newsino)
 

##########################################################

def ringb(data,m,n,step): 

	mydata = numpy.transpose(data)
        
	R = numpy.size(mydata,0)
	N = numpy.size(mydata,1)
    	
	#### Removing NaN !!!!!!!!  :-D

	pos = numpy.where( numpy.isnan(mydata) == True )
	
	mydata[pos] = 0

	### Kernel & regularisation parameter

	h = kernel(m,n)

        #alpha = 1 / (2*(mydata.sum(0).max() - mydata.sum(0).min()))

        #### mathematical correction by blocks

	nblocks = N/step

	new = numpy.ones((R,N))

	for k in range (0,nblocks):

		sino_block = mydata[:, k*step:(k+1)*step]
                
                alpha = 1 / (2*(sino_block.sum(0).max() - sino_block.sum(0).min()))
	
		pp = sino_block.mean(1)
		
		##

		f = -ringMatXvec(h,pp)

		q = ringCGM(h,alpha,f)

	        ### update sinogram

		q.shape = (R,1)

		K = numpy.kron(q,numpy.ones((1,step)))

		new[:,k*step:(k+1)*step] = numpy.add(sino_block,K)

		
	newsino = new.astype(numpy.float32)

	return numpy.transpose(newsino)
 
##########################################################
