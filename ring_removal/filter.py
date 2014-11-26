#!/usr/bin/env python

"""filtros.py: Code for ring filtering"""

__author__     = "Eduardo X Miqueles"
__copyright__  = "Copyright 2014, CNPEM/LNLS" 
__credits__    = "Juan V.Bermudez & Hugo H.Slepicka"
__maintainer__ = "Eduardo X Miqueles"
__email__      = "edu.miqueles@gmail.com" 
__date__       = "14.Jan.2014" 

import ring
import numpy


def filter_block(SINO, nblocks):

	#half = int(SINO.shape[0]/2)
	
	size = int(SINO.shape[0]/nblocks)

	d1 = ring.ringb(SINO,1,1,size)

	d2 = ring.ringb(SINO,2,1,size)

	p = d1*d2

	alpha = 1.5
	
	d = numpy.sqrt(p + alpha*numpy.fabs(p.min()))
	
	return d


def filter(SINO):

	d1 = ring.ring(SINO,1,1)
	
	d2 = ring.ring(SINO,2,1)
	
	p = d1*d2

	alpha = 1.5

	d = numpy.sqrt(p + alpha*numpy.abs(p.min()))

	return d
