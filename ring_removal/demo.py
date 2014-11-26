import ring_removal
import imagesc
import numpy

s = numpy.loadtxt('sino.dat')

n1 = ring_removal.ring_removal(s);

imagesc.imagesc(s,n1,n1-s)

