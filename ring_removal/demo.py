import ring_removal
import imagesc
import numpy

s = numpy.loadtxt('sino.dat')

n1 = ring_removal.ring_removal(s);
n2 = ring_removal.ring_removal(s,2);
n3 = ring_removal.ring_removal(s,5);

imagesc.imagesc(s,n1,n1-s)

imagesc.imagesc(s,n1,n2,n3)
imagesc.imagesc(n1-s,n2-s,n3-s)

