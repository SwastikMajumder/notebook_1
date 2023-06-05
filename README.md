# notebook_1
Digit recognition neural network
This software can recongnize numbers written on a paper... the pipeline of the process is like this:
1) Remove background by guassian adpative thresholding to get a binary image
2) Analyzed pixels which are connected together following 8 direction of connectivity, to remove lone stops and locate position of digits
3) Thicken the digits (so after resize still some information is left)
4) Then pad it, resize it to 28x28 form
5) Then thin the digits again (thinning is a part of neural network, and it won't trouble losing information because its carefully built)
6) Use neural networks to identify what digits they are (neural network trained on MNIST numbers after thinning of it)
7) Draw bounding boxes and then return the resultant image
