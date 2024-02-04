# deepLearning-pyTorch

## ANN - Artificial Neural Network - concepts in deep learning

* A way to transform input into output
  * ![image](https://github.com/nsaqui4c/deepLearning-pyTorch/assets/45531263/e3397885-aa61-4961-b4ea-9e6956261459)

* it does so by analyzing all the input variable and assigning different weightage to each variable and then create a rule to get specific output
  * ![image](https://github.com/nsaqui4c/deepLearning-pyTorch/assets/45531263/866d4944-2ab3-4631-9fbe-7410d208886f)
  * ![image](https://github.com/nsaqui4c/deepLearning-pyTorch/assets/45531263/b254bf49-612d-420c-8eee-425887952c0f)

* the above equation is for linear equation for non linear solution we multiply it by sigma(a non linear func).
  * ![image](https://github.com/nsaqui4c/deepLearning-pyTorch/assets/45531263/3708638b-82fb-4227-b93b-39b5a212686c)

  * ![image](https://github.com/nsaqui4c/deepLearning-pyTorch/assets/45531263/f72579a5-2360-4941-8158-6815787492ea)

* using this equatin as building block and then creating neural network we get our output.
  * ![image](https://github.com/nsaqui4c/deepLearning-pyTorch/assets/45531263/591f605d-fe7d-49ad-9711-fa88fe0c11ed)

* There are multiple ANN architecture:
  *  ![image](https://github.com/nsaqui4c/deepLearning-pyTorch/assets/45531263/60c37ed6-3d38-47ed-a140-32903c4ba42e)
 
* Forward and back propagation
  * ![image](https://github.com/nsaqui4c/deepLearning-pyTorch/assets/45531263/f5b3ea80-286c-4569-abf4-ea4378454ef8)


 
### Philosphy of deep Learning
![image](https://github.com/nsaqui4c/deepLearning-pyTorch/assets/45531263/63b0e896-13fb-466f-80f9-9fa779edcb85)



## Dot product
* A single number represents the comanalites between two objects (vector, matrices, tensor, sigal, images)
![image](https://github.com/nsaqui4c/deepLearning-pyTorch/assets/45531263/55548268-33e7-41e9-8289-d5af9945dcc9)
![image](https://github.com/nsaqui4c/deepLearning-pyTorch/assets/45531263/91a601f3-6b2e-4829-86ed-6e11ec6783a2)

## Matrix multiplication
![image](https://github.com/nsaqui4c/deepLearning-pyTorch/assets/45531263/f20a4468-e445-413e-9f81-42d3e039bd28)
![image](https://github.com/nsaqui4c/deepLearning-pyTorch/assets/45531263/6d662023-42da-4148-8833-c12ea36a611b)


## Softmax
* Softmax function has property that after transformation all the values are between 0 and 1 and there sum is equal to 1
```
z={1,2,3}
e^z = e^1 ,e^2 , e^3 = {2.72,7.39,20.01}
sum of e^z = {{2.72 + 7.39 +20.01}}=30.19
sigma = 2.72/30.19   ,   7.39/30.19     , 20.01/30.19  =  {.09 , .24 , .67}

SUM OF SIGMA WILL ALWAYS BE 1
```
![image](https://github.com/nsaqui4c/deepLearning-pyTorch/assets/45531263/caf9a96c-b79d-4b9b-b616-8624f5d09fe9)
```
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# USING numpy
z= np.random.randint(-5,high=15, size=35)
print(z)
num = np.exp(z)
sum = np.sum(num)
sigmas = num/sum


#using Torch
## instance of softmax class
softfun = nn.Softmax(dim=0)
## Converting z to tensor and finding softmax
sigmaT = softfun( torch.Tensor(z) )

plt.plot(z,sigmaT,'ko')
##plt.plot(z,sigmas,'ko')
plt.ylabel('proablility value')
plt.xlabel('random Z value')
plt.title('$\sum\sigma$ = %g' %np.sum(sigmas))
plt.show()
```
## Logarithm
* Innverse of natural exponential (e)
* Log is monotomoc in nature -> log(x) increases when x increases and decreases with x also
* sin(x) in non monotomic
* profit of using log is that:
  * We are dealing with very small number in ML (probability)
  * as x increases value of log(x) increases exponentially for small number, then the gap decreases for large number
  * computer have issue while dealing with small number
  * converting to log(x) make the calculation easy
![image](https://github.com/nsaqui4c/deepLearning-pyTorch/assets/45531263/18a6d63a-1fb3-498c-89e9-1b6deaf31754)

![image](https://github.com/nsaqui4c/deepLearning-pyTorch/assets/45531263/1139d496-91fb-4422-bc87-8029b2f94b57)

## Entropy and cross-entropy

* In ML entropy is max when probability is 0.5  -> anything can happen
* entropy is lease when probability is either 1 or 0
![image](https://github.com/nsaqui4c/deepLearning-pyTorch/assets/45531263/60a448bd-afc6-4b71-996e-8009ffd56c9c)

* Formula for entropy -> summation of ( probability of all event * log<sub>2</sub> of  probability of all event )
* for coin toss n=2 -> summation of probability of head and tails
* bits -> if we use base 2
* nats -> if we use natural log
![image](https://github.com/nsaqui4c/deepLearning-pyTorch/assets/45531263/9fc5aa89-fc4c-4533-8b22-4110fa99f5eb)
![image](https://github.com/nsaqui4c/deepLearning-pyTorch/assets/45531263/1fde8804-f466-4b08-b49a-8e1d52c1a529)
* Cross entropy -> proablity of a picture being a cat given that it is a cat  -> p=1, q=proability of it being a cat by machine
![image](https://github.com/nsaqui4c/deepLearning-pyTorch/assets/45531263/f8515d1a-8cea-49d6-a949-4bf26b19af2b)

#### Entropy
![image](https://github.com/nsaqui4c/deepLearning-pyTorch/assets/45531263/ad5db056-0e30-44b3-9bc8-488d75def7dd)

#### Cross Entropy
![image](https://github.com/nsaqui4c/deepLearning-pyTorch/assets/45531263/9cae8fc7-234b-4b3b-9fd2-bd5c86a926b1)




