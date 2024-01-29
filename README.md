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
e^z^ = e^1^ ,e^2^ , e^3^ = {2.72,7.39,20.01}
sum of e^z^ = {{2.72 + 7.39 +20.01}}=30.19
sigma = 2.72/30.19   ,   7.39/30.19     , 20.01/30.19  =  {.09 , .24 , .67}

SUM OF SIGMA WILL ALWAYS BE 1
```
![image](https://github.com/nsaqui4c/deepLearning-pyTorch/assets/45531263/caf9a96c-b79d-4b9b-b616-8624f5d09fe9)



