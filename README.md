# PyTorchLinearSVM

## Implementation
Implements a simple Linear SVM (soft margin formulation) trained in PyTorch by gradient descent on a 2D toy dataset with binary labels. The hyper-parameter search (C parameter) is done by cross-validation via scikit-learn. It includes a visualization of the decision function and associated support vectors throughout the epochs of training for various C values.
Inspiration from:
- Python Engineer: https://youtu.be/UX0f9BNBcsY
- A Developer Diary: http://www.adeveloperdiary.com/data-science/machine-learning/support-vector-machines-for-beginners-linear-svm/
	
## Results
<img width="640" alt="hyperplane_C_0p01" src="https://user-images.githubusercontent.com/17934718/117080196-3408ff00-acf2-11eb-8f93-c84cd46486a2.png">
<img width="640" alt="hyperplane_C_100" src="https://user-images.githubusercontent.com/17934718/117080226-42efb180-acf2-11eb-9527-957dff130b5b.png">
<img width="400" alt="results" src="https://user-images.githubusercontent.com/17934718/117080229-45520b80-acf2-11eb-8efd-5bd22c7c2ee5.png">
