# Adversarial-Training

In this assignment, we will play around with adversarial techniques in deep learning.
The main goal is to build some understanding of *Generative Adversarial Networks* (GANs).
However, we will also take a look at *adversarial training*,
which is a collection of methods for fooling neural networks.
Although both approaches are completely unrelated,
it turns out that is possible to build GANs, starting from adversarial examples.


### Exercise 1: Projected Gradient Ascent 

One of the easiest ways to construct an adversarial example
is to maximise the loss by using gradient ascent on the inputs.
After all, the parameters of a neural network are also the result
of a gradient descent that minises the loss function.

In order to keep the perturbations to the original image small,
a constraint is added to the gradient ascent for adversarial examples:
the adversarial example must lie in an epsilon-ball around the original input.
This procedure is also known as *projected gradient ascent*.

------------------------------------------------------------------------------------------------------------------------------------
## Fooling a Dummy Network

Adversarial examples are considered an analysis tool
that can help to gain insights in how neural networks behave.
However, when pushing these methods, it turns out that
adversarial examples can actually also be used to generate new images.
The key point to getting useful images, is to attack the right model.

Consider the task of predicting whether an input is *real* or *fake*.
A *real* sample would be defined as an input that comes from the dataset,
whereas a *fake* example is an input that does not appear in the dataset.
It should be relatively easy to build a binary classification model for this task.
Let us denote this model as a *discriminator*,
since it discriminates between *real* and *fake* images.

By using an adversarial attack on this model
*fake* inputs can be altered to look more like *real* samples.

### Exercise 2: Discriminator Data 

Training a model that is able to discriminate between *real* and *fake*,
is not entirely straightforward.
The main problem is to obtain a reasonable dataset
that represents both classes equally well.

One solution is to use random noise for the *fake* inputs.
However, this often makes things too easy
and will not produce a particularly strong model.
To counter this issue, we can use an adversarial attack
to make the *fake* inputs appear more real
after every epoch of discriminator training.
This turns out to produce a reasonable discriminator
as well as reasonably good *fake* inputs.

------------------------------------------------------------------------------------------------------------------------------------
### Exercise 3: Training the Discriminator (1 point)

Finally, we can use this dataset for training the discriminator.
By updating the *fake* inputs after every epoch of training,
the quality of the *fake* inputs, and therefore also the discriminator,
should improve steadily during training.

Time to test if this actually works.
The `train_discriminator` function (in the preamble)
implements a training loop that updates the fake examples every epoch.
The code below is able to train the discriminator to zero loss.
However, the *fake* images do not look very realistic.

 > Play around with some of the hyperparameters (lr, epsilon, ...)
 > to get more realistic *fake* images,
 > so that the task of the discriminator becomes harder.

------------------------------------------------------------------------------------------------------------------------------------
### Excercise 4: Min-Max Game 


The goal of the generator is to create images that maximise the loss
that the discriminator attempts to minimise.
The result is a min-max game between generator and discriminator.
Mathematically, this leads to the following optimisation target:
$$\begin{aligned}
\max_d \min_g \mathbb{E}[\log d(X)] + \mathbb{E}[\log(1 - d(g(Z)))].
\end{aligned}$$
Note that this function is simply the negated binary cross-entropy loss.

------------------------------------------------------------------------------------------------------------------------------------
<img width="578" alt="image" src="https://github.com/user-attachments/assets/740f4a83-8318-489b-814d-327b54b071da">


