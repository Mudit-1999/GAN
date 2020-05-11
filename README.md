# GAN 
The GAN algorithm involves the interplay of two adversaries namely generator and discriminator. In this the generator tries to generate points such that they closely resemble to the point in the original distribution. While a discriminator tries to judge the truthfull ness of the points(if they are generated from original distribution or not).  We train the discriminator  just like  the way we would train any network using both true and false (generated) samples to learn. But for generator we feed the sample generated, through the disciminator to spot the fake and backpropagate the error through the discriminator and the generator.
## Function f(x) to generate true samples:
\begin{equation}
    f(x) = 0.4\mathcal{N}(0,4)+0.3\mathcal{N}(-6,4)+0.3\mathcal{N}(6,4)
\end{equation}
where $\mathcal{N}(\mu,\sigma^2)$ is the Gaussian distrubution with mean $\mu$ and variance $\sigma^2$.

## Generator Architecture
- I have designed the Generatore to model the above function $f(x)$. Given random samples as input, it is expected to output samples very close to $f(x)$,.
- It contained three fully connected network with the activation of Relu.

## Discriminator Architecture
- The Discriminator is designed in such a way to check the validity of samples,ie, the samples are generated from a true source ( $f(x)$ disrtibution) or from some other source (generator).
-  It contained three fully connected network with the activation of Tanh and relu. 

## Loss

- Loss for dicriminator is given by:
\begin{equation}
min_{G}max_D V(D,G) = -0.5* \mathbb{E}_{x \sim f(x)}\big[logD(x) + log(1-D(G(x)))\big]
\end{equation}
- Loss for generator is given by:
\begin{equation}
min_{G}max_D V(D,G) = -\mathbb{E}_{x \sim f(x)}\big[logD(G(x))\big]
\end{equation}
where $D(x)$ and $G(x)$ are discriminator and generator networks respectively.

## Training our GAN Network.
- Training and choosing the right hyperparameter for GANs is the most difficuilt part as the loss goes on oscillating. 
- We expect the loss of Generator and Discriminator will find a mid way.