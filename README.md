[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/51DI_w4M)
# 1. Theoretical exercise: Local linear regression as a linear smoother

Recall our basic setup: we are given i.i.d. samples $\left(x_i, y_i\right), i=1, \ldots n$ from the model

$$y_i=m\left(x_i\right)+\epsilon_i, \quad i=1, \ldots n$$

and our goal is to estimate $m$ with some function $\hat{m}$. Assume that each $x_i \in \mathbb{R}$ (i.e., the predictors are 1-dimensional).

The local linear regression estimator at a point $x$ is defined by

$$(\hat\beta_0(x), \hat\beta_1(x)) 
= \arg\min_{\beta_0,\beta_1 \in \mathbb{R}} 
\sum_{i=1}^n \Big(Y_i - \beta_0 - \beta_1 (X_i - x)\Big)^2 
K\left(\frac{X_i - x}{h}\right),$$

where $K$ is a kernel function and $h>0$ is a bandwidth. The fitted value is then given by $\hat m(x) = \hat\beta_0(x)$ and we will show that such estimator belongs to the class of linear smoothers, so that

$$\hat{m}(x)=\sum_{i=1}^n w\left(x, x_i\right) \cdot y_i$$

for some choice of weights $w\left(x, x_i\right)$. 

1. Show that $\hat m(x)$ can be expressed as a weighted average of the observations:
   
   $$\hat m(x) = \sum_{i=1}^n w_{ni}(x) Y_i,$$
   
   where the weights $w_{ni}(x)$ depend only on $x$, $\{X_i\}$, $K$, and $h$, but not on the $Y_i$’s.
   
3. Using the notation
   
   $$S_{n,k}(x) = \frac{1}{nh}\sum_{i=1}^n (X_i - x)^k K\left(\frac{X_i - x}{h}\right), \quad k=0,1,2,$$
   
   derive an explicit expression for $w_{ni}(x)$ in terms of $S_{n,0}(x), S_{n,1}(x), S_{n,2}(x)$, and the kernel.  

5. Prove that the weights satisfy $\sum_{i=1}^n w_{ni}(x) = 1$.  

# 2. Practical exercise: Global bandwidth selection

Assume that we have a sample $\{(X_i,Y_i)\}_{i=1}^n$ of i.i.d. random vectors and that we are interested in estimating the conditional expectation $m(x) = {\rm{E}}(Y \mid X=x)$. We consider here the local linear estimator $\hat{m}$, as defined in Slide 14 of [Lecture 3](https://math-516-517-main.github.io/math_517_website/lectures/04_Smoothing.pdf).

Throughout the assignment, we will assume homoscedasticity, i.e., the local variance $\sigma^2(x) = {\rm{var}}(Y \mid X=x) \equiv \sigma^2$, as well as a quartic (biweight) kernel for $\hat{m}$. Under these assumptions, we know that the optimal bandwidth minimising the asymptotic mean integrated squared error is given by 

$$h_{AMISE} = n^{-1/5} \bigg( \frac{35 \sigma^2 \vert supp(X) \vert}{\theta_{22}} \bigg)^{1/5}, \quad \theta_{22}= \int \lbrace m''(x) \rbrace^2 f_{X}(x) dx$$

where the two unknown quantities $\sigma^2$ and $\theta_{22}$ can be estimated by parametric OLS. For instance, one can

-   Block the sample in $N$ blocks and fit, in each block $j$, the model $$y_i = \beta_{0j} + \beta_{1j} x_i + \beta_{2j} x_i^2 + \beta_{3j} x_i^3 + \beta_{4j} x_i^4 + \epsilon_i$$ to obtain estimate $$\hat{m}\_j = \hat{\beta}\_{0j} + \hat{\beta}\_{1j} x_i + \hat{\beta}\_{2j} x_i^2 + \hat{\beta}\_{3j} x_i^3 + \hat{\beta}_{4j} x_i^4$$


-   Estimate the unknown quantities by
   
    $$\hat{\theta}\_{22}(N) = \frac{1}{n} \sum_{i=1}^n \sum_{j=1}^N \hat{m}\_j''(X_i) \hat{m}\_j''(X_i)  \mathbb{1}\_{X_i \in \mathcal{X}\_j}$$
    
     $$\hat{\sigma}^2(N) = \frac{1}{n-5N} \sum_{i=1}^n \sum_{j=1}^N \lbrace Y_i - \hat{m}\_j(X_i) \rbrace^2 \mathbb{1}_{X_i \in \mathcal{X}_j}$$



## Task

The goal is to perform a simulation study to assess the impact of some parameters/hyperparameters on the optimal bandwidth $h_{AMISE}$. For instance, we will assume the following setting for the simulation study

-   a covariate $X$ from a beta distribution Beta $(\alpha,\beta)$ 
-   a response values $Y = m(X) + \epsilon$ where
    -   the regression function $m$ is given by $\sin\left\lbrace\left(\frac{x}{3}+0.1\right)^{-1}\right\rbrace$
    -   $\epsilon \sim \mathcal{N}(0,\sigma^2)$
-   fix $\sigma^2$ at some visually appealing value (e.g., $\sigma^2=1$ should be fine)

From there, estimate $h_{AMISE}$ as described above and in [Lecture 3](https://math-516-517-main.github.io/math_517_website/lectures/04_Smoothing.pdf) for different values of the following parameters/hyperparameters

 -  the sample size $n$ (to assess the impact of the amount of available information), 
 -  the block size $N$ in the estimation of the unknown quantities $\sigma^2$ and $\theta_{22}$, and 
 -  the parameters $\alpha$ and $\beta$ of the beta density of the covariate (to assess the impact of the shape of the distribution of the covariate). 

Comment and report your findings using appropriate visualisation tools. Possible questions to address:

 - How does $h_{AMISE}$ behave when $N$ grows? Can you explain why?
 - Should $N$ depend on $n$? Why?
 - What happens when the number of observations varies a lot between different regions in the support of $X$? How is this linked to the parameters of the Beta distribution?

When assessing the effect of the sample size $n$ or the density support of the covariate $X$ on the optimal bandwidth $h_{AMISE}$, you can fix the value of $N$ at an optimal value. This value could be considered as optimal in the sense that it minimizes the Mallow's $C_p$

$$ C_p(N)=\text{RSS}(N) / \lbrace \text{RSS} (N_{\max }) / (n-5 N_{\max })\rbrace -(n-10 N), $$

where 

$$\text{RSS}(N) =  \sum_{i=1}^n \sum_{j=1}^N \lbrace Y_i - \hat{m}\_j(X_i) \rbrace^2 \mathbb{1}_{X_i \in \mathcal{X}_j}$$

and $N_{\max}= \max \lbrace \min (\lfloor n / 20\rfloor, 5 ), 1\rbrace$; see [Ruppert et al. (1995)](https://sites.stat.washington.edu/courses/stat527/s13/readings/Ruppert_etal_JASA_1995.pdf) for choosing the optimal block size.

**Optional**: One could visualize the results using a Shiny App with sliders for tweaking the different values of the parameters/hyperparameters. 

# Deliverables

Use the template to hand-in your assignment (in a PDF format) that should include your answers to both the theoretical and practical questions (explaining and commenting your findings). If using Quarto, use the `report.qmd` file to write your report [^1]. Alternatively, any other format that produces a pdf file is fine (e.g., Rmarkdown), as long as the code is well commented and made available too.

The report should be self-contained and we should have access to all code necessary to reproduce your results (for help with virtual environments, see [this page](../resources/tips/virtual_environments.html)).


**Optional**: You can create a [Shiny App](https://shiny.rstudio.com/tutorial/written-tutorial/lesson1/) (or an interactive Jupyter notebook, Pluto notebook) as it might be easier to play with the data. An interactive report (shiny app, notebook with interactive sliders, ...) is not required, but it is a good way to explore the data interactively. Make sure to add a link to the Shiny App in your report, if it is not interactive.


[^1]: `.qmd` files are [Quarto](https://quarto.org/) files, which can be used to write markdown report with R, Julia, and Python (sometimes simultaneously).

