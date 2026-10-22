Title:

Content selection saved. Describe the issue below:

Description:

![](https://arxiv.org/static/base/1.0.1/images/icons/smileybones-small.svg)arXiv is now an independent nonprofit! [Learn more](https://info.arxiv.org/about) ×

[License: arXiv.org perpetual non-exclusive license](https://info.arxiv.org/help/license/index.html#licenses-available)

arXiv:2609.26679v1 \[cs.LG\] 22 Sep 2026

# A Spectral Theory of Grokking: Weight Decay induces Feature Learning

Lenz Pracher
Affiliation: Arnold-Sommerfeld-Center for Theoretical Physics,
Affiliation: Ludwig-Maximilians-Universität München, Munich, Germany
Pascal de Jong
Affiliation: Arnold-Sommerfeld-Center for Theoretical Physics,
Affiliation: Ludwig-Maximilians-Universität München, Munich, Germany
Oskar Lieshaus
Affiliation: Arnold-Sommerfeld-Center for Theoretical Physics,
Affiliation: Ludwig-Maximilians-Universität München, Munich, Germany
Alan Jeffares
Affiliation: Department of Mathematics, University of Cambridge, Cambridge, UK\*Equal contribution.   †Corresponding author (rulands@lmu.de).
Steffen Rulands
Affiliation: Department of Applied Physics, Standford University, Standford, CA, USA

###### Abstract

In grokking an early fit to the training data separates from a much later improvement in generalization. During this delay, training can move from a fixed neural tangent kernel (NTK) regime to one in which task-relevant kernel eigendirections continue to evolve. We provide a quantitative theory for how this transition from lazy to rich learning can produce delayed generalization. For homogeneous networks trained with squared loss and L2L\_{2} weight decay, we show that a finite residual remains after memorization, with larger residual fractions in target components associated with smaller NTK eigenvalues. These residuals feed back into the dynamics of the NTK itself, and projecting the resulting dynamics onto task-relevant spectral directions yields a reduced system in which residual-driven kernel growth competes with weight decay. This system predicts that the grokking timescale is controlled by the product of learning rate and weight decay, that feature learning slows logarithmically near a critical decay above which task-aligned NTK structure can no longer support generalization, and that stronger decay can prevent fitting altogether. We test these predictions in modular addition. In a homogeneous MLP, task-aligned Fourier structure continues to emerge in the NTK after training accuracy has saturated, and an 84×\\times90-grid of trained networks across varying learning rate and weight decay recovers the predicted phase geometry and inverse-product scaling of the generalization time with learning rate and weight decay. A one-block Transformer shows similar macroscopic phase structure in a 42×\\times45-grid, as well as the same transition-time scaling despite violating exact homogeneity. Together, these results provide a mechanistic derivation connecting post-fit feature learning to both the onset of generalization and its phase structure in the learning rate and weight decay plane.

|     |
| --- |
|  |

## 1 Introduction

Analyzing the dynamics of generalization in deep neural networks is challenging, because optimization processes that lead to memorizing or overfitting, representation learning, and changes in validation set predictions usually occur together. However, grokking separates these timescales, as a network can reach high training accuracy, while remaining inaccurate on held-out examples for thousands of additional updates and only then generalize ( [Power et al., 2022](https://arxiv.org/html/2609.26679v1#bib.bib28 "")). This separation lets us ask what happens after the training labels are already fitted and predicted correctly. _What error remains after the training labels are already fitted, how does it reshape the learned features, and what sets the delay before those changes improve held-out predictions?_

The neural tangent kernel (NTK) provides a framework for distinguishing approximately fixed-feature dynamics from feature learning. Let 𝐟\\mathbf{f} denote the network outputs, J=∇𝜽𝐟J=\\nabla\_{\\boldsymbol{\\theta}}\\mathbf{f} the parameter Jacobian, and K=J​J⊤K=JJ^{\\top} the empirical NTK, which quantifies how network outputs adapt to changes to the parameters. For gradient flow on a differentiable data loss ℒdata\\mathcal{L}\_{\\mathrm{data}}, the chain rule gives

|     |     |     |     |
| --- | --- | --- | --- |
|  | 𝐟˙data=−K​∇𝐟ℒdata,\\dot{\\mathbf{f}}\_{\\rm data}=-K\\nabla\_{\\mathbf{f}}\\mathcal{L}\_{\\mathrm{data}}\\,, |  | (1) |

which is architecture-independent. When KK remains constant, training is thus described by a fixed kernel. On the other hand, changes in KK reflect changes in the tangent features of the network ( [Jacot et al., 2018](https://arxiv.org/html/2609.26679v1#bib.bib9 "")). The neural tangent hierarchy extends this description by expressing the evolution of KK in terms of higher-order tangent tensors ( [Huang & Yau, 2020](https://arxiv.org/html/2609.26679v1#bib.bib8 "")). The residual error, the deviation between the network prediction and the true output, then enters the dynamics of KK itself. Hence, the neural tangent hierarchy framework links the residual that is left after saturation of the training accuracy to subsequent changes in the NTK structure that are relevant to the task at hand.

Here, we derive a post-fit mechanism for delayed generalization in homogeneous networks trained with squared loss and coupled L2L\_{2} weight decay. Weight decay leaves residual error after fitting, and this drives continued task-aligned evolution of the empirical NTK. For an approximately fixed NTK during the initial fit, we prove that the fraction of each target component remaining as residual error decreases monotonically with the corresponding NTK eigenvalue. The neural tangent hierarchy couples these residuals to the evolution of the NTK, allowing task-aligned tangent structure to continue developing after memorization. Modular addition provides a natural Fourier basis for projecting these dynamics onto a reduced residual-NTK system in which residual-driven growth competes with weight decay. Its adiabatic limit predicts a slow post-fit timescale controlled by the product η​λW\\eta\\lambda\_{W} of learning rate and weight decay, together with finite-training boundaries in the corresponding (η,λW\\eta,\\lambda\_{W})-phase space. Empirically, we observe the corresponding post-fit NTK organization and optimizer-space structure in a homogeneous MLP, with the same scaling of generalization time with learning rate and weight decay also appearing in a Transformer. Figure [1](https://arxiv.org/html/2609.26679v1#S1.F1 "Figure 1 ‣ 1 Introduction ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning") summarizes the proposed mechanism.

Fast fitK⁡(t)≃K0K(t)\\simeq K\_{0}training accuracysaturatesPost-fit residuala finite residual remainsafter memorizationResidual–NTK feedbackΛ˙NTK∼−a​σ\\dot{\\Lambda}\_{\\rm NTK}\\sim-a\\sigmaσ˙∼−ΛNTK​σ\\dot{\\sigma}\\sim-\\Lambda\_{\\rm NTK}\\sigmaHeld-out transitiontask-aligned NTK structurebecomes strong enoughto support generalizationKK reshapes σ\\sigmaσ\\sigma reshapes KKupdates sssfits\_{\\rm fit}sgroks\_{\\rm grok}training accuracyheld-out accuracyslow coordinate τ=η​λW​(s−sfit)\\tau=\\eta\\lambda\_{W}(s-s\_{\\rm fit})

Figure 1: A fast, near-fixed-kernel stage fits the training data but leaves a finite residual under coupled weight decay. In the reduced dynamics that follow, the residual and task-aligned NTK strength evolve together. That is, the kernel reshapes the residual, and the residual reshapes the kernel, until task-aligned structure is strong enough to move held-out predictions across their decision margins. The lower panel separates the training fit from the later generalization transition and shows the slow coordinate τ≃η​λW​(s−sfit)\\tau\\simeq\\eta\\lambda\_{W}(s-s\_{\\rm fit}).

##### Related works.

Delayed generalization can arise without evolving neural features. Linear estimators, Gaussian processes, and logistic models exhibit grokking under suitable conditions ( [Levi et al., 2024](https://arxiv.org/html/2609.26679v1#bib.bib14 ""); [Miller et al., 2024](https://arxiv.org/html/2609.26679v1#bib.bib24 ""); [Beck et al., 2025](https://arxiv.org/html/2609.26679v1#bib.bib2 "")). Most directly,  [Xu et al. (2026)](https://arxiv.org/html/2609.26679v1#bib.bib35 "") prove end-to-end grokking in overparameterized ridge regression trained by gradient descent with weight decay and derive quantitative hyperparameter dependence of the grokking time, while [Kim (2026)](https://arxiv.org/html/2609.26679v1#bib.bib12 "") derives an exactly solvable late-time weight-decay relaxation recovering the (η​λW)−1(\\eta\\lambda\_{W})^{-1} scale in linear models. These results show that (η​λW)−1(\\eta\\lambda\_{W})^{-1}-type timing alone does not distinguish feature learning from fixed-feature dynamics.

Mechanistic work on neural-network grokking instead points to gradual representation change during the apparent plateau. In modular arithmetic, this includes the emergence of Fourier-structured circuits ( [Nanda et al., 2023](https://arxiv.org/html/2609.26679v1#bib.bib27 "")), structured features and competition between memorizing and generalizing solutions ( [Liu et al., 2022](https://arxiv.org/html/2609.26679v1#bib.bib16 ""); [Liu et al., 2023a](https://arxiv.org/html/2609.26679v1#bib.bib17 ""); [Varma et al., 2024](https://arxiv.org/html/2609.26679v1#bib.bib34 ""); [Merrill et al., 2023](https://arxiv.org/html/2609.26679v1#bib.bib22 ""); [Ding et al., 2024](https://arxiv.org/html/2609.26679v1#bib.bib5 "")), and transitions from an early kernel-like regime to later feature learning ( [Kumar et al., 2024](https://arxiv.org/html/2609.26679v1#bib.bib13 ""); [Lyu et al., 2024](https://arxiv.org/html/2609.26679v1#bib.bib19 ""); [Mohamadi et al., 2024](https://arxiv.org/html/2609.26679v1#bib.bib25 ""); [Rubin et al., 2024](https://arxiv.org/html/2609.26679v1#bib.bib30 ""); [Tian, 2026](https://arxiv.org/html/2609.26679v1#bib.bib33 "")). A related line of empirical work tracks how the tangent features of the network reorganize over this transition. In particular, leading empirical-NTK eigenfunctions become increasingly task-relevant as delayed generalization emerges ( [Sanguino Bautiste et al., 2024](https://arxiv.org/html/2609.26679v1#bib.bib31 "")), while substantial empirical-NTK movement can precede the representational changes that more closely track generalization ( [Zheng et al., 2024](https://arxiv.org/html/2609.26679v1#bib.bib36 "")), and empirical-NTK eigenspaces in modular-arithmetic MLPs and Transformers align increasingly with Fourier features used by the learned solution ( [Lin, 2025](https://arxiv.org/html/2609.26679v1#bib.bib15 "")). These works show that learned representations and the empirical NTK can continue to evolve during grokking, but leave open what drives this post-memorization feature learning and how the evolution of task-aligned features is linked to generalization. As pointed out by [Xu et al. (2026)](https://arxiv.org/html/2609.26679v1#bib.bib35 ""), a rigorous theoretical analysis that connects grokking to the transition from the lazy to the rich regime of training neural networks is missing.

In this work, we provide, to the best of our knowledge, the first quantitative theory linking this transition from lazy to rich training to delayed generalization. Appendix [A](https://arxiv.org/html/2609.26679v1#A1 "Appendix A Related mechanisms for delayed generalization ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning") develops the connections to other works in more detail. Specifically, we make the following contributions:

##### Contributions.

We study delayed generalization in homogeneous networks trained with squared loss and coupled L2L\_{2} weight decay and derive a mechanism connecting post-fit residual error to continued task-aligned feature learning:

- •


We prove that, after the initial fit, coupled weight decay leaves a finite residual in the eigendirections of an approximately fixed NTK, with the largest relative residuals in directions that are weakly represented by the current features (Theorem [1](https://arxiv.org/html/2609.26679v1#Thmtheorem1 "Theorem 1 (Persistent ridge residual under homogeneous dynamics). ‣ 2.1 Weak target modes retain larger residual fractions ‣ 2 Post-fit residuals drive tangent feature learning ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning")).

- •


We show that these residuals enter the subsequent evolution of the empirical NTK. Projecting the neural tangent hierarchy onto a task-aligned Fourier direction yields a reduced residual–NTK system in which residual-driven growth competes with weight decay (Theorem [2](https://arxiv.org/html/2609.26679v1#Thmtheorem2 "Theorem 2 (Spectral selection in the reduced tangent-mode system). ‣ 2.2 Residuals drive task-aligned tangent growth ‣ 2 Post-fit residuals drive tangent feature learning ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning")).

- •


We derive a decay-controlled slow timescale for post-fit feature learning, predicting (η​λW)−1(\\eta\\lambda\_{W})^{-1} scaling in the grokking regime and high-decay cutoffs where feature growth or training fit fails (Section [2.3](https://arxiv.org/html/2609.26679v1#S2.SS3 "2.3 Grokking boundaries in the (𝜂,𝜆_𝑊) plane ‣ 2 Post-fit residuals drive tangent feature learning ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning")).

- •


We test these predictions in homogeneous MLPs on modular addition, observing continued Fourier organization of the NTK and the predicted weak-decay timing and optimizer-space structure. A non-homogeneous one-block Transformer shows similar macroscopic behavior (Section [3](https://arxiv.org/html/2609.26679v1#S3 "3 Experiments ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning")).


## 2 Post-fit residuals drive tangent feature learning

We start by deriving the post-fit residual under coupled weight decay, and we show how it drives subsequent NTK evolution and reduce the dynamics to a task-aligned spectral mode. We then connect this mode growth to held-out generalization and the resulting optimizer-space grokking boundaries.

### 2.1 Weak target modes retain larger residual fractions

We perform our analysis in the setting of modular arithmetic, where the task is to learn the mapping (a,b)↦c=(a+b)modp(a,b)\\mapsto c=(a+b)\\bmod p, for a given prime pp. Inputs and targets are one-hot encoded such that the network has pp output coordinates. Let ℤp\\mathbb{Z}\_{p} denote the additive group of integers modulo pp. Its characters are χk​(z)=exp⁡(2​π​i​k​z/p)\\chi\_{k}(z)=\\exp(2\\pi ikz/p), k∈ℤpk\\in\\mathbb{Z}\_{p}, such that for output class cc the target can be written as

|     |     |     |     |
| --- | --- | --- | --- |
|  | yc(a,b)=1p∑k=0p−1e−2πikc/pχk(a)χk(b).y\_{c}(a,b)=\\frac{1}{p}\\sum\_{k=0}^{p-1}e^{-2\\pi ikc/p}\\,\\chi\_{k}(a)\\chi\_{k}(b). |  | (2) |

Because the target contains only products χk​(a)​χk​(b)\\chi\_{k}(a)\\chi\_{k}(b) with the same frequency kk, this specifies the relevant, task-aligned Fourier components that a successful model must learn to generalize. Note that the fixed-kernel residual dynamics derived below are not specific to modular addition, but hold for arbitrary datasets and targets under the homogeneous squared-loss setting. Modularity enters only when we use the task symmetry to identify an explicit Fourier basis for the subsequent spectral reduction.

Let nn be the number of training examples and stack all pp-dimensional predictions and targets into 𝐟,𝐲∈ℝn​p\\mathbf{f},\\mathbf{y}\\in\\mathbb{R}^{np}. With residual 𝝈=𝐟−𝐲∈ℝn​p\\boldsymbol{\\sigma}=\\mathbf{f}-\\mathbf{y}\\in\\mathbb{R}^{np}, the training objective can be written as

|     |     |     |     |
| --- | --- | --- | --- |
|  | ℒ⁡(𝜽)=𝒩2​‖𝝈‖2+λW2​‖𝜽‖2,\\mathcal{L}(\\boldsymbol{\\theta})=\\frac{\\mathcal{N}}{2}\\left\\lVert\\boldsymbol{\\sigma}\\right\\rVert^{2}+\\frac{\\lambda\_{W}}{2}\\left\\lVert\\boldsymbol{\\theta}\\right\\rVert^{2}\\,, |  | (3) |

where 𝒩\\mathcal{N} is an arbitrary normalization factor111



As shown below, the slow update time governing grokking dynamics is invariant to the specific normalization., and λW\\lambda\_{W} is the coefficient of the L2L\_{2} penalty. Throughout, lowercase λW\\lambda\_{W} denotes optimizer weight decay, whereas capital Λ\\Lambda denotes an NTK spectral strength or eigenvalue. We define λWnorm=λW/𝒩{\\lambda\_{W}^{\\mathrm{norm}}}=\\lambda\_{W}/\\mathcal{N} as the decay coefficient in normalized gradient-flow time. One gradient-descent update with learning rate η\\eta advances this time by 𝒩​η\\mathcal{N}\\eta to first order. For PP trainable parameters, J=∇𝜽𝐟∈ℝn​p×PJ=\\nabla\_{\\boldsymbol{\\theta}}\\mathbf{f}\\in\\mathbb{R}^{np\\times P} is the parameter Jacobian and K=J​J⊤∈ℝn​p×n​pK=JJ^{\\top}\\in\\mathbb{R}^{np\\times np} is the empirical NTK, which measures how strongly the network output can change along any direction in output space under parameter updates. Gradient flow then leads to

|     |     |     |     |
| --- | --- | --- | --- |
|  | 𝜽˙=−J⊤​𝝈−λWnorm​𝜽,𝝈˙=−K​𝝈−λWnorm​J​𝜽.\\dot{\\boldsymbol{\\theta}}=-J^{\\top}\\boldsymbol{\\sigma}-{\\lambda\_{W}^{\\mathrm{norm}}}\\boldsymbol{\\theta},\\qquad\\dot{\\boldsymbol{\\sigma}}=-K\\boldsymbol{\\sigma}-{\\lambda\_{W}^{\\mathrm{norm}}}J\\boldsymbol{\\theta}. |  | (4) |

Furthermore, if 𝐟𝜽\\mathbf{f}\_{\\boldsymbol{\\theta}} is DD-homogeneous under uniform parameter rescaling, Euler’s identity gives J​𝜽=D​𝐟J\\boldsymbol{\\theta}=D\\mathbf{f}. Bias-free feed-forward ReLU networks satisfy this identity, whereas biases, normalization, mixed-degree residual paths, and standard attention generally break it. Substituting this identity into Eq. ( [4](https://arxiv.org/html/2609.26679v1#S2.E4 "In 2.1 Weak target modes retain larger residual fractions ‣ 2 Post-fit residuals drive tangent feature learning ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning")) closes the dynamics in residual space.

###### Theorem 1(Persistent ridge residual under homogeneous dynamics).

Let 𝐟𝛉\\mathbf{f}\_{\\boldsymbol{\\theta}} be DD-homogeneous and let λWnorm>0{\\lambda\_{W}^{\\mathrm{norm}}}>0. Gradient flow on Eq. ( [3](https://arxiv.org/html/2609.26679v1#S2.E3 "In 2.1 Weak target modes retain larger residual fractions ‣ 2 Post-fit residuals drive tangent feature learning ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning")) obeys

|     |     |     |     |
| --- | --- | --- | --- |
|  | 𝝈˙=−(K+D​λWnorm​I)​𝝈−D​λWnorm​𝐲.\\dot{\\boldsymbol{\\sigma}}=-(K+D{\\lambda\_{W}^{\\mathrm{norm}}}I)\\boldsymbol{\\sigma}-D{\\lambda\_{W}^{\\mathrm{norm}}}\\mathbf{y}. |  | (5) |

If K⁡(t)≡KK(t)\\equiv K, then

|     |     |     |     |
| --- | --- | --- | --- |
|  | 𝝈⁡(t)⟶𝝈∗=−D​λWnorm​(K+D​λWnorm​I)−1​𝐲.\\boldsymbol{\\sigma}(t)\\longrightarrow\\boldsymbol{\\sigma}\_{\*}=-D{\\lambda\_{W}^{\\mathrm{norm}}}(K+D{\\lambda\_{W}^{\\mathrm{norm}}}I)^{-1}\\mathbf{y}. |  | (6) |

Every finite-eigenvalue direction with a nonzero target projection therefore retains a nonzero residual.

For ReLU networks, the derivation holds on each time interval along the gradient-flow trajectory over which the activation pattern is fixed, and extends piecewise across activation-boundary crossings. We prove the theorem in Appendix [B.1](https://arxiv.org/html/2609.26679v1#A2.SS1 "B.1 Proof of Theorem ‣ Appendix B Exact homogeneous function-space dynamics ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning"). Now, we diagonalize KK, writing K​𝐮j=Λj​𝐮jK\\mathbf{u}\_{j}=\\Lambda\_{j}\\mathbf{u}\_{j}, where 𝐮j\\mathbf{u}\_{j} is an eigenvector of KK with eigenvalue Λj\\Lambda\_{j}, and denote the corresponding target and residual coefficients by yj=𝐮j⊤​𝐲y\_{j}=\\mathbf{u}\_{j}^{\\top}\\mathbf{y} and σj=𝐮j⊤​𝝈\\sigma\_{j}=\\mathbf{u}\_{j}^{\\top}\\boldsymbol{\\sigma}. The fixed-kernel limit in Eq. ( [6](https://arxiv.org/html/2609.26679v1#S2.E6 "In Theorem 1 (Persistent ridge residual under homogeneous dynamics). ‣ 2.1 Weak target modes retain larger residual fractions ‣ 2 Post-fit residuals drive tangent feature learning ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning")) then gives the equilibrium residual coefficient in the jjth NTK eigendirection,

|     |     |     |     |
| --- | --- | --- | --- |
|  | σj,∗=−D​λWnormΛj+D​λWnorm​yj.\\sigma\_{j,\*}=-\\frac{D{\\lambda\_{W}^{\\mathrm{norm}}}}{\\Lambda\_{j}+D{\\lambda\_{W}^{\\mathrm{norm}}}}\\,y\_{j}. |  | (7) |

This shows that for fixed \|yj\|\|y\_{j}\|, the fraction of the jjth target component that remains as residual error is larger for smaller Λj\\Lambda\_{j}. These weaker modes correspond to directions that the current features represent poorly. However, in a finite network the NTK can continue to evolve after training accuracy has saturated. Therefore, we now study whether the larger relative residuals in weak task-aligned modes can drive continued task-aligned NTK evolution.

### 2.2 Residuals drive task-aligned tangent growth

For a one-hidden-layer, D=2D=2, bias-free ReLU MLP, combining the neural tangent hierarchy ( [Huang & Yau, 2020](https://arxiv.org/html/2609.26679v1#bib.bib8 "")) with the two-homogeneous weight-decay terms gives

|     |     |     |     |
| --- | --- | --- | --- |
|  | 𝝈˙=−(K+2​λWnorm​I)​𝝈−2​λWnorm​𝐲,K˙=−K(2)​𝝈−2​λWnorm​K.\\dot{\\boldsymbol{\\sigma}}=-(K+2{\\lambda\_{W}^{\\mathrm{norm}}}I)\\boldsymbol{\\sigma}-2{\\lambda\_{W}^{\\mathrm{norm}}}\\mathbf{y},\\qquad\\dot{K}=-K^{(2)}\\boldsymbol{\\sigma}-2{\\lambda\_{W}^{\\mathrm{norm}}}K. |  | (8) |

Here K(2)K^{(2)} is a third-order tangent tensor, and K(2)​𝝈K^{(2)}\\boldsymbol{\\sigma} denotes contraction over its residual index,
(K(2)​𝝈)i​j:=∑kKi​j​k(2)​σk.(K^{(2)}\\boldsymbol{\\sigma})\_{ij}:=\\sum\_{k}K^{(2)}\_{ijk}\\sigma\_{k}.
This contraction couples the residual and kernel dynamics, and again employing two-homogeneity the parameter decay term becomes −2​λWnorm​K-2{\\lambda\_{W}^{\\mathrm{norm}}}K. Although this term shrinks the NTK, the residual-dependent part −K(2)​𝝈-K^{(2)}\\boldsymbol{\\sigma} can simultaneously increase NTK strength along task-aligned directions.

To proceed, choose a normalized real Fourier direction 𝐮\\mathbf{u} with nonzero target projection and orient it so that y=𝐮⊤​𝐲≥0y=\\mathbf{u}^{\\top}\\mathbf{y}\\geq 0. We define the task-aligned NTK strength Λ\\Lambda and projected residual σ\\sigma by

|     |     |     |     |
| --- | --- | --- | --- |
|  | Λ=𝐮⊤​K​𝐮,σ=𝐮⊤​𝝈.\\Lambda=\\mathbf{u}^{\\top}K\\mathbf{u},\\qquad\\sigma=\\mathbf{u}^{\\top}\\boldsymbol{\\sigma}. |  | (9) |

Λ\\Lambda measures how strongly the network can change its output along the selected task direction under parameter updates.

The Fourier structure of modular addition makes 𝐮\\mathbf{u} a natural task direction. This motivates a local one-mode approximation in which 𝐮\\mathbf{u} remains approximately an eigendirection of the NTK, K​𝐮≃Λ​𝐮K\\mathbf{u}\\simeq\\Lambda\\mathbf{u}, with weak mixing into other directions, and its dynamics are driven mainly by the residual component along 𝐮\\mathbf{u}. Appendix [C.1](https://arxiv.org/html/2609.26679v1#A3.SS1 "C.1 Assumptions behind the modal closure ‣ Appendix C One-mode neural-tangent-hierarchy closure ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning") states these conditions explicitly. Section [3](https://arxiv.org/html/2609.26679v1#S3 "3 Experiments ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning") later shows the corresponding Fourier organization empirically. Projecting the hierarchy dynamics onto 𝐮\\mathbf{u} then gives

|     |     |     |     |
| --- | --- | --- | --- |
|  | σ˙≃−(Λ+2​λWnorm)​σ−2​λWnorm​y,Λ˙≃−a​σ−2​λWnorm​Λ.\\dot{\\sigma}\\simeq-(\\Lambda+2{\\lambda\_{W}^{\\mathrm{norm}}})\\sigma-2{\\lambda\_{W}^{\\mathrm{norm}}}y,\\qquad\\dot{\\Lambda}\\simeq-a\\sigma-2{\\lambda\_{W}^{\\mathrm{norm}}}\\Lambda. |  | (10) |

Here, a⁡(𝜽)a(\\boldsymbol{\\theta}) is the local coupling between the projected residual and the residual-driven change in Λ\\Lambda, as defined in Appendix [C.1](https://arxiv.org/html/2609.26679v1#A3.SS1 "C.1 Assumptions behind the modal closure ‣ Appendix C One-mode neural-tangent-hierarchy closure ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning"). We consider a⁡(𝜽)>0a(\\boldsymbol{\\theta})>0, qualitatively consistent with the observed post-fit emergence of task-aligned NTK structure. It implies that correcting the output along 𝐮\\mathbf{u} tends to increase the NTK strength in the same direction. This sign is a dynamical condition rather than a consequence of homogeneity or symmetry. Equation ( [10](https://arxiv.org/html/2609.26679v1#S2.E10 "In 2.2 Residuals drive task-aligned tangent growth ‣ 2 Post-fit residuals drive tangent feature learning ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning")) then describes a feedback. For σ<0\\sigma<0, larger Λ\\Lambda accelerates residual relaxation, while the remaining residual drives Λ\\Lambda upward.

We further assume that a⁡(𝜽)a(\\boldsymbol{\\theta}) is locally constant over the post-fit interval considered. Appendix [C.1](https://arxiv.org/html/2609.26679v1#A3.SS1 "C.1 Assumptions behind the modal closure ‣ Appendix C One-mode neural-tangent-hierarchy closure ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning") shows that once the residual is close to its instantaneous ridge value, weight decay does not introduce an additional fast timescale for aa, supporting this approximation in the slow regime considered below. After the fast ridge relaxation, σ<0\\sigma<0 for y>0y>0, so −a​σ>0-a\\sigma>0 and the residual increases the NTK strength along the selected task direction. Under the locally constant-aa approximation, the fixed points of Eq. ( [10](https://arxiv.org/html/2609.26679v1#S2.E10 "In 2.2 Residuals drive task-aligned tangent growth ‣ 2 Post-fit residuals drive tangent feature learning ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning")) and their stability can be characterized exactly.

###### Theorem 2(Spectral selection in the reduced tangent-mode system).

Under the locally constant-aa approximation, for λWnorm>0{\\lambda\_{W}^{\\mathrm{norm}}}>0, a>0a>0, and y≥0y\\geq 0, Eq. ( [10](https://arxiv.org/html/2609.26679v1#S2.E10 "In 2.2 Residuals drive task-aligned tangent growth ‣ 2 Post-fit residuals drive tangent feature learning ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning")) has one equilibrium with Λ≥0\\Lambda\\geq 0:

|     |     |     |     |
| --- | --- | --- | --- |
|  | Λ∗=−λWnorm+λWnorm2+a​y,σ∗=−2​λWnorma​Λ∗.\\Lambda\_{\*}=-{\\lambda\_{W}^{\\mathrm{norm}}}+\\sqrt{{\\lambda\_{W}^{\\mathrm{norm}}}^{2}+ay},\\qquad\\sigma\_{\*}=-\\frac{2{\\lambda\_{W}^{\\mathrm{norm}}}}{a}\\Lambda\_{\*}. |  | (11) |

The equilibrium is locally asymptotically stable. It satisfies Λ∗=0\\Lambda\_{\*}=0 for y=0y=0 and Λ∗>0\\Lambda\_{\*}>0 for y>0y>0.

We give the fixed-point and stability calculations in Appendix [C.2](https://arxiv.org/html/2609.26679v1#A3.SS2 "C.2 Proof of Theorem ‣ Appendix C One-mode neural-tangent-hierarchy closure ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning"). Within the reduced system, unsupported directions decay, while positive residual-to-NTK coupling sustains NTK strength for target-aligned directions. Under the adiabatic approximation that residual relaxation is faster than mode motion, we set σ˙≃0\\dot{\\sigma}\\simeq 0 at the current Λ\\Lambda and obtain

|     |     |     |     |
| --- | --- | --- | --- |
|  | σad​(Λ)=−2​λWnorm​yΛ+2​λWnorm,Λ˙=2​λWnorm​(a​yΛ+2​λWnorm−Λ).\\sigma\_{\\rm ad}(\\Lambda)=-\\frac{2{\\lambda\_{W}^{\\mathrm{norm}}}y}{\\Lambda+2{\\lambda\_{W}^{\\mathrm{norm}}}},\\qquad\\dot{\\Lambda}=2{\\lambda\_{W}^{\\mathrm{norm}}}\\left(\\frac{ay}{\\Lambda+2{\\lambda\_{W}^{\\mathrm{norm}}}}-\\Lambda\\right). |  | (12) |

This shows that mode evolution occurs on a normalized-time scale 1/λWnorm1/{\\lambda\_{W}^{\\mathrm{norm}}}. Since t≃𝒩​η​st\\simeq\\mathcal{N}\\eta s after ss updates, we define the normalization-invariant slow time

|     |     |     |     |
| --- | --- | --- | --- |
|  | τ:=λWnorm​t≃η​λW​s.\\tau:={\\lambda\_{W}^{\\mathrm{norm}}}t\\simeq\\eta\\lambda\_{W}s\\,. |  | (13) |

Here, ss is the optimizer update index and η​λW\\eta\\lambda\_{W} is the per-update decay scale. If sfits\_{\\rm fit} denotes the end of the initial fitting stage, the subsequent slow-time increment is Δ​τ≃η​λW​(s−sfit)\\Delta\\tau\\simeq\\eta\\lambda\_{W}(s-s\_{\\rm fit}). Thus the reduced dynamics predict the leading scaling of post-fit transition times with (η​λW)−1(\\eta\\lambda\_{W})^{-1}.

### 2.3 Grokking boundaries in the (η,λW)(\\eta,\\lambda\_{W}) plane

The same adiabatic calculation shows how the task-aligned NTK strength Λ\\Lambda controls how much of the corresponding target component is expressed in the network output. Since σ=𝐮⊤​(𝐟−𝐲)\\sigma=\\mathbf{u}^{\\top}(\\mathbf{f}-\\mathbf{y}), the projected output coefficient f=𝐮⊤​𝐟f=\\mathbf{u}^{\\top}\\mathbf{f} satisfies

|     |     |     |     |
| --- | --- | --- | --- |
|  | f=y+σad​(Λ)=g⁡(Λ)​y,g⁡(Λ)=ΛΛ+2​λWnorm.f=y+\\sigma\_{\\rm ad}(\\Lambda)=g(\\Lambda)y,\\quad g(\\Lambda)=\\frac{\\Lambda}{\\Lambda+2{\\lambda\_{W}^{\\mathrm{norm}}}}. |  | (14) |

Hence, increasing Λ\\Lambda along a task-aligned mode increases the corresponding component of the training outputs continuously. This training coordinate is related to generalization through the evolution of held-out margins, the difference between the correct-class logit and the largest competing logit, along the same trajectory. Because modular addition is represented by several matched Fourier components as shown in Eq. ( [2](https://arxiv.org/html/2609.26679v1#S2.E2 "In 2.1 Weak target modes retain larger residual fractions ‣ 2 Post-fit residuals drive tangent feature learning ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning")), these margins can change smoothly as the corresponding components grow, while the predicted label changes only when a margin crosses zero. If many held-out examples cross that threshold at similar mode strengths, smooth spectral growth can therefore produce a sharp rise in validation accuracy. Appendix [D.1](https://arxiv.org/html/2609.26679v1#A4.SS1 "D.1 From smooth mode growth to an accuracy transition ‣ Appendix D Adiabatic reduction and finite-training boundary ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning") formalizes this connection using the empirical distribution of the mode strengths at which individual held-out examples change class.

The scalar dynamics in Eq. ( [10](https://arxiv.org/html/2609.26679v1#S2.E10 "In 2.2 Residuals drive task-aligned tangent growth ‣ 2 Post-fit residuals drive tangent feature learning ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning")) track a single task-aligned Fourier mode and do not resolve the individual held-out margins that produce the sharp accuracy transition. We therefore now replace the collection of margin crossings by a threshold Λg\\Lambda\_{\\mathrm{g}} on the task-aligned NTK strength Λ\\Lambda. Reaching Λg\\Lambda\_{\\mathrm{g}} represents reaching a chosen held-out-accuracy threshold. We denote the NTK strength at the beginning of the slow stage by Λ0\\Lambda\_{0}. Further, we consider a total of TT training updates, and recall the stable equilibrium Λ∗\\Lambda\_{\*} from Eq. ( [11](https://arxiv.org/html/2609.26679v1#S2.E11 "In Theorem 2 (Spectral selection in the reduced tangent-mode system). ‣ 2.2 Residuals drive task-aligned tangent growth ‣ 2 Post-fit residuals drive tangent feature learning ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning")). Since Eq. ( [12](https://arxiv.org/html/2609.26679v1#S2.E12 "In 2.2 Residuals drive task-aligned tangent growth ‣ 2 Post-fit residuals drive tangent feature learning ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning")) gives Λ˙>0\\dot{\\Lambda}>0 for Λ<Λ∗\\Lambda<\\Lambda\_{\*}, a trajectory with Λ0<Λg<Λ∗\\Lambda\_{0}<\\Lambda\_{\\mathrm{g}}<\\Lambda\_{\*} crosses the threshold Λg\\Lambda\_{\\mathrm{g}} in finite time. Separating variables in Eq. ( [12](https://arxiv.org/html/2609.26679v1#S2.E12 "In 2.2 Residuals drive task-aligned tangent growth ‣ 2 Post-fit residuals drive tangent feature learning ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning")) and integrating then gives the boundary in the (η,λW)(\\eta,\\lambda\_{W}) plane between trajectories that reach Λg\\Lambda\_{\\mathrm{g}} within TT updates and those that remain below it.

###### Corollary 1(Finite-training grokking boundary).

Under the one-mode, adiabatic, and gradient-flow approximations,

|     |     |     |     |
| --- | --- | --- | --- |
|  | η∗​(λW)=12​λW​T​∫Λ0ΛgΛ+2​λWnorma​y−Λ2−2​λWnorm​Λ​𝑑Λ.\\eta\_{\*}(\\lambda\_{W})=\\frac{1}{2\\lambda\_{W}T}\\int\_{\\Lambda\_{0}}^{\\Lambda\_{\\mathrm{g}}}\\frac{\\Lambda+2{\\lambda\_{W}^{\\mathrm{norm}}}}{ay-\\Lambda^{2}-2{\\lambda\_{W}^{\\mathrm{norm}}}\\Lambda}\\,d\\Lambda\\,. |  | (15) |

At small normalized decay and away from the fixed-point boundary, the leading dependence is η∗∝1/λW\\eta\_{\*}\\propto 1/\\lambda\_{W}.

A derivation and closed form solution of this integral can be found in Appendices [D.2](https://arxiv.org/html/2609.26679v1#A4.SS2 "D.2 Adiabatic slow equation ‣ Appendix D Adiabatic reduction and finite-training boundary ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning")– [D.3](https://arxiv.org/html/2609.26679v1#A4.SS3 "D.3 Closed form, critical slowing, and the fitting cutoff ‣ Appendix D Adiabatic reduction and finite-training boundary ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning"). Equation ( [15](https://arxiv.org/html/2609.26679v1#S2.E15 "In Corollary 1 (Finite-training grokking boundary). ‣ 2.3 Grokking boundaries in the (𝜂,𝜆_𝑊) plane ‣ 2 Post-fit residuals drive tangent feature learning ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning")) predicts that increasing Λ0\\Lambda\_{0} removes a positive portion of the crossing-time integral and thereby shortens the delay before grokking occurs. We elaborate on this prediction and provide an empirical test in Appendices [F.1](https://arxiv.org/html/2609.26679v1#A6.SS1 "F.1 Earlier task alignment shortens the predicted delay ‣ Appendix F Directional NTK-alignment intervention ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning")– [F.2](https://arxiv.org/html/2609.26679v1#A6.SS2 "F.2 A temporary NTK intervention advances held-out generalization ‣ Appendix F Directional NTK-alignment intervention ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning"). This finite-training grokking boundary is obtained under the assumption that the threshold is reachable in the first place, Λg<Λ∗\\Lambda\_{\\mathrm{g}}<\\Lambda\_{\*}. Whether or not this is possible follows from the limiting case Λg=Λ∗\\Lambda\_{\\mathrm{g}}=\\Lambda\_{\*}, which gives

|     |     |     |     |
| --- | --- | --- | --- |
|  | λWmode=𝒩​a​y−Λg22​Λg,\\lambda\_{W}^{\\mathrm{mode}}=\\mathcal{N}\\,\\frac{ay-\\Lambda\_{\\mathrm{g}}^{2}}{2\\Lambda\_{\\mathrm{g}}}, |  | (16) |

provided a​y>Λg2ay>\\Lambda\_{\\mathrm{g}}^{2}. For λW<λWmode\\lambda\_{W}<\\lambda\_{W}^{\\mathrm{mode}}, the equilibrium lies above Λg\\Lambda\_{\\mathrm{g}} so the mode can reach Λg\\Lambda\_{\\mathrm{g}} in finite time, with Corollary [1](https://arxiv.org/html/2609.26679v1#Thmcorollary1 "Corollary 1 (Finite-training grokking boundary). ‣ 2.3 Grokking boundaries in the (𝜂,𝜆_𝑊) plane ‣ 2 Post-fit residuals drive tangent feature learning ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning") determining whether this occurs within TT updates. For λW≥λWmode\\lambda\_{W}\\geq\\lambda\_{W}^{\\mathrm{mode}} the threshold cannot be crossed in finite time. As λW\\lambda\_{W} approaches this cutoff from below, the net growth rate of the task-aligned mode at Λ=Λg\\Lambda=\\Lambda\_{\\mathrm{g}} approaches zero, causing η∗​(λW)\\eta\_{\*}(\\lambda\_{W}) to diverge logarithmically as λW→λWmode\\lambda\_{W}\\to\\lambda\_{W}^{\\mathrm{mode}} (Appendix [D.3](https://arxiv.org/html/2609.26679v1#A4.SS3 "D.3 Closed form, critical slowing, and the fitting cutoff ‣ Appendix D Adiabatic reduction and finite-training boundary ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning")). This shows that as λW\\lambda\_{W} approaches λWmode\\lambda\_{W}^{\\mathrm{mode}} from below, reaching Λg\\Lambda\_{\\mathrm{g}} and thereby generalizing within a fixed update budget requires an increasingly large learning rate. In the (η,λW)(\\eta,\\lambda\_{W}) plane, this produces an upward turn of the finite-time boundary near the cutoff at λWmode\\lambda\_{W}^{\\mathrm{mode}}.

A further constraint arises from fitting the training data. This boundary depends on the relative gains of multiple output modes and is therefore not fixed exactly by the one-mode dynamics. To retain a scalar description, we summarize the onset of fitting failure by the gain of an effective training mode. Requiring g⁡(Λfit)≥gming(\\Lambda\_{\\mathrm{fit}})\\geq g\_{\\min} gives

|     |     |     |     |
| --- | --- | --- | --- |
|  | λW≤𝒩​Λfit​(1−gmin)2​gmin,\\lambda\_{W}\\leq\\mathcal{N}\\,\\frac{\\Lambda\_{\\mathrm{fit}}(1-g\_{\\min})}{2g\_{\\min}}, |  | (17) |

This provides an approximately vertical effective decay scale beyond which the network no longer reaches the chosen training-accuracy criterion.

Finally, sufficiently large learning rates encounter the discrete-time stability edge of the approximately fixed-kernel dynamics. As derived in Appendix [D.4](https://arxiv.org/html/2609.26679v1#A4.SS4 "D.4 Discrete-time stability of the frozen-kernel dynamics ‣ Appendix D Adiabatic reduction and finite-training boundary ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning"), linear stability requires

|     |     |     |     |
| --- | --- | --- | --- |
|  | η<2𝒩​Λmax​(K)+2​λW,\\eta<\\frac{2}{\\mathcal{N}\\Lambda\_{\\max}(K)+2\\lambda\_{W}}\\,, |  | (18) |

where Λmax​(K)\\Lambda\_{\\max}(K) is the largest eigenvalue of the frozen NTK. This is the frozen-kernel analogue of the usual edge-of-stability condition ( [Cohen et al., 2021](https://arxiv.org/html/2609.26679v1#bib.bib4 "")). Because the NTK subsequently evolves, it provides only a local estimate of the large-η\\eta boundary of the full training dynamics. Appendices [D.3](https://arxiv.org/html/2609.26679v1#A4.SS3 "D.3 Closed form, critical slowing, and the fitting cutoff ‣ Appendix D Adiabatic reduction and finite-training boundary ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning")– [D.4](https://arxiv.org/html/2609.26679v1#A4.SS4 "D.4 Discrete-time stability of the frozen-kernel dynamics ‣ Appendix D Adiabatic reduction and finite-training boundary ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning") give the derivations and validity conditions for the fitting and stability bounds.

To summarize, these constraints predict the following organization of the (η,λW)(\\eta,\\lambda\_{W}) plane. When the training data are fitted and Λg\\Lambda\_{\\mathrm{g}} is reachable but the finite-training condition in Corollary [1](https://arxiv.org/html/2609.26679v1#Thmcorollary1 "Corollary 1 (Finite-training grokking boundary). ‣ 2.3 Grokking boundaries in the (𝜂,𝜆_𝑊) plane ‣ 2 Post-fit residuals drive tangent feature learning ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning") is not satisfied, the network remains in the memorization regime over the available training interval. Grokking is possible when the training fit succeeds (Eq. ( [17](https://arxiv.org/html/2609.26679v1#S2.E17 "In 2.3 Grokking boundaries in the (𝜂,𝜆_𝑊) plane ‣ 2 Post-fit residuals drive tangent feature learning ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning"))), the task-aligned NTK strength can reach Λg\\Lambda\_{\\mathrm{g}} (Eq. ( [16](https://arxiv.org/html/2609.26679v1#S2.E16 "In 2.3 Grokking boundaries in the (𝜂,𝜆_𝑊) plane ‣ 2 Post-fit residuals drive tangent feature learning ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning"))), the threshold is crossed within TT updates (Eq. ( [15](https://arxiv.org/html/2609.26679v1#S2.E15 "In Corollary 1 (Finite-training grokking boundary). ‣ 2.3 Grokking boundaries in the (𝜂,𝜆_𝑊) plane ‣ 2 Post-fit residuals drive tangent feature learning ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning"))), and gradient descent remains stable (Eq. ( [18](https://arxiv.org/html/2609.26679v1#S2.E18 "In 2.3 Grokking boundaries in the (𝜂,𝜆_𝑊) plane ‣ 2 Post-fit residuals drive tangent feature learning ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning"))). Increasing weight decay can eventually prevent either task-mode growth or training fit, while sufficiently large learning rates produce a separate instability boundary.

## 3 Experiments

We now test these predictions and measure post-fit NTK reorganization as well as the (η,λW)(\\eta,\\lambda\_{W})-plane boundaries, and whether similar transition-time behavior appears in Transformers. Appendix [G](https://arxiv.org/html/2609.26679v1#A7 "Appendix G Experimental configurations and analysis conventions ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning") provides experimental details, and all results can be reproduced from our GitHub page222



The code will be made publicly available upon publication..

### 3.1 Task-aligned NTK structure develops after fitting

![Refer to caption](https://arxiv.org/html/2609.26679v1/figure_ntk_accuracy_eigenvectors.png)Figure 2: Post-fit change in the normalized label-space NTK for a bias-free MLP on addition modulo 9797. (a) We project the sample NTK onto the true-label, average by output label, and normalize the resulting 97×9797\\times 97 label kernel. (b) Training and held-out accuracy for the same run. (c) Four leading eigenvectors of the same normalized label kernel, showing Fourier-like standing waves that sharpen after training accuracy is already high.

We trained a bias-free one-hidden-layer ReLU MLP on addition modulo 9797, using half of the samples for training and the remainder for evaluation. We computed the empirical NTK, projected it onto the true-label output of each sample, and averaged the resulting entries by output label. This isolates tangent feature organization with respect to the task labels rather than individual examples. To further isolate structure from changes in overall NTK scale, we normalized the resulting 97×9797\\times 97 kernel. Figure [2](https://arxiv.org/html/2609.26679v1#S3.F2 "Figure 2 ‣ 3.1 Task-aligned NTK structure develops after fitting ‣ 3 Experiments ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning") a shows that periodic structure in this kernel strengthens between updates 3,0003{,}000 and 20,00020{,}000, while Figure [2](https://arxiv.org/html/2609.26679v1#S3.F2 "Figure 2 ‣ 3.1 Task-aligned NTK structure develops after fitting ‣ 3 Experiments ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning") b shows that training accuracy has already saturated during much of this evolution. In Figure [2](https://arxiv.org/html/2609.26679v1#S3.F2 "Figure 2 ‣ 3.1 Task-aligned NTK structure develops after fitting ‣ 3 Experiments ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning") c, the leading eigenvectors dynamics show the emergence of standing-wave profiles in label space, consistent with the Fourier task structure in Eq. ( [2](https://arxiv.org/html/2609.26679v1#S2.E2 "In 2.1 Weak target modes retain larger residual fractions ‣ 2 Post-fit residuals drive tangent feature learning ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning")). Their continued organization after fitting is also qualitatively consistent with the post-fit growth of task-aligned NTK modes described by Eq. ( [10](https://arxiv.org/html/2609.26679v1#S2.E10 "In 2.2 Residuals drive task-aligned tangent growth ‣ 2 Post-fit residuals drive tangent feature learning ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning")).

### 3.2 Grokking boundaries in the MLP optimizer plane

![Refer to caption](https://arxiv.org/html/2609.26679v1/figure_mlp_phase_and_clock.png)Figure 3: Optimizer-plane structure for the homogeneous MLP on addition modulo 2323. (a) Outcomes after 24,00024{,}000 updates on an 84×9084\\times 90 learning-rate/decay grid. The overlays show the calibrated finite-time boundary, a separately fitted inverse-decay relation, the mode-reachability cutoff, the training-fit cutoff, and the calibrated frozen-kernel stability form. (b) First held-out 10%10\\% crossing among runs that eventually meet the 90%90\\% grokking criterion, plotted against (η​λW)−1(\\eta\\lambda\_{W})^{-1}. The dashed segment indicates the predicted scaling s10∝(η​λW)−1s\_{10}\\propto(\\eta\\lambda\_{W})^{-1} and is not fitted to the data.

Next, we trained an (84×\\times90)-grid of MLPs for varying (η,λW)(\\eta,\\lambda\_{W}) on addition modulo 2323 for a fixed total of 24,00024{,}000 updates. Figure [3](https://arxiv.org/html/2609.26679v1#S3.F3 "Figure 3 ‣ 3.2 Grokking boundaries in the MLP optimizer plane ‣ 3 Experiments ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning") shows the resulting (ηCLOSE(\\eta,OPENλW)\\lambda\_{W})-plane, categorized into memorizing, grokking, forgetting, and non-fitting runs, as described in Appendix [G.2](https://arxiv.org/html/2609.26679v1#A7.SS2 "G.2 High-resolution MLP phase sweep on addition modulo 23 ‣ Appendix G Experimental configurations and analysis conventions ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning"). At small η​λW\\eta\\lambda\_{W}, the task-aligned NTK strength Λ\\Lambda does not have enough time to reach the generalization threshold Λg\\Lambda\_{\\mathrm{g}} within the fixed training budget, producing the broad memorization region in Figure [3](https://arxiv.org/html/2609.26679v1#S3.F3 "Figure 3 ‣ 3.2 Grokking boundaries in the MLP optimizer plane ‣ 3 Experiments ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning") a. The red curve fits the finite-training grokking boundary from Eq. ( [15](https://arxiv.org/html/2609.26679v1#S2.E15 "In Corollary 1 (Finite-training grokking boundary). ‣ 2.3 Grokking boundaries in the (𝜂,𝜆_𝑊) plane ‣ 2 Post-fit residuals drive tangent feature learning ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning")). Note that since the fit-parameters are inferred from the boundary, they are not direct measurements of Λ⁡(t)\\Lambda(t). Hence, this comparison tests the predicted optimizer dependence of the crossing time rather than the absolute amount of task-aligned NTK growth. At weak decay, it follows the transition from memorization to grokking and closely tracks the yellow η∝λW−1\\eta\\propto\\lambda\_{W}^{-1} fit. As λW\\lambda\_{W} increases, the finite-time curve turns upward toward the blue mode-reachability cutoff from Eq. ( [16](https://arxiv.org/html/2609.26679v1#S2.E16 "In 2.3 Grokking boundaries in the (𝜂,𝜆_𝑊) plane ‣ 2 Post-fit residuals drive tangent feature learning ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning")), beyond which the task-aligned NTK strength cannot reach the required threshold. The green line shows the training-fit cutoff from Eq. ( [17](https://arxiv.org/html/2609.26679v1#S2.E17 "In 2.3 Grokking boundaries in the (𝜂,𝜆_𝑊) plane ‣ 2 Post-fit residuals drive tangent feature learning ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning")), while the black line corresponds to the frozen-kernel stability condition in Eq. ( [18](https://arxiv.org/html/2609.26679v1#S2.E18 "In 2.3 Grokking boundaries in the (𝜂,𝜆_𝑊) plane ‣ 2 Post-fit residuals drive tangent feature learning ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning")), fitted to the observed large-learning-rate boundary. The full fitting procedures for all curves are given in Appendix [G.2.1](https://arxiv.org/html/2609.26679v1#A7.SS2.SSS1 "G.2.1 Boundary extraction and calibration ‣ G.2 High-resolution MLP phase sweep on addition modulo 23 ‣ Appendix G Experimental configurations and analysis conventions ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning").

Figure [3](https://arxiv.org/html/2609.26679v1#S3.F3 "Figure 3 ‣ 3.2 Grokking boundaries in the MLP optimizer plane ‣ 3 Experiments ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning") b tests the predicted slow-time scaling. For every run that reaches 90%90\\% held-out accuracy, we plot the first update s10s\_{10} at which that accuracy reaches 10%10\\% against (η​λW)−1(\\eta\\lambda\_{W})^{-1}. Equation ( [13](https://arxiv.org/html/2609.26679v1#S2.E13 "In 2.2 Residuals drive task-aligned tangent growth ‣ 2 Post-fit residuals drive tangent feature learning ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning")) predicts s10∝(η​λW)−1s\_{10}\\propto(\\eta\\lambda\_{W})^{-1} if the transition is governed by the slow time τ≃η​λW​s\\tau\\simeq\\eta\\lambda\_{W}s. Indeed, the observed crossing times follow this scaling across the grokking region.

### 3.3 Grokking boundaries and transition-time scaling in a Transformer

![Refer to caption](https://arxiv.org/html/2609.26679v1/figure_transformer_phase_and_clock.png)Figure 4: Phase and transition-time measurements for a one-block Transformer on the same modulo-2323 split. (a) Outcomes on a 42×4542\\times 45 optimizer grid with an inverse-decay fit to the main memorization–grokking boundary. (b) First held-out 80%80\\% crossing among runs that ultimately grok at the 90%90\\% criterion, plotted against (η​λW)−1(\\eta\\lambda\_{W})^{-1}. The dashed segment indicates the scaling s80∝(η​λW)−1s\_{80}\\propto(\\eta\\lambda\_{W})^{-1}.

Biases, LayerNorm, residual paths, and attention all break the global homogeneity used in Theorems [1](https://arxiv.org/html/2609.26679v1#Thmtheorem1 "Theorem 1 (Persistent ridge residual under homogeneous dynamics). ‣ 2.1 Weak target modes retain larger residual fractions ‣ 2 Post-fit residuals drive tangent feature learning ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning") and [2](https://arxiv.org/html/2609.26679v1#Thmtheorem2 "Theorem 2 (Spectral selection in the reduced tangent-mode system). ‣ 2.2 Residuals drive task-aligned tangent growth ‣ 2 Post-fit residuals drive tangent feature learning ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning"). However, differentiating K=J​J⊤K=JJ^{\\top} under gradient flow still yields an exact residual-dependent contribution to K˙\\dot{K} for any differentiable model (Appendix [E](https://arxiv.org/html/2609.26679v1#A5 "Appendix E Jacobian-mediated feature dynamics beyond homogeneity ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning")). We therefore tested whether the homogeneous predictions also hold in a non-homogeneous case, and trained a (42×\\times45)-grid of Transformers. In line with the MLP phase diagram, Figure [4](https://arxiv.org/html/2609.26679v1#S3.F4 "Figure 4 ‣ 3.3 Grokking boundaries and transition-time scaling in a Transformer ‣ 3 Experiments ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning") a shows a weak-decay memorization region, an intermediate grokking band, and failure at stronger decay or large learning rate. The fitted curve of the form η=A/λW\\eta=A/\\lambda\_{W} describes the grokking-memorization boundary well, in line with the homogeneous theory. Additionally, within the grokking networks, Figure [4](https://arxiv.org/html/2609.26679v1#S3.F4 "Figure 4 ‣ 3.3 Grokking boundaries and transition-time scaling in a Transformer ‣ 3 Experiments ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning") b exhibits the same linear scaling of the first held-out 80%80\\% crossing s80s\_{80} with (η​λW)−1(\\eta\\lambda\_{W})^{-1}. We use 80%80\\% here because early 10%10\\% accuracy crossings are often reversible in practice. This suggests that the optimizer-level phase and transition-time scaling may extend beyond exact homogeneity.

## 4 Limitations and Outlook

Our theory holds for homogeneous networks trained with squared loss, coupled weight decay, and full-batch gradient flow. Extending the reduced dynamics to more general objectives and architectures is a natural next step. The Transformer experiments already show that similar macroscopic optimizer-space structure can persist without exact homogeneity, but the corresponding microscopic dynamics remain open. Further, the scalar reduction follows a single decoupled task direction. Although the natural Fourier basis of modular addition motivates this, interactions among task-relevant components are neglected, and more general datasets need not provide an equally natural spectral basis. Future work should therefore extend the dynamics to interacting task directions and identify suitable coordinates beyond cyclic modular arithmetic. In this direction, [Marchetti et al. (2026)](https://arxiv.org/html/2609.26679v1#bib.bib21 "") show that for sequential group-composition tasks, networks learn irreducible representations one at a time, suggesting a route from the present one-mode reduction toward multiple representation-theoretic components. Finally, modular addition and grokking provide explicit task structure and separate fitting from generalization, but parts of our mechanism are more general. The fixed-kernel residual result does not rely on modular arithmetic, and the residual-dependent contribution to NTK evolution persists for differentiable models. This raises the question of whether our framework also applies to broader representation-learning problems. The J-space framework of [Gurnee et al. (2026)](https://arxiv.org/html/2609.26679v1#bib.bib6 "") provides a possible connection, using an activation-to-output Jacobian to identify representations with strong downstream influence. The NTK similarly uses the parameter Jacobian to track task-aligned directions during training. Establishing whether these directions emerge together would connect the optimization dynamics studied here to the broader formation of internal representations used for generalization.

## 5 Conclusions

We developed a spectral theory of grokking in which coupled weight decay leaves a finite post-fit residual, with target components at smaller NTK eigenvalues retaining a larger residual fraction. We used the neural tangent hierarchy to show that this residual drives task-aligned NTK growth. The resulting slow dynamics determine whether task-aligned NTK strength can grow sufficiently far and sufficiently quickly within the training budget, while training fit and discrete-time stability impose additional constraints. Empirically, a homogeneous MLP on modular addition shows continued Fourier organization of the NTK after training accuracy saturates, an MLP hyperparameter sweep recovers the predicted (η,λW)(\\eta,\\lambda\_{W}) phase geometry and (η​λW)−1(\\eta\\lambda\_{W})^{-1} transition-time scaling, and a one-block Transformer reproduces the same macroscopic phase structure and transition-time scaling despite lacking exact homogeneity. Together, these results suggest that residual error left after fitting can actively shape how and when neural networks acquire generalizing representations, connecting feature learning, generalization time, and optimizer-space structure within a single mechanistic framework.

## Reproducibility statement

We provide the proofs and derivations of the homogeneous results in Appendices [B.1](https://arxiv.org/html/2609.26679v1#A2.SS1 "B.1 Proof of Theorem ‣ Appendix B Exact homogeneous function-space dynamics ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning")– [D.4](https://arxiv.org/html/2609.26679v1#A4.SS4 "D.4 Discrete-time stability of the frozen-kernel dynamics ‣ Appendix D Adiabatic reduction and finite-training boundary ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning") and the differentiable-model identities in Appendix [E.1](https://arxiv.org/html/2609.26679v1#A5.SS1 "E.1 Exact differentiable-model identities ‣ Appendix E Jacobian-mediated feature dynamics beyond homogeneity ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning"). Appendix [F.2](https://arxiv.org/html/2609.26679v1#A6.SS2 "F.2 A temporary NTK intervention advances held-out generalization ‣ Appendix F Directional NTK-alignment intervention ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning") records the complete paired intervention, including architectures, seeds, splits, losses, optimizers, kernel normalization, and reporting convention. Appendices [G.1](https://arxiv.org/html/2609.26679v1#A7.SS1 "G.1 Spectral diagnostics on addition modulo 97 ‣ Appendix G Experimental configurations and analysis conventions ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning")– [G.4](https://arxiv.org/html/2609.26679v1#A7.SS4 "G.4 Reduced-mode potential illustration ‣ Appendix G Experimental configurations and analysis conventions ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning") give the corresponding details for the spectral diagnostic, MLP and Transformer phase sweeps, boundary calibration, and reduced-mode illustration. The accompanying code333



The code will be made publicly available upon publication. centralizes the configurations and provides training and analysis entry points.

## AI use statement

AI was used to assist with text editing, coding, and with literature research. AI was not involved in research ideation, the development of theoretical results, research methodology, and experimental design. All AI-assisted outputs were reviewed, revised where necessary, and verified by the authors.

## References

- Baratin et al. (2021)
Aristide Baratin, Thomas George, César Laurent, R. Devon Hjelm, Guillaume Lajoie, Pascal Vincent, and Simon Lacoste-Julien.

Implicit regularization via neural feature alignment.

In _Proceedings of the 24th International Conference on Artificial Intelligence and Statistics_, volume 130 of _Proceedings of Machine Learning Research_, pp. 2269–2277, 2021.

- Beck et al. (2025)
Alon Beck, Noam Itzhak Levi, and Yohai Bar-Sinai.

Grokking at the edge of linear separability.

In _Proceedings of the 42nd International Conference on Machine Learning_, volume 267 of _Proceedings of Machine Learning Research_, pp. 3307–3334, 2025.

- Bertolotti & Cazzola (2026)
Francesco Bertolotti and Walter Cazzola.

A survey on grokking.

_ACM Computing Surveys_, 58(13):327:1–327:25, 2026.

doi: 10.1145/3814603.

- Cohen et al. (2021)
Jeremy Cohen, Simran Kaur, Yuanzhi Li, J Zico Kolter, and Ameet Talwalkar.

Gradient descent on neural networks typically occurs at the edge of stability.

In _International Conference on Learning Representations_, 2021.

URL [https://openreview.net/forum?id=jh-rTtvkGeM](https://openreview.net/forum?id=jh-rTtvkGeM "").

- Ding et al. (2024)
Xiaoman Delores Ding, Zifan Carl Guo, Eric J. Michaud, Ziming Liu, and Max Tegmark.

Survival of the fittest representation: A case study with modular addition.

_arXiv preprint arXiv:2405.17420_, 2024.

- Gurnee et al. (2026)
Wes Gurnee, Nicholas Sofroniew, Adam Pearce, Mateusz Piotrowski, Isaac Kauvar, Runjin Chen, Anna Soligo, Paul Bogdan, Euan Ong, Rowan Wang, T. Ben Thompson, David Abrahams, Subhash Kantamneni, Emmanuel Ameisen, Joshua Batson, and Jack Lindsey.

Verbalizable representations form a global workspace in language models.

_arXiv preprint arXiv:2607.15495_, 2026.

- He et al. (2026)
Jianliang He, Leda Wang, Siyu Chen, and Zhuoran Yang.

On the mechanism and dynamics of modular addition: Fourier features, lottery ticket, and grokking.

_arXiv preprint arXiv:2602.16849_, 2026.

- Huang & Yau (2020)
Jiaoyang Huang and Horng-Tzer Yau.

Dynamics of deep neural networks and neural tangent hierarchy.

In _Proceedings of the 37th International Conference on Machine Learning_, 2020.

- Jacot et al. (2018)
Arthur Jacot, Franck Gabriel, and Clement Hongler.

Neural tangent kernel: Convergence and generalization in neural networks.

In _Advances in Neural Information Processing Systems_, 2018.

- Jeffares et al. (2024)
Alan Jeffares, Alicia Curth, and Mihaela van der Schaar.

Deep learning through a telescoping lens: A simple model provides empirical insights on grokking, gradient boosting and beyond.

In _Advances in Neural Information Processing Systems_, volume 37, 2024.

- Junior et al. (2025)
Tikeng Notsawo Pascal Junior, Guillaume Dumas, and Guillaume Rabusseau.

Grokking beyond the euclidean norm of model parameters.

In _Proceedings of the 42nd International Conference on Machine Learning_, volume 267 of _Proceedings of Machine Learning Research_, pp. 28552–28618, 2025.

- Kim (2026)
Taeyoung Kim.

Grokking on the weight-decay clock: A rate hierarchy from softly broken symmetries.

_arXiv preprint arXiv:2607.23967_, 2026.

- Kumar et al. (2024)
Tanishq Kumar, Blake Bordelon, Samuel J. Gershman, and Cengiz Pehlevan.

Grokking as the transition from lazy to rich training dynamics.

In _International Conference on Learning Representations_, 2024.

- Levi et al. (2024)
Noam Itzhak Levi, Alon Beck, and Yohai Bar-Sinai.

Grokking in linear estimators: A solvable model that groks without understanding.

In _International Conference on Learning Representations_, 2024.

- Lin (2025)
Jennifer Lin.

Feature identification via the empirical ntk.

_arXiv preprint arXiv:2510.00468_, 2025.

- Liu et al. (2022)
Ziming Liu, Ouail Kitouni, Niklas Nolte, Eric J. Michaud, Max Tegmark, and Mike Williams.

Towards understanding grokking: An effective theory of representation learning.

In _Advances in Neural Information Processing Systems_, 2022.

- Liu et al. (2023a)
Ziming Liu, Eric J. Michaud, and Max Tegmark.

Omnigrok: Grokking beyond algorithmic data.

In _International Conference on Learning Representations_, 2023a.

- Liu et al. (2023b)
Ziming Liu, Ziqian Zhong, and Max Tegmark.

Grokking as compression: A nonlinear complexity perspective.

_arXiv preprint arXiv:2310.05918_, 2023b.

- Lyu et al. (2024)
Kaifeng Lyu, Jikai Jin, Zhiyuan Li, Simon Shaolei Du, Jason D. Lee, and Wei Hu.

Dichotomy of early and late phase implicit biases can provably induce grokking.

In _International Conference on Learning Representations_, 2024.

- Mallinar et al. (2025)
Neil Rohit Mallinar, Daniel Beaglehole, Libin Zhu, Adityanarayanan Radhakrishnan, Parthe Pandit, and Mikhail Belkin.

Emergence in non-neural models: Grokking modular arithmetic via average gradient outer product.

In _Proceedings of the 42nd International Conference on Machine Learning_, volume 267 of _Proceedings of Machine Learning Research_, pp. 42834–42856, 2025.

- Marchetti et al. (2026)
Giovanni Luca Marchetti, Daniel Kunin, Adele Myers, Francisco Acosta, and Nina Miolane.

Sequential group composition: A window into the mechanics of deep learning.

In _Forty-third International Conference on Machine Learning_, 2026.

URL [https://openreview.net/forum?id=oFD9pc53pF](https://openreview.net/forum?id=oFD9pc53pF "").

- Merrill et al. (2023)
William Merrill, Nikolaos Tsilivis, and Aman Shukla.

A tale of two circuits: Grokking as competition of sparse and dense subnetworks.

_ICLR Workshop on Mathematical and Empirical Understanding of Foundation Models_, 2023.

- Michaud et al. (2023)
Eric J. Michaud, Ziming Liu, Uzay Girit, and Max Tegmark.

The quantization model of neural scaling.

In _Advances in Neural Information Processing Systems_, volume 36, pp. 28699–28722, 2023.

- Miller et al. (2024)
Jack William Miller, Charles O’Neill, and Thang D. Bui.

Grokking beyond neural networks: An empirical exploration with model complexity.

_Transactions on Machine Learning Research_, 2024.

- Mohamadi et al. (2024)
Mohamad Amin Mohamadi, Zhiyuan Li, Lei Wu, and Danica J. Sutherland.

Why do you grok? a theoretical analysis on grokking modular addition.

In _Proceedings of the 41st International Conference on Machine Learning_, volume 235 of _Proceedings of Machine Learning Research_, pp. 35934–35967, 2024.

- Naidu et al. (2026)
Prudhviraj Naidu, Zixian Wang, Leon Bergen, and Ramamohan Paturi.

Quiet feature learning in algorithmic tasks.

In _Proceedings of the AAAI Conference on Artificial Intelligence_, volume 40, pp. 37756–37764, 2026.

- Nanda et al. (2023)
Neel Nanda, Lawrence Chan, Tom Lieberum, Jess Smith, and Jacob Steinhardt.

Progress measures for grokking via mechanistic interpretability.

In _International Conference on Learning Representations_, 2023.

- Power et al. (2022)
Alethea Power, Yuri Burda, Harri Edwards, Igor Babuschkin, and Vedant Misra.

Grokking: Generalization beyond overfitting on small algorithmic datasets.

_arXiv preprint arXiv:2201.02177_, 2022.

- Prieto et al. (2025)
Lucas Prieto, Melih Barsbey, Pedro A. M. Mediano, and Tolga Birdal.

Grokking at the edge of numerical stability.

In _International Conference on Learning Representations_, 2025.

- Rubin et al. (2024)
Noa Rubin, Inbar Seroussi, and Zohar Ringel.

Grokking as a first order phase transition in two layer networks.

In _International Conference on Learning Representations_, 2024.

- Sanguino Bautiste et al. (2024)
Javier Sanguino Bautiste, Gregor Bachmann, Bobby He, Lorenzo Noci, and Thomas Hofmann.

Feature learning dynamics under grokking in a sparse parity task.

_ICML 2024 Workshop on High-dimensional Learning Dynamics: The Emergence of Structure and Reasoning_, 2024.

- Thilak et al. (2022)
Vimal Thilak, Etai Littwin, Shuangfei Zhai, Omid Saremi, Roni Paiss, and Joshua M. Susskind.

The slingshot mechanism: An empirical study of adaptive optimizers and the grokking phenomenon.

In _Has it Trained Yet? Workshop at NeurIPS_, 2022.

- Tian (2026)
Yuandong Tian.

Provable scaling laws of feature emergence from learning dynamics of grokking.

In _International Conference on Learning Representations_, 2026.

- Varma et al. (2024)
Vikrant Varma, Rohin Shah, Zachary Kenton, Janos Kramar, and Ramana Kumar.

Explaining grokking through circuit efficiency.

In _International Conference on Learning Representations_, 2024.

- Xu et al. (2026)
Mingyue Xu, Gal Vardi, and Itay Safran.

To grok grokking: Provable grokking in ridge regression.

_arXiv preprint arXiv:2601.19791_, 2026.

- Zheng et al. (2024)
Xingyu Zheng, Kyle Daruwalla, Ari S. Benjamin, and David Klindt.

Delays in generalization match delayed changes in representational geometry.

In _Proceedings of UniReps: the Second Edition of the Workshop on Unifying Representations in Neural Models_, volume 285 of _Proceedings of Machine Learning Research_, pp. 324–334, 2024.


## Appendix A Related mechanisms for delayed generalization

Grokking has been explained through circuit formation, implicit bias, tangent-feature motion, fixed spectra, and optimizer state. Surveys emphasize that several mechanisms can coexist across losses, architectures, and training regimes ( [Bertolotti & Cazzola, 2026](https://arxiv.org/html/2609.26679v1#bib.bib3 "")). We position our account by asking which dynamical object changes during the plateau and which observations distinguish that change from a slow fixed-feature relaxation.

### A.1 Mechanistic progress and representation competition

Modular arithmetic provides an interpretable Fourier basis for circuit analysis. Nanda et al. ( [Nanda et al., 2023](https://arxiv.org/html/2609.26679v1#bib.bib27 "")) reverse-engineer a trigonometric algorithm in a grokked Transformer and introduce progress measures that reveal gradual circuit formation during the apparent plateau. Varma et al. ( [Varma et al., 2024](https://arxiv.org/html/2609.26679v1#bib.bib34 "")) explain delayed generalization through competition between memorizing and generalizing circuits, while Merrill et al. ( [Merrill et al., 2023](https://arxiv.org/html/2609.26679v1#bib.bib22 "")) study competition between sparse and dense subnetworks. Ding et al. ( [Ding et al., 2024](https://arxiv.org/html/2609.26679v1#bib.bib5 "")) track competing circular Fourier representations with low-dimensional dynamics. He et al. ( [He et al., 2026](https://arxiv.org/html/2609.26679v1#bib.bib7 "")) analyze how frequency competition, phase alignment, and weight decay shape Fourier-feature formation in two-layer networks on modular addition. Across a broader set of algorithmic Transformer tasks, Naidu et al. ( [Naidu et al., 2026](https://arxiv.org/html/2609.26679v1#bib.bib26 "")) find causally relevant features developing during long plateaus before the output loss improves. Other representation-level studies connect grokking to changes in norms, learned features, and compression ( [Liu et al., 2022](https://arxiv.org/html/2609.26679v1#bib.bib16 ""); [Liu et al., 2023a](https://arxiv.org/html/2609.26679v1#bib.bib17 ""); [Liu et al., 2023b](https://arxiv.org/html/2609.26679v1#bib.bib18 "")). We use the same task-defined Fourier basis but study the residual that drives tangent-feature motion within it.

### A.2 Kernel-to-feature transitions and empirical NTK dynamics

Several theories describe grokking as a transition from an early linearized regime to later feature learning. Kumar et al. ( [Kumar et al., 2024](https://arxiv.org/html/2609.26679v1#bib.bib13 "")) use a controllable laziness parameter to induce or remove grokking in polynomial regression, MNIST, and modular addition. Lyu et al. ( [Lyu et al., 2024](https://arxiv.org/html/2609.26679v1#bib.bib19 "")) prove a separation between early kernel-like and late implicit biases in homogeneous networks with weight decay. Mohamadi et al. ( [Mohamadi et al., 2024](https://arxiv.org/html/2609.26679v1#bib.bib25 "")) show that an early permutation-equivariant kernel regime can be sample inefficient for modular addition, whereas bounded-norm feature-learning solutions generalize from fewer examples. Rubin et al. ( [Rubin et al., 2024](https://arxiv.org/html/2609.26679v1#bib.bib30 "")) analyze an adaptive kernel and relate grokking in two-layer networks to a first-order phase transition. Tian ( [Tian, 2026](https://arxiv.org/html/2609.26679v1#bib.bib33 "")) derives a sequence of lazy, independent-feature, and interacting-feature stages with scaling laws for feature emergence and generalization.

Empirical NTKs distinguish departure from the lazy regime from the task content acquired by the kernel. On sparse parity, Sanguino Bautiste et al. ( [Sanguino Bautiste et al., 2024](https://arxiv.org/html/2609.26679v1#bib.bib31 "")) observe leading NTK eigenfunctions moving from non-predictive directions toward predictive features as generalization emerges. On image classification, Zheng et al. ( [Zheng et al., 2024](https://arxiv.org/html/2609.26679v1#bib.bib36 "")) find substantial empirical-NTK movement before delayed test improvement and closer synchronization between representational geometry and generalization. For modular arithmetic, Lin ( [Lin, 2025](https://arxiv.org/html/2609.26679v1#bib.bib15 "")) shows that leading empirical-NTK eigenspaces align with Fourier feature families in trained MLPs and Transformers and that the alignment changes through grokking. These results establish feature motion but do not identify its source. We derive a residual-dependent equation for task-aligned kernel components. Equation ( [8](https://arxiv.org/html/2609.26679v1#S2.E8 "In 2.2 Residuals drive task-aligned tangent growth ‣ 2 Post-fit residuals drive tangent feature learning ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning")) inserts the post-fit residual into K˙\\dot{K}, and the Fourier projection yields the reduced feedback, slow time, and finite-training boundary. This mechanism is also compatible with dynamic alignment between finite-network tangent features and task directions ( [Baratin et al., 2021](https://arxiv.org/html/2609.26679v1#bib.bib1 "")).

### A.3 Static-feature grokking and the weight-decay transition-time scaling

Fixed-feature models show that delayed generalization can occur without representation change. Levi et al. ( [Levi et al., 2024](https://arxiv.org/html/2609.26679v1#bib.bib14 "")) analyze grokking in linear estimators, Miller et al. ( [Miller et al., 2024](https://arxiv.org/html/2609.26679v1#bib.bib24 "")) document related non-neural behavior, and Beck et al. ( [Beck et al., 2025](https://arxiv.org/html/2609.26679v1#bib.bib2 "")) study long delays near linear separability. Xu et al. ( [Xu et al., 2026](https://arxiv.org/html/2609.26679v1#bib.bib35 "")) prove end-to-end grokking for overparameterized ridge regression trained with gradient descent and weight decay. Their bounds separate early training fit from later population improvement and express the delay in terms of learning rate, decay, sample size, feature dimension, and initialization. Kim ( [Kim, 2026](https://arxiv.org/html/2609.26679v1#bib.bib12 "")) derives a complementary late-time relaxation in linear models with weight decay and heavy-ball optimization; the zero-momentum limit again yields 1/(η​λ)1/(\\eta\\lambda) scaling.

The transition-time scaling in Eq. ( [13](https://arxiv.org/html/2609.26679v1#S2.E13 "In 2.2 Residuals drive task-aligned tangent growth ‣ 2 Post-fit residuals drive tangent feature learning ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning")) is therefore shared by static-feature and feature-learning mechanisms. Our finite-network account adds the empirical NTK as a state variable and predicts task-aligned motion after training accuracy saturates. The normalized kernel measurement, the residual-dependent hierarchy term, and the directional intervention separate this claim from a fixed-feature delay.

Feature learning can also arise outside ordinary neural-network gradient descent. Mallinar et al. ( [Mallinar et al., 2025](https://arxiv.org/html/2609.26679v1#bib.bib20 "")) demonstrate grokking in recursive feature machines driven by an average gradient outer product; block-circulant task features emerge after training loss has vanished. Jeffares et al. ( [Jeffares et al., 2024](https://arxiv.org/html/2609.26679v1#bib.bib10 "")) analyze grokking through a sequence of local linear approximations, providing another time-resolved view of representation change.

### A.4 Jacobian representations and emergent capabilities

Jacobian geometry can isolate directions with direct behavioral leverage. Gurnee et al. ( [Gurnee et al., 2026](https://arxiv.org/html/2609.26679v1#bib.bib6 "")) introduce the Jacobian lens and identify a small “J-space” of verbalizable representations in language models; interventions show that these directions can mediate downstream reasoning and reporting. Their Jacobian transports downstream activations to outputs, whereas our NTK uses the parameter Jacobian, but both approaches focus on directions that couple strongly to behavior. We study how such task-relevant parameter-Jacobian directions form during training.

The quantization model of neural scaling ( [Michaud et al., 2023](https://arxiv.org/html/2609.26679v1#bib.bib23 "")) proposes that knowledge and skills may be acquired in discrete modules even when aggregate loss follows smooth power laws. Grokking offers a small setting in which abrupt behavior can be resolved in time. Equation ( [58](https://arxiv.org/html/2609.26679v1#A4.E58 "In D.1 From smooth mode growth to an accuracy transition ‣ Appendix D Adiabatic reduction and finite-training boundary ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning")) gives one route from a smooth internal coordinate to a sharp output transition, so behavioral discontinuity need not imply discontinuous internal learning. At the same time, task-aligned modes provide a concrete object for testing more modular accounts of capability acquisition.

### A.5 Regularization and optimization mechanisms

Regularization mechanisms extend beyond Euclidean weight decay. Junior et al. ( [Junior et al., 2025](https://arxiv.org/html/2609.26679v1#bib.bib11 "")) show that sparsity, low-rank penalties, implicit regularization, and depth can induce grokking-like transitions. Adaptive optimizers can produce late-training instabilities ( [Thilak et al., 2022](https://arxiv.org/html/2609.26679v1#bib.bib32 "")), while cross-entropy training can involve numerical and logit-scaling effects ( [Prieto et al., 2025](https://arxiv.org/html/2609.26679v1#bib.bib29 "")). The relevant slow variable changes with the objective and optimizer. We focus on homogeneous squared-loss training because coupled L2L\_{2} decay leaves the ridge residual in Theorem [1](https://arxiv.org/html/2609.26679v1#Thmtheorem1 "Theorem 1 (Persistent ridge residual under homogeneous dynamics). ‣ 2.1 Weak target modes retain larger residual fractions ‣ 2 Post-fit residuals drive tangent feature learning ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning") and directly contracts the two-homogeneous empirical NTK in Eq. ( [8](https://arxiv.org/html/2609.26679v1#S2.E8 "In 2.2 Residuals drive task-aligned tangent growth ‣ 2 Post-fit residuals drive tangent feature learning ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning")).

### A.6 Relation to fixed-feature and circuit-level accounts

Our account links the residual left by an early approximately fixed kernel to later tangent-feature motion in a task-defined Fourier direction. The resulting feature strength determines whether held-out margins are reachable and how much optimizer time is required to reach them. Fixed-feature analyses explain how weight decay can create a long transition-time without changing the representation; circuit analyses describe the algorithm implemented after generalization. Residual-dependent neural-tangent-hierarchy dynamics connect the two by specifying how post-fit supervision can reshape the tangent geometry from which a generalizing circuit is built.

## Appendix B Exact homogeneous function-space dynamics

For reference, the notation used throughout the derivations is summarized below.

| Symbol | Meaning |
| --- | --- |
| 𝝈=𝐟−𝐲\\boldsymbol{\\sigma}=\\mathbf{f}-\\mathbf{y} | stacked training residual |
| J=∇𝜽𝐟J=\\nabla\_{\\boldsymbol{\\theta}}\\mathbf{f} | training-set parameter Jacobian |
| K=J​J⊤K=JJ^{\\top} | empirical neural tangent kernel |
| DD | degree of parameter homogeneity |
| 𝒩\\mathcal{N} | global normalization multiplying ‖𝝈‖2/2\\left\\lVert\\boldsymbol{\\sigma}\\right\\rVert^{2}/2 in the implemented loss |
| λW\\lambda\_{W} | optimizer coupled L2L\_{2} weight-decay coefficient |
| λWnorm=λW/𝒩{\\lambda\_{W}^{\\mathrm{norm}}}=\\lambda\_{W}/\\mathcal{N} | normalized decay used in continuous time |
| η\\eta | optimizer gradient-descent learning rate |
| tt | normalized gradient-flow time, t=𝒩​toptt=\\mathcal{N}t\_{\\rm opt} |
| ss | discrete optimizer update index |
| Λ\\Lambda | task-direction NTK strength 𝐮⊤​K​𝐮\\mathbf{u}^{\\top}K\\mathbf{u} |
| aa | projected residual-to-NTK coupling coefficient |
| yy | target coefficient in the selected task direction |
| Λg\\Lambda\_{\\mathrm{g}} | reduced strength representing a held-out criterion |
| TT | finite optimizer update budget |

Table 1: Notation used in the main derivations.

### B.1 Proof of Theorem [1](https://arxiv.org/html/2609.26679v1\#Thmtheorem1 "Theorem 1 (Persistent ridge residual under homogeneous dynamics). ‣ 2.1 Weak target modes retain larger residual fractions ‣ 2 Post-fit residuals drive tangent feature learning ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning")

We absorb the global data-loss prefactor into the normalized time introduced in Section [2.1](https://arxiv.org/html/2609.26679v1#S2.SS1 "2.1 Weak target modes retain larger residual fractions ‣ 2 Post-fit residuals drive tangent feature learning ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning"). The data-gradient term then has unit coefficient, and optimizer weight decay appears through λWnorm{\\lambda\_{W}^{\\mathrm{norm}}},

|     |     |     |     |     |
| --- | --- | --- | --- | --- |
|  | 𝜽˙\\displaystyle\\dot{\\boldsymbol{\\theta}} | =−J⊤​𝝈−λWnorm​𝜽,\\displaystyle=-J^{\\top}\\boldsymbol{\\sigma}-{\\lambda\_{W}^{\\mathrm{norm}}}\\boldsymbol{\\theta}, |  | (19) |
|  | 𝝈˙\\displaystyle\\dot{\\boldsymbol{\\sigma}} | =J​𝜽˙=−J​J⊤​𝝈−λWnorm​J​𝜽=−K​𝝈−λWnorm​J​𝜽.\\displaystyle=J\\dot{\\boldsymbol{\\theta}}=-JJ^{\\top}\\boldsymbol{\\sigma}-{\\lambda\_{W}^{\\mathrm{norm}}}J\\boldsymbol{\\theta}=-K\\boldsymbol{\\sigma}-{\\lambda\_{W}^{\\mathrm{norm}}}J\\boldsymbol{\\theta}. |  | (20) |

For a DD-homogeneous network, uniform rescaling of all trainable weights rescales the output by degree DD. Differentiating this identity at unit scale gives the Euler relation that closes the parameter-space decay term in function space.

|     |     |     |     |     |
| --- | --- | --- | --- | --- |
|  | 𝐟c​𝜽\\displaystyle\\mathbf{f}\_{c\\boldsymbol{\\theta}} | =cD​𝐟𝜽,\\displaystyle=c^{D}\\mathbf{f}\_{\\boldsymbol{\\theta}}, |  | (21) |
|  | dd​c​𝐟c​𝜽\|c=1\\displaystyle\\left.\\frac{d}{dc}\\mathbf{f}\_{c\\boldsymbol{\\theta}}\\right\|\_{c=1} | =J​𝜽=D​𝐟=D⁡(𝝈+𝐲).\\displaystyle=J\\boldsymbol{\\theta}=D\\mathbf{f}=D(\\boldsymbol{\\sigma}+\\mathbf{y}). |  | (22) |

Substituting this relation removes the remaining explicit dependence on 𝜽\\boldsymbol{\\theta}.

|     |     |     |     |
| --- | --- | --- | --- |
|  | 𝝈˙=−(K+D​λWnorm​I)​𝝈−D​λWnorm​𝐲,\\dot{\\boldsymbol{\\sigma}}=-(K+D{\\lambda\_{W}^{\\mathrm{norm}}}I)\\boldsymbol{\\sigma}-D{\\lambda\_{W}^{\\mathrm{norm}}}\\mathbf{y}, |  | (23) |

which is Eq. ( [5](https://arxiv.org/html/2609.26679v1#S2.E5 "In Theorem 1 (Persistent ridge residual under homogeneous dynamics). ‣ 2.1 Weak target modes retain larger residual fractions ‣ 2 Post-fit residuals drive tangent feature learning ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning")).

For fixed KK, the residual equation is affine and linear. The matrix M:=K+D​λWnorm​IM:=K+D{\\lambda\_{W}^{\\mathrm{norm}}}I is positive definite for λWnorm>0{\\lambda\_{W}^{\\mathrm{norm}}}>0, even when KK has null directions, so every residual component relaxes exponentially to a unique fixed point. We set

|     |     |     |
| --- | --- | --- |
|  | M:=K+D​λWnorm​I≻0,𝝈∗:=−D​λWnorm​M−1​𝐲,M:=K+D{\\lambda\_{W}^{\\mathrm{norm}}}I\\succ 0,\\qquad\\boldsymbol{\\sigma}\_{\*}:=-D{\\lambda\_{W}^{\\mathrm{norm}}}M^{-1}\\mathbf{y}, |  |

and the solution can be written as

|     |     |     |     |     |
| --- | --- | --- | --- | --- |
|  | 𝝈˙\\displaystyle\\dot{\\boldsymbol{\\sigma}} | =−M⁡(𝝈−𝝈∗),\\displaystyle=-M(\\boldsymbol{\\sigma}-\\boldsymbol{\\sigma}\_{\*}), |  | (24) |
|  | 𝝈⁡(t)−𝝈∗\\displaystyle\\boldsymbol{\\sigma}(t)-\\boldsymbol{\\sigma}\_{\*} | =e−M​t​(𝝈⁡(0)−𝝈∗).\\displaystyle=e^{-Mt}\\bigl(\\boldsymbol{\\sigma}(0)-\\boldsymbol{\\sigma}\_{\*}\\bigr). |  | (25) |

We diagonalize KK to expose the ridge form. Each eigenmode follows an independent scalar relaxation, and the ratio of its equilibrium residual to its target coefficient is determined by Λj\\Lambda\_{j} relative to D​λWnormD{\\lambda\_{W}^{\\mathrm{norm}}}. For K​𝐮j=Λj​𝐮jK\\mathbf{u}\_{j}=\\Lambda\_{j}\\mathbf{u}\_{j},

|     |     |     |     |     |
| --- | --- | --- | --- | --- |
|  | σ˙j\\displaystyle\\dot{\\sigma}\_{j} | =−(Λj+D​λWnorm)​σj−D​λWnorm​yj,\\displaystyle=-(\\Lambda\_{j}+D{\\lambda\_{W}^{\\mathrm{norm}}})\\sigma\_{j}-D{\\lambda\_{W}^{\\mathrm{norm}}}y\_{j}, |  | (26) |
|  | σj,∗\\displaystyle\\sigma\_{j,\*} | =−D​λWnormΛj+D​λWnorm​yj.\\displaystyle=-\\frac{D{\\lambda\_{W}^{\\mathrm{norm}}}}{\\Lambda\_{j}+D{\\lambda\_{W}^{\\mathrm{norm}}}}\\,y\_{j}. |  | (27) |

For ReLU networks, these identities hold on every interval on which the activation pattern is fixed. At activation-boundary crossings the network remains continuous and the gradient-flow equation holds almost everywhere, so the same function-space relation extends piecewise across the trajectory.

This proves Theorem [1](https://arxiv.org/html/2609.26679v1#Thmtheorem1 "Theorem 1 (Persistent ridge residual under homogeneous dynamics). ‣ 2.1 Weak target modes retain larger residual fractions ‣ 2 Post-fit residuals drive tangent feature learning ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning"). □\\square

## Appendix C One-mode neural-tangent-hierarchy closure

### C.1 Assumptions behind the modal closure

For the normalized Fourier direction 𝐮\\mathbf{u} used in the main text, define

|     |     |     |     |
| --- | --- | --- | --- |
|  | q𝜽:=𝐮⊤​𝐟𝜽,Λ:=𝐮⊤​K​𝐮=‖∇𝜽q𝜽‖2,σ:=𝐮⊤​𝝈.q\_{\\boldsymbol{\\theta}}:=\\mathbf{u}^{\\top}\\mathbf{f}\_{\\boldsymbol{\\theta}},\\qquad\\Lambda:=\\mathbf{u}^{\\top}K\\mathbf{u}=\\left\\lVert\\nabla\_{\\boldsymbol{\\theta}}q\_{\\boldsymbol{\\theta}}\\right\\rVert^{2},\\qquad\\sigma:=\\mathbf{u}^{\\top}\\boldsymbol{\\sigma}. |  | (28) |

The reduction follows one task-bearing direction rather than approximating the full kernel. Over the transition interval, we assume that the kernel acts approximately diagonally on this direction and that the residual driving it is dominated by the component along uu:

|     |     |     |     |
| --- | --- | --- | --- |
|  | 𝝈≈σ​𝐮,K​𝐮≈Λ​𝐮.\\boldsymbol{\\sigma}\\approx\\sigma\\mathbf{u},\\qquad K\\mathbf{u}\\approx\\Lambda\\mathbf{u}. |  | (29) |

These conditions suppress leading-order mixing with the remaining modes.

With

|     |     |     |     |
| --- | --- | --- | --- |
|  | Ki​j​k(2)=⟨∇𝜽Ki​j,∇𝜽fk⟩,K^{(2)}\_{ijk}=\\left\\langle\\nabla\_{\\boldsymbol{\\theta}}K\_{ij},\\nabla\_{\\boldsymbol{\\theta}}f\_{k}\\right\\rangle, |  | (30) |

the projected hierarchy term becomes

|     |     |     |     |     |
| --- | --- | --- | --- | --- |
|  | 𝐮⊤​(K(2)​𝝈)​𝐮\\displaystyle\\mathbf{u}^{\\top}(K^{(2)}\\boldsymbol{\\sigma})\\mathbf{u} | ≈σ​∑i​j​kui​uj​uk​Ki​j​k(2),\\displaystyle\\approx\\sigma\\sum\_{ijk}u\_{i}u\_{j}u\_{k}K^{(2)}\_{ijk}, |  | (31) |
|  | a⁡(𝜽)\\displaystyle a(\\boldsymbol{\\theta}) | :=∑i​j​kui​uj​uk​Ki​j​k(2)=⟨∇𝜽Λ,∇𝜽q𝜽⟩.\\displaystyle:=\\sum\_{ijk}u\_{i}u\_{j}u\_{k}K^{(2)}\_{ijk}=\\left\\langle\\nabla\_{\\boldsymbol{\\theta}}\\Lambda,\\nabla\_{\\boldsymbol{\\theta}}q\_{\\boldsymbol{\\theta}}\\right\\rangle. |  | (32) |

Thus a⁡(𝜽)a(\\boldsymbol{\\theta}) is the local proportionality between the residual coefficient σ\\sigma and the residual-driven change in the NTK strength Λ\\Lambda. When a>0a>0 and σ<0\\sigma<0 for a positive target coefficient, this term increases Λ\\Lambda. The sign of aa is a dynamical property; homogeneity does not determine it. The projected kernel equation is therefore

|     |     |     |     |
| --- | --- | --- | --- |
|  | Λ˙=−a⁡(𝜽)​σ−2​λWnorm​Λ.\\dot{\\Lambda}=-a(\\boldsymbol{\\theta})\\sigma-2{\\lambda\_{W}^{\\mathrm{norm}}}\\Lambda. |  | (33) |

Whenever σ≠0\\sigma\\neq 0, the same equation can be inverted along a measured trajectory:

|     |     |     |     |
| --- | --- | --- | --- |
|  | aeff​(t)=−Λ˙​(t)+2​λWnorm​Λ​(t)σ⁡(t).a\_{\\rm eff}(t)=-\\frac{\\dot{\\Lambda}(t)+2{\\lambda\_{W}^{\\mathrm{norm}}}\\Lambda(t)}{\\sigma(t)}. |  | (34) |

The constant-aa approximation is local and does not require aa to remain fixed throughout training. In the two-homogeneous MLP, both Λ\\Lambda and aa are degree two under uniform parameter rescaling,

|     |     |     |     |
| --- | --- | --- | --- |
|  | Λ⁡(c​𝜽)=c2​Λ​(𝜽),a⁡(c​𝜽)=c2​a​(𝜽).\\Lambda(c\\boldsymbol{\\theta})=c^{2}\\Lambda(\\boldsymbol{\\theta}),\\qquad a(c\\boldsymbol{\\theta})=c^{2}a(\\boldsymbol{\\theta}). |  | (35) |

Under the one-mode parameter approximation,

|     |     |     |     |     |
| --- | --- | --- | --- | --- |
|  | 𝜽˙\\displaystyle\\dot{\\boldsymbol{\\theta}} | ≈−σ​∇𝜽q𝜽−λWnorm​𝜽,\\displaystyle\\approx-\\sigma\\nabla\_{\\boldsymbol{\\theta}}q\_{\\boldsymbol{\\theta}}-{\\lambda\_{W}^{\\mathrm{norm}}}\\boldsymbol{\\theta}, |  | (36) |
|  | a˙\\displaystyle\\dot{a} | =−b⁡(𝜽)​σ−2​λWnorm​a,b⁡(𝜽):=⟨∇𝜽a,∇𝜽q𝜽⟩.\\displaystyle=-b(\\boldsymbol{\\theta})\\sigma-2{\\lambda\_{W}^{\\mathrm{norm}}}a,\\qquad b(\\boldsymbol{\\theta}):=\\left\\langle\\nabla\_{\\boldsymbol{\\theta}}a,\\nabla\_{\\boldsymbol{\\theta}}q\_{\\boldsymbol{\\theta}}\\right\\rangle. |  | (37) |

After the residual relaxes near its instantaneous ridge value, substitution into Eq. ( [37](https://arxiv.org/html/2609.26679v1#A3.E37 "In C.1 Assumptions behind the modal closure ‣ Appendix C One-mode neural-tangent-hierarchy closure ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning")) gives

|     |     |     |     |
| --- | --- | --- | --- |
|  | a˙≃2​λWnorm​(b⁡(𝜽)​yΛ+2​λWnorm−a),\\dot{a}\\simeq 2{\\lambda\_{W}^{\\mathrm{norm}}}\\left(\\frac{b(\\boldsymbol{\\theta})y}{\\Lambda+2{\\lambda\_{W}^{\\mathrm{norm}}}}-a\\right), |  | (38) |

and

|     |     |     |     |
| --- | --- | --- | --- |
|  | dd​t​(aΛ)=σΛ2​(a2−b​Λ).\\frac{d}{dt}\\left(\\frac{a}{\\Lambda}\\right)=\\frac{\\sigma}{\\Lambda^{2}}\\left(a^{2}-b\\Lambda\\right). |  | (39) |

Equations ( [37](https://arxiv.org/html/2609.26679v1#A3.E37 "In C.1 Assumptions behind the modal closure ‣ Appendix C One-mode neural-tangent-hierarchy closure ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning"))–( [38](https://arxiv.org/html/2609.26679v1#A3.E38 "In C.1 Assumptions behind the modal closure ‣ Appendix C One-mode neural-tangent-hierarchy closure ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning")) show that weight decay does not introduce an additional fast timescale for aa after residual relaxation. Pure radial decay also leaves a/Λa/\\Lambda unchanged, so any remaining drift of this ratio is residual driven. We therefore use a locally constant coupling subject to four explicit conditions:

1. 1.


the projected residual and kernel satisfy Eq. ( [29](https://arxiv.org/html/2609.26679v1#A3.E29 "In C.1 Assumptions behind the modal closure ‣ Appendix C One-mode neural-tangent-hierarchy closure ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning"));

2. 2.


a⁡(t)a(t) remains positive over the interval of interest;

3. 3.


a⁡(t)a(t) changes slowly relative to residual relaxation; and

4. 4.


coupling to other task modes is weak enough to be absorbed into the local coefficient aa and the effective threshold Λg\\Lambda\_{\\mathrm{g}}.


The cyclic symmetry of modular addition and the difference-structured split motivate approximate Fourier decoupling. The remaining conditions are assumptions of the scalar reduction.

### C.2 Proof of Theorem [2](https://arxiv.org/html/2609.26679v1\#Thmtheorem2 "Theorem 2 (Spectral selection in the reduced tangent-mode system). ‣ 2.2 Residuals drive task-aligned tangent growth ‣ 2 Post-fit residuals drive tangent feature learning ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning")

With aa fixed locally, we combine the two first-order equations into a second-order equation for Λ\\Lambda. This form separates damping from the force that selects the equilibrium. Equation ( [10](https://arxiv.org/html/2609.26679v1#S2.E10 "In 2.2 Residuals drive task-aligned tangent growth ‣ 2 Post-fit residuals drive tangent feature learning ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning")) gives

|     |     |     |     |     |
| --- | --- | --- | --- | --- |
|  | Λ¨\\displaystyle\\ddot{\\Lambda} | =−a​σ˙−2​λWnorm​Λ˙\\displaystyle=-a\\dot{\\sigma}-2{\\lambda\_{W}^{\\mathrm{norm}}}\\dot{\\Lambda} |  | (40) |
|  |  | =a⁡(Λ+2​λWnorm)​σ+2​a​λWnorm​y−2​λWnorm​Λ˙,\\displaystyle=a(\\Lambda+2{\\lambda\_{W}^{\\mathrm{norm}}})\\sigma+2a{\\lambda\_{W}^{\\mathrm{norm}}}y-2{\\lambda\_{W}^{\\mathrm{norm}}}\\dot{\\Lambda}, |  | (41) |
|  | σ\\displaystyle\\sigma | =−Λ˙+2​λWnorm​Λa,\\displaystyle=-\\frac{\\dot{\\Lambda}+2{\\lambda\_{W}^{\\mathrm{norm}}}\\Lambda}{a}, |  | (42) |

hence

|     |     |     |     |
| --- | --- | --- | --- |
|  | Λ¨+(Λ+4​λWnorm)​Λ˙+2​λWnorm​Λ​(Λ+2​λWnorm)−2​a​λWnorm​y=0.\\ddot{\\Lambda}+(\\Lambda+4{\\lambda\_{W}^{\\mathrm{norm}}})\\dot{\\Lambda}+2{\\lambda\_{W}^{\\mathrm{norm}}}\\Lambda(\\Lambda+2{\\lambda\_{W}^{\\mathrm{norm}}})-2a{\\lambda\_{W}^{\\mathrm{norm}}}y=0. |  | (43) |

Equation ( [43](https://arxiv.org/html/2609.26679v1#A3.E43 "In C.2 Proof of Theorem ‣ Appendix C One-mode neural-tangent-hierarchy closure ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning")) describes one-dimensional damped motion with state-dependent coefficient Λ+4​λWnorm\\Lambda+4{\\lambda\_{W}^{\\mathrm{norm}}}. We collect the remaining terms into the cubic potential

|     |     |     |     |     |
| --- | --- | --- | --- | --- |
|  | V⁡(Λ)\\displaystyle V(\\Lambda) | =2​λWnorm3​Λ3+2​λWnorm2​Λ2−2​a​λWnorm​y​Λ,\\displaystyle=\\frac{2{\\lambda\_{W}^{\\mathrm{norm}}}}{3}\\Lambda^{3}+2{\\lambda\_{W}^{\\mathrm{norm}}}^{2}\\Lambda^{2}-2a{\\lambda\_{W}^{\\mathrm{norm}}}y\\Lambda, |  | (44) |
|  | ℋ\\displaystyle\\mathcal{H} | =12​Λ˙2+V⁡(Λ)−V⁡(Λ∗).\\displaystyle=\\frac{1}{2}\\dot{\\Lambda}^{2}+V(\\Lambda)-V(\\Lambda\_{\*}). |  | (45) |

On the branch Λ≥0\\Lambda\\geq 0, positive decay makes the potential strictly convex. The shifted energy ℋ\\mathcal{H} combines displacement from its unique minimum with the kinetic term, and

|     |     |     |     |     |
| --- | --- | --- | --- | --- |
|  | V′​(Λ)\\displaystyle V^{\\prime}(\\Lambda) | =2​λWnorm​Λ​(Λ+2​λWnorm)−2​a​λWnorm​y,\\displaystyle=2{\\lambda\_{W}^{\\mathrm{norm}}}\\Lambda(\\Lambda+2{\\lambda\_{W}^{\\mathrm{norm}}})-2a{\\lambda\_{W}^{\\mathrm{norm}}}y, |  | (46) |
|  | V′′​(Λ)\\displaystyle V^{\\prime\\prime}(\\Lambda) | =4​λWnorm​(Λ+λWnorm)>0,\\displaystyle=4{\\lambda\_{W}^{\\mathrm{norm}}}(\\Lambda+{\\lambda\_{W}^{\\mathrm{norm}}})>0, |  | (47) |
|  | ℋ˙\\displaystyle\\dot{\\mathcal{H}} | =−(Λ+4​λWnorm)​Λ˙2≤0.\\displaystyle=-(\\Lambda+4{\\lambda\_{W}^{\\mathrm{norm}}})\\dot{\\Lambda}^{2}\\leq 0. |  | (48) |

The stationary points of the first-order system coincide with the extrema of this potential. We solve the equilibrium conditions to obtain

|     |     |     |     |     |
| --- | --- | --- | --- | --- |
|  | σ∗\\displaystyle\\sigma\_{\*} | =−2​λWnorma​Λ∗,\\displaystyle=-\\frac{2{\\lambda\_{W}^{\\mathrm{norm}}}}{a}\\Lambda\_{\*}, |  | (49) |
|  | Λ∗​(Λ∗+2​λWnorm)\\displaystyle\\Lambda\_{\*}(\\Lambda\_{\*}+2{\\lambda\_{W}^{\\mathrm{norm}}}) | =a​y,\\displaystyle=ay, |  | (50) |
|  | Λ±\\displaystyle\\Lambda\_{\\pm} | =−λWnorm±λWnorm2+a​y.\\displaystyle=-{\\lambda\_{W}^{\\mathrm{norm}}}\\pm\\sqrt{{\\lambda\_{W}^{\\mathrm{norm}}}^{2}+ay}. |  | (51) |

For y=0y=0, the only nonnegative root is zero. For y>0y>0, Λ+>0\\Lambda\_{+}>0 and Λ−<0\\Lambda\_{-}<0, giving one positive target-bearing equilibrium. The Lyapunov calculation establishes dissipative motion toward the potential minimum; we use the Jacobian to verify the local asymptotic stability in Theorem [2](https://arxiv.org/html/2609.26679v1#Thmtheorem2 "Theorem 2 (Spectral selection in the reduced tangent-mode system). ‣ 2.2 Residuals drive task-aligned tangent growth ‣ 2 Post-fit residuals drive tangent feature learning ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning"). At the positive equilibrium,

|     |     |     |     |
| --- | --- | --- | --- |
|  | A=(−(Λ∗+2​λWnorm)−σ∗−a−2​λWnorm),A=\\begin{pmatrix}-(\\Lambda\_{\*}+2{\\lambda\_{W}^{\\mathrm{norm}}})&-\\sigma\_{\*}\\\<br>-a&-2{\\lambda\_{W}^{\\mathrm{norm}}}\\end{pmatrix}, |  | (52) |

with

|     |     |     |     |     |
| --- | --- | --- | --- | --- |
|  | tr⁡A\\displaystyle\\operatorname{tr}A | =−(Λ∗+4​λWnorm)<0,\\displaystyle=-(\\Lambda\_{\*}+4{\\lambda\_{W}^{\\mathrm{norm}}})<0, |  | (53) |
|  | detA\\displaystyle\\det A | =2​λWnorm​(Λ∗+2​λWnorm)−a​σ∗\\displaystyle=2{\\lambda\_{W}^{\\mathrm{norm}}}(\\Lambda\_{\*}+2{\\lambda\_{W}^{\\mathrm{norm}}})-a\\sigma\_{\*} |  | (54) |
|  |  | =4​λWnorm​(Λ∗+λWnorm)>0.\\displaystyle=4{\\lambda\_{W}^{\\mathrm{norm}}}(\\Lambda\_{\*}+{\\lambda\_{W}^{\\mathrm{norm}}})>0. |  | (55) |

The negative trace and positive determinant place both eigenvalues in the open left half-plane, so the positive equilibrium is locally asymptotically stable on the nonnegative branch. □\\square

Figure 5: Reduced potential and adiabatic trajectories. (a) With illustrative values λWnorm=0.15\\lambda\_{W}^{\\mathrm{norm}}=0.15 and a=1a=1, the task-aligned mode (y=1y=1) has its potential minimum at positive Λ\\Lambda, whereas the unsupported mode (y=0y=0) is minimized at Λ=0\\Lambda=0. (b) Starting from the same initial strength, the two adiabatic trajectories move toward their respective minima. These dimensionless values illustrate the reduced dynamics and are not fitted to an empirical run.

## Appendix D Adiabatic reduction and finite-training boundary

### D.1 From smooth mode growth to an accuracy transition

The reduced dynamics produce a smooth trajectory Λ⁡(t)\\Lambda(t), whereas held-out accuracy changes only when individual predictions change class. For held-out example μ\\mu with correct class cμc\_{\\mu}, define its correct-class margin along the training trajectory by

|     |     |     |     |
| --- | --- | --- | --- |
|  | mμ​(t)=fcμ​(𝐱μ,t)−maxc≠cμ⁡fc​(𝐱μ,t).m\_{\\mu}(t)=f\_{c\_{\\mu}}(\\mathbf{x}\_{\\mu};t)-\\max\_{c\\neq c\_{\\mu}}f\_{c}(\\mathbf{x}\_{\\mu};t). |  | (56) |

When Λ⁡(t)\\Lambda(t) is monotone over the transition, the trajectory can be reparameterized by Λ\\Lambda, so that m~μ​(Λ):=mμ​(t⁡(Λ))\\widetilde{m}\_{\\mu}(\\Lambda):=m\_{\\mu}(t(\\Lambda)). If each relevant m~μ​(Λ)\\widetilde{m}\_{\\mu}(\\Lambda) changes monotonically through its classification transition, then each example has a mode strength at which its predicted class changes. Denote this threshold by

|     |     |     |     |
| --- | --- | --- | --- |
|  | m~μ​(Λμ,g)=0.\\widetilde{m}\_{\\mu}(\\Lambda\_{\\mu,\\mathrm{g}})=0. |  | (57) |

In this case, held-out accuracy at a given Λ\\Lambda is the empirical cumulative distribution of the example-specific thresholds,

|     |     |     |     |
| --- | --- | --- | --- |
|  | Acc(Λ)=1M∑μ=1M𝟏{Λ>Λμ,g}=F^g(Λ).\\operatorname{Acc}(\\Lambda)=\\frac{1}{M}\\sum\_{\\mu=1}^{M}\\mathbf{1}\\{\\Lambda>\\Lambda\_{\\mu,\\mathrm{g}}\\}=\\widehat{F}\_{\\mathrm{g}}(\\Lambda). |  | (58) |

A narrow distribution of Λμ,g\\Lambda\_{\\mu,\\mathrm{g}} therefore produces a sharp accuracy rise even when Λ⁡(t)\\Lambda(t) changes smoothly. No discontinuity in the optimization dynamics is required. As an illustration, a logistic threshold distribution with location μg\\mu\_{g} and scale bgb\_{g} gives

|     |     |     |     |
| --- | --- | --- | --- |
|  | Acc⁡(Λ)≈11+exp\[−(Λ−μg)/bg\].\\operatorname{Acc}(\\Lambda)\\approx\\frac{1}{1+\\exp\[-(\\Lambda-\\mu\_{g})/b\_{g}\]}. |  | (59) |

This logistic form is only an example of the readout, the one-mode dynamics do not depend on it.

### D.2 Adiabatic slow equation

The residual relaxes at rate Λ+2​λWnorm\\Lambda+2{\\lambda\_{W}^{\\mathrm{norm}}}, while the NTK mode moves on the slower decay-controlled scale once the residual is close to its ridge value. Under this separation of timescales, we set σ˙≃0\\dot{\\sigma}\\simeq 0 at the current Λ\\Lambda and define the resulting adiabatic residual σad​(Λ)\\sigma\_{\\rm ad}(\\Lambda). Equation ( [10](https://arxiv.org/html/2609.26679v1#S2.E10 "In 2.2 Residuals drive task-aligned tangent growth ‣ 2 Post-fit residuals drive tangent feature learning ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning")) gives

|     |     |     |     |     |
| --- | --- | --- | --- | --- |
|  | 0\\displaystyle 0 | =−(Λ+2​λWnorm)​σ−2​λWnorm​y,\\displaystyle=-(\\Lambda+2{\\lambda\_{W}^{\\mathrm{norm}}})\\sigma-2{\\lambda\_{W}^{\\mathrm{norm}}}y, |  | (60) |
|  | σad​(Λ)\\displaystyle\\sigma\_{\\rm ad}(\\Lambda) | =−2​λWnorm​yΛ+2​λWnorm,\\displaystyle=-\\frac{2{\\lambda\_{W}^{\\mathrm{norm}}}y}{\\Lambda+2{\\lambda\_{W}^{\\mathrm{norm}}}}, |  | (61) |
|  | Λ˙\\displaystyle\\dot{\\Lambda} | =−a​σad​(Λ)−2​λWnorm​Λ\\displaystyle=-a\\sigma\_{\\rm ad}(\\Lambda)-2{\\lambda\_{W}^{\\mathrm{norm}}}\\Lambda |  | (62) |
|  |  | =2​λWnorm​a​y−Λ2−2​λWnorm​ΛΛ+2​λWnorm.\\displaystyle=2{\\lambda\_{W}^{\\mathrm{norm}}}\\frac{ay-\\Lambda^{2}-2{\\lambda\_{W}^{\\mathrm{norm}}}\\Lambda}{\\Lambda+2{\\lambda\_{W}^{\\mathrm{norm}}}}. |  | (63) |

The numerator in Eq. ( [63](https://arxiv.org/html/2609.26679v1#A4.E63 "In D.2 Adiabatic slow equation ‣ Appendix D Adiabatic reduction and finite-training boundary ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning")) determines the sign of the mode growth. It is positive below the stable fixed point and vanishes at that fixed point, so the upward motion slows as Λ\\Lambda approaches equilibrium. Separating variables gives

|     |     |     |     |
| --- | --- | --- | --- |
|  | d​t=12​λWnorm​Λ+2​λWnorma​y−Λ2−2​λWnorm​Λ​d​Λ.dt=\\frac{1}{2{\\lambda\_{W}^{\\mathrm{norm}}}}\\frac{\\Lambda+2{\\lambda\_{W}^{\\mathrm{norm}}}}{ay-\\Lambda^{2}-2{\\lambda\_{W}^{\\mathrm{norm}}}\\Lambda}\\,d\\Lambda. |  | (64) |

Integrating from Λ0\\Lambda\_{0} to Λg\\Lambda\_{\\mathrm{g}} gives the normalized gradient-flow time required to reach the held-out threshold. After TT optimizer updates,

|     |     |     |     |     |
| --- | --- | --- | --- | --- |
|  | t\\displaystyle t | ≃𝒩​η​T,\\displaystyle\\simeq\\mathcal{N}\\eta T, |  | (65) |
|  | 𝒩​λWnorm\\displaystyle\\mathcal{N}{\\lambda\_{W}^{\\mathrm{norm}}} | =λW.\\displaystyle=\\lambda\_{W}. |  | (66) |

Equating the available normalized time to the crossing time gives Eq. ( [15](https://arxiv.org/html/2609.26679v1#S2.E15 "In Corollary 1 (Finite-training grokking boundary). ‣ 2.3 Grokking boundaries in the (𝜂,𝜆_𝑊) plane ‣ 2 Post-fit residuals drive tangent feature learning ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning")). The conversion from updates contributes 𝒩\\mathcal{N}, while 𝒩​λWnorm=λW\\mathcal{N}{\\lambda\_{W}^{\\mathrm{norm}}}=\\lambda\_{W}, leaving optimizer decay in the prefactor and normalized decay inside the integrand.

### D.3 Closed form, critical slowing, and the fitting cutoff

We factor the denominator of the crossing-time integrand at the two fixed points of the slow equation. Define

|     |     |     |     |
| --- | --- | --- | --- |
|  | q:=a​y,Λ±:=−λWnorm±λWnorm2+q.q:=ay,\\qquad\\Lambda\_{\\pm}:=-{\\lambda\_{W}^{\\mathrm{norm}}}\\pm\\sqrt{{\\lambda\_{W}^{\\mathrm{norm}}}^{2}+q}. |  | (67) |

Then

|     |     |     |     |     |
| --- | --- | --- | --- | --- |
|  | q−Λ2−2​λWnorm​Λ\\displaystyle q-\\Lambda^{2}-2{\\lambda\_{W}^{\\mathrm{norm}}}\\Lambda | =−(Λ−Λ+)​(Λ−Λ−),\\displaystyle=-(\\Lambda-\\Lambda\_{+})(\\Lambda-\\Lambda\_{-}), |  | (68) |
|  | C−\\displaystyle C\_{-} | :=2​λWnorm+Λ+Λ+−Λ−,\\displaystyle:=\\frac{2{\\lambda\_{W}^{\\mathrm{norm}}}+\\Lambda\_{+}}{\\Lambda\_{+}-\\Lambda\_{-}}, |  | (69) |
|  | C+\\displaystyle C\_{+} | :=2​λWnorm+Λ−Λ−−Λ+.\\displaystyle:=\\frac{2{\\lambda\_{W}^{\\mathrm{norm}}}+\\Lambda\_{-}}{\\Lambda\_{-}-\\Lambda\_{+}}. |  | (70) |

The integrand becomes

|     |     |     |     |
| --- | --- | --- | --- |
|  | Λ+2​λWnormq−Λ2−2​λWnorm​Λ=−C−Λ−Λ+−C+Λ−Λ−.\\frac{\\Lambda+2{\\lambda\_{W}^{\\mathrm{norm}}}}{q-\\Lambda^{2}-2{\\lambda\_{W}^{\\mathrm{norm}}}\\Lambda}=-\\frac{C\_{-}}{\\Lambda-\\Lambda\_{+}}-\\frac{C\_{+}}{\\Lambda-\\Lambda\_{-}}. |  | (71) |

Integrating between Λ0\\Lambda\_{0} and Λg\\Lambda\_{\\mathrm{g}} gives

|     |     |     |     |     |
| --- | --- | --- | --- | --- |
|  | η∗(λW)=12​λW​T\[\\displaystyle\\eta\_{\*}(\\lambda\_{W})=\\frac{1}{2\\lambda\_{W}T}\\Bigg\[ | C−log\|Λ0−Λ+Λg−Λ+\|+C+log\|Λ0−Λ−Λg−Λ−\|\].\\displaystyle C\_{-}\\log\\left\|\\frac{\\Lambda\_{0}-\\Lambda\_{+}}{\\Lambda\_{\\mathrm{g}}-\\Lambda\_{+}}\\right\|+C\_{+}\\log\\left\|\\frac{\\Lambda\_{0}-\\Lambda\_{-}}{\\Lambda\_{\\mathrm{g}}-\\Lambda\_{-}}\\right\|\\Bigg\]. |  | (72) |

As the requested threshold approaches the stable fixed point Λ+\\Lambda\_{+}, the pole at Λ+\\Lambda\_{+} reaches the upper integration limit while the Λ−\\Lambda\_{-} contribution remains finite. Hence, for Λg↑Λ+\\Lambda\_{\\mathrm{g}}\\uparrow\\Lambda\_{+},

|     |     |     |     |     |
| --- | --- | --- | --- | --- |
|  | η∗​(λW)\\displaystyle\\eta\_{\*}(\\lambda\_{W}) | ∼C−2​λW​T​log⁡1Λ+−Λg,\\displaystyle\\sim\\frac{C\_{-}}{2\\lambda\_{W}T}\\log\\frac{1}{\\Lambda\_{+}-\\Lambda\_{\\mathrm{g}}}, |  | (73) |
|  | Λ+−Λg\\displaystyle\\Lambda\_{+}-\\Lambda\_{\\mathrm{g}} | =Θ⁡(λWmode−λW).\\displaystyle=\\Theta\\!\\left(\\lambda\_{W}^{\\mathrm{mode}}-\\lambda\_{W}\\right)\\,. |  | (74) |

The logarithm arises because the growth rate of Λ\\Lambda vanishes at the reachability boundary. Away from this cutoff, the logarithm remains finite and the explicit prefactor gives the leading 1/(λW​T)1/(\\lambda\_{W}T) dependence.

The training-fit cutoff in Eq. ( [17](https://arxiv.org/html/2609.26679v1#S2.E17 "In 2.3 Grokking boundaries in the (𝜂,𝜆_𝑊) plane ‣ 2 Post-fit residuals drive tangent feature learning ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning")) follows from the same adiabatic gain. For a representative training mode, the expressed fraction of its target component is

|     |     |     |     |
| --- | --- | --- | --- |
|  | g⁡(Λfit)=ΛfitΛfit+2​λWnorm.g(\\Lambda\_{\\rm fit})=\\frac{\\Lambda\_{\\rm fit}}{\\Lambda\_{\\rm fit}+2{\\lambda\_{W}^{\\mathrm{norm}}}}. |  | (75) |

Requiring g⁡(Λfit)≥gming(\\Lambda\_{\\rm fit})\\geq g\_{\\min} gives

|     |     |     |     |     |
| --- | --- | --- | --- | --- |
|  | ΛfitΛfit+2​λWnorm\\displaystyle\\frac{\\Lambda\_{\\rm fit}}{\\Lambda\_{\\rm fit}+2{\\lambda\_{W}^{\\mathrm{norm}}}} | ≥gmin,\\displaystyle\\geq g\_{\\min}, |  | (76) |
|  | λWnorm\\displaystyle{\\lambda\_{W}^{\\mathrm{norm}}} | ≤Λfit​(1−gmin)2​gmin,\\displaystyle\\leq\\frac{\\Lambda\_{\\rm fit}(1-g\_{\\min})}{2g\_{\\min}}, |  | (77) |
|  | λW\\displaystyle\\lambda\_{W} | ≤𝒩​Λfit​(1−gmin)2​gmin,\\displaystyle\\leq\\mathcal{N}\\frac{\\Lambda\_{\\rm fit}(1-g\_{\\min})}{2g\_{\\min}}, |  | (78) |

which is Eq. ( [17](https://arxiv.org/html/2609.26679v1#S2.E17 "In 2.3 Grokking boundaries in the (𝜂,𝜆_𝑊) plane ‣ 2 Post-fit residuals drive tangent feature learning ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning")). The approximately vertical boundary used in the phase diagram further assumes that the representative training-mode strength Λfit\\Lambda\_{\\rm fit} varies weakly with η\\eta over the range of interest.

### D.4 Discrete-time stability of the frozen-kernel dynamics

Our continuous-time reduction does not describe the large-step edge of the optimizer plane. We hold the kernel fixed to isolate the Euler stability condition for the fast residual dynamics and set

|     |     |     |
| --- | --- | --- |
|  | h:=𝒩​η.h:=\\mathcal{N}\\eta. |  |

For fixed KK,

|     |     |     |     |     |
| --- | --- | --- | --- | --- |
|  | 𝝈s+1\\displaystyle\\boldsymbol{\\sigma}\_{s+1} | =𝝈s−h⁡\[(K+2​λWnorm​I)​𝝈s+2​λWnorm​𝐲\],\\displaystyle=\\boldsymbol{\\sigma}\_{s}-h\\left\[(K+2{\\lambda\_{W}^{\\mathrm{norm}}}I)\\boldsymbol{\\sigma}\_{s}+2{\\lambda\_{W}^{\\mathrm{norm}}}\\mathbf{y}\\right\], |  | (79) |
|  | 𝝈∗\\displaystyle\\boldsymbol{\\sigma}\_{\*} | =−2​λWnorm​(K+2​λWnorm​I)−1​𝐲,\\displaystyle=-2{\\lambda\_{W}^{\\mathrm{norm}}}(K+2{\\lambda\_{W}^{\\mathrm{norm}}}I)^{-1}\\mathbf{y}, |  | (80) |
|  | 𝜹s\\displaystyle\\boldsymbol{\\delta}\_{s} | :=𝝈s−𝝈∗,\\displaystyle:=\\boldsymbol{\\sigma}\_{s}-\\boldsymbol{\\sigma}\_{\*}, |  | (81) |
|  | 𝜹s+1\\displaystyle\\boldsymbol{\\delta}\_{s+1} | =\[I−h⁡(K+2​λWnorm​I)\]​𝜹s.\\displaystyle=\\left\[I-h(K+2{\\lambda\_{W}^{\\mathrm{norm}}}I)\\right\]\\boldsymbol{\\delta}\_{s}. |  | (82) |

After we subtract the fixed point, each kernel eigendirection evolves independently. For K​𝐮j=Λj​𝐮jK\\mathbf{u}\_{j}=\\Lambda\_{j}\\mathbf{u}\_{j},

|     |     |     |     |
| --- | --- | --- | --- |
|  | δj,s+1=\[1−h⁡(Λj+2​λWnorm)\]​δj,s.\\delta\_{j,s+1}=\\left\[1-h(\\Lambda\_{j}+2{\\lambda\_{W}^{\\mathrm{norm}}})\\right\]\\delta\_{j,s}. |  | (83) |

Therefore

|     |     |     |     |     |
| --- | --- | --- | --- | --- |
|  | \|1−h⁡(Λj+2​λWnorm)\|\\displaystyle\\left\|1-h(\\Lambda\_{j}+2{\\lambda\_{W}^{\\mathrm{norm}}})\\right\| | <1,\\displaystyle<1, |  | (84) |
|  | 𝒩​η\\displaystyle\\mathcal{N}\\eta | <2Λmax​(K)+2​λWnorm,\\displaystyle<\\frac{2}{\\Lambda\_{\\max}(K)+2{\\lambda\_{W}^{\\mathrm{norm}}}}, |  | (85) |
|  | η\\displaystyle\\eta | <2𝒩​Λmax​(K)+2​λW,\\displaystyle<\\frac{2}{\\mathcal{N}\\Lambda\_{\\max}(K)+2\\lambda\_{W}}, |  | (86) |

which recovers Eq. ( [18](https://arxiv.org/html/2609.26679v1#S2.E18 "In 2.3 Grokking boundaries in the (𝜂,𝜆_𝑊) plane ‣ 2 Post-fit residuals drive tangent feature learning ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning")). The most restrictive eigenvalue is Λmax​(K)\\Lambda\_{\\max}(K). Feature learning changes this spectrum, so we use the bound only as a local stability condition for the frozen-kernel approximation, not as a global guarantee for nonlinear training.

## Appendix E Jacobian-mediated feature dynamics beyond homogeneity

For any differentiable model trained with squared loss and coupled L2L\_{2} decay, the chain rule separates the residual-dependent contribution to tangent-kernel evolution from the contribution of radial parameter decay. Homogeneity is only needed to convert the latter into fixed multiples of the output and NTK. Without homogeneity, the residual-dependent term remains exact, while the radial contractions become architecture dependent.

### E.1 Exact differentiable-model identities

###### Proposition 1(Exact Jacobian-mediated dynamics for differentiable models).

Let

|     |     |     |     |
| --- | --- | --- | --- |
|  | ℒ~​(𝜽)=12​‖𝐟𝜽−𝐲‖2+λWnorm2​‖𝜽‖2,\\widetilde{\\mathcal{L}}(\\boldsymbol{\\theta})=\\frac{1}{2}\\left\\lVert\\mathbf{f}\_{\\boldsymbol{\\theta}}-\\mathbf{y}\\right\\rVert^{2}+\\frac{{\\lambda\_{W}^{\\mathrm{norm}}}}{2}\\left\\lVert\\boldsymbol{\\theta}\\right\\rVert^{2}, |  | (87) |

with 𝛔=𝐟−𝐲\\boldsymbol{\\sigma}=\\mathbf{f}-\\mathbf{y}, J=∇𝛉𝐟J=\\nabla\_{\\boldsymbol{\\theta}}\\mathbf{f}, and K=J​J⊤K=JJ^{\\top}. Then

|     |     |     |     |     |
| --- | --- | --- | --- | --- |
|  | 𝝈˙\\displaystyle\\dot{\\boldsymbol{\\sigma}} | =−K​𝝈−λWnorm​J​𝜽,\\displaystyle=-K\\boldsymbol{\\sigma}-{\\lambda\_{W}^{\\mathrm{norm}}}J\\boldsymbol{\\theta}, |  | (88) |
|  | K˙i​j\\displaystyle\\dot{K}\_{ij} | =−∑kK(2)i​j​kσk−λWnormRi​j,\\displaystyle=-\\sum\_{k}K^{(2)}\_{ijk}\\sigma\_{k}-{\\lambda\_{W}^{\\mathrm{norm}}}R\_{ij}, |  | (89) |

where

|     |     |     |     |
| --- | --- | --- | --- |
|  | Ki​j​k(2)=⟨∇𝜽Ki​j,∇𝜽fk⟩,Ri​j=⟨∇𝜽Ki​j,𝜽⟩.K^{(2)}\_{ijk}=\\left\\langle\\nabla\_{\\boldsymbol{\\theta}}K\_{ij},\\nabla\_{\\boldsymbol{\\theta}}f\_{k}\\right\\rangle,\\qquad R\_{ij}=\\left\\langle\\nabla\_{\\boldsymbol{\\theta}}K\_{ij},\\boldsymbol{\\theta}\\right\\rangle. |  | (90) |

Along the same flow, the loss is monotone because

|     |     |     |     |
| --- | --- | --- | --- |
|  | dd​t​ℒ~​(𝜽⁡(t))=−‖J⊤​𝝈+λWnorm​𝜽‖2≤0.\\frac{d}{dt}\\widetilde{\\mathcal{L}}(\\boldsymbol{\\theta}(t))=-\\left\\lVert J^{\\top}\\boldsymbol{\\sigma}+{\\lambda\_{W}^{\\mathrm{norm}}}\\boldsymbol{\\theta}\\right\\rVert^{2}\\leq 0. |  | (91) |

##### Proof.

We differentiate the model output along parameter gradient flow to obtain the residual identity. Applying the same derivative to each entry of KK produces one contraction with the data gradient and one with the radial weight-decay direction.

|     |     |     |     |     |
| --- | --- | --- | --- | --- |
|  | 𝜽˙\\displaystyle\\dot{\\boldsymbol{\\theta}} | =−J⊤​𝝈−λWnorm​𝜽,\\displaystyle=-J^{\\top}\\boldsymbol{\\sigma}-{\\lambda\_{W}^{\\mathrm{norm}}}\\boldsymbol{\\theta}, |  | (92) |
|  | 𝝈˙\\displaystyle\\dot{\\boldsymbol{\\sigma}} | =J​𝜽˙=−K​𝝈−λWnorm​J​𝜽,\\displaystyle=J\\dot{\\boldsymbol{\\theta}}=-K\\boldsymbol{\\sigma}-{\\lambda\_{W}^{\\mathrm{norm}}}J\\boldsymbol{\\theta}, |  | (93) |
|  | K˙i​j\\displaystyle\\dot{K}\_{ij} | =⟨∇𝜽Ki​j,𝜽˙⟩\\displaystyle=\\left\\langle\\nabla\_{\\boldsymbol{\\theta}}K\_{ij},\\dot{\\boldsymbol{\\theta}}\\right\\rangle |  | (94) |
|  |  | =−∑k⟨∇𝜽Ki​j,∇𝜽fk⟩σk−λWnorm⟨∇𝜽Ki​j,𝜽⟩,\\displaystyle=-\\sum\_{k}\\left\\langle\\nabla\_{\\boldsymbol{\\theta}}K\_{ij},\\nabla\_{\\boldsymbol{\\theta}}f\_{k}\\right\\rangle\\sigma\_{k}-{\\lambda\_{W}^{\\mathrm{norm}}}\\left\\langle\\nabla\_{\\boldsymbol{\\theta}}K\_{ij},\\boldsymbol{\\theta}\\right\\rangle, |  | (95) |
|  | dd​t​ℒ~\\displaystyle\\frac{d}{dt}\\widetilde{\\mathcal{L}} | =⟨∇𝜽ℒ~,𝜽˙⟩=−‖∇𝜽ℒ~‖2.\\displaystyle=\\left\\langle\\nabla\_{\\boldsymbol{\\theta}}\\widetilde{\\mathcal{L}},\\dot{\\boldsymbol{\\theta}}\\right\\rangle=-\\left\\lVert\\nabla\_{\\boldsymbol{\\theta}}\\widetilde{\\mathcal{L}}\\right\\rVert^{2}. |  | (96) |

□\\square

Equation ( [89](https://arxiv.org/html/2609.26679v1#A5.E89 "In Proposition 1 (Exact Jacobian-mediated dynamics for differentiable models). ‣ E.1 Exact differentiable-model identities ‣ Appendix E Jacobian-mediated feature dynamics beyond homogeneity ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning")) isolates the extension needed beyond homogeneous networks. Whenever the K(2)K^{(2)} contraction is nonzero, the residual that controls prediction error also changes the tangent kernel. The contraction may vanish at individual states; homogeneity is needed only to replace the remaining radial term by fixed coefficients.

### E.2 Local projected dynamics

For a task direction 𝐮\\mathbf{u}, we suppose over a training interval that

|     |     |     |     |
| --- | --- | --- | --- |
|  | K​𝐮≈Λ​𝐮,𝝈≈σ​𝐮,𝐮⊤​J​𝜽≈β⁡(σ+y),𝐮⊤​R​𝐮≈ρ​Λ,𝐮⊤​(K(2)​𝝈)​𝐮≈a​σ,K\\mathbf{u}\\approx\\Lambda\\mathbf{u},\\quad\\boldsymbol{\\sigma}\\approx\\sigma\\mathbf{u},\\quad\\mathbf{u}^{\\top}J\\boldsymbol{\\theta}\\approx\\beta(\\sigma+y),\\quad\\mathbf{u}^{\\top}R\\mathbf{u}\\approx\\rho\\Lambda,\\quad\\mathbf{u}^{\\top}(K^{(2)}\\boldsymbol{\\sigma})\\mathbf{u}\\approx a\\sigma, |  | (97) |

with slowly varying β,ρ,a\\beta,\\rho,a. The coefficients β\\beta and ρ\\rho summarize radial parameter decay after projection onto the local task direction, while aa retains its residual-to-kernel role. Projecting gives

|     |     |     |     |
| --- | --- | --- | --- |
|  | σ˙≈−(Λ+β​λWnorm)​σ−β​λWnorm​y,Λ˙≈−a​σ−ρ​λWnorm​Λ.\\dot{\\sigma}\\approx-(\\Lambda+\\beta{\\lambda\_{W}^{\\mathrm{norm}}})\\sigma-\\beta{\\lambda\_{W}^{\\mathrm{norm}}}y,\\qquad\\dot{\\Lambda}\\approx-a\\sigma-\\rho{\\lambda\_{W}^{\\mathrm{norm}}}\\Lambda. |  | (98) |

For a two-homogeneous model, Euler identities set β=ρ=2\\beta=\\rho=2 and recover Eq. ( [10](https://arxiv.org/html/2609.26679v1#S2.E10 "In 2.2 Residuals drive task-aligned tangent growth ‣ 2 Post-fit residuals drive tangent feature learning ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning")). In a non-homogeneous model, we must measure these local contractions; they need not equal the homogeneous coefficients. If they remain order one and vary slowly through the transition, both decay terms retain the factor λWnorm{\\lambda\_{W}^{\\mathrm{norm}}}, leaving 1/λWnorm1/{\\lambda\_{W}^{\\mathrm{norm}}} as the natural normalized-time scale and (η​λW)−1(\\eta\\lambda\_{W})^{-1} as the corresponding update scale.

For the Transformer in Appendix [G.3](https://arxiv.org/html/2609.26679v1#A7.SS3 "G.3 Transformer phase sweep on addition modulo 23 ‣ Appendix G Experimental configurations and analysis conventions ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning"), we let 𝐟𝜽\\mathbf{f}\_{\\boldsymbol{\\theta}} denote the softmax probabilities entering the mean-squared-error loss. Proposition [1](https://arxiv.org/html/2609.26679v1#Thmproposition1 "Proposition 1 (Exact Jacobian-mediated dynamics for differentiable models). ‣ E.1 Exact differentiable-model identities ‣ Appendix E Jacobian-mediated feature dynamics beyond homogeneity ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning") then applies despite attention, LayerNorm, residual paths, and biases.

## Appendix F Directional NTK-alignment intervention

### F.1 Earlier task alignment shortens the predicted delay

Starting closer to the held-out threshold removes a positive part of the crossing-time integral without invoking the global kinetic calibration. Equation ( [15](https://arxiv.org/html/2609.26679v1#S2.E15 "In Corollary 1 (Finite-training grokking boundary). ‣ 2.3 Grokking boundaries in the (𝜂,𝜆_𝑊) plane ‣ 2 Post-fit residuals drive tangent feature learning ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning")) gives

|     |     |     |     |
| --- | --- | --- | --- |
|  | sg=12​η​λW​∫Λ0ΛgΛ+2​λWnorma​y−Λ2−2​λWnorm​Λ​𝑑Λ,s\_{\\mathrm{g}}=\\frac{1}{2\\eta\\lambda\_{W}}\\int\_{\\Lambda\_{0}}^{\\Lambda\_{\\mathrm{g}}}\\frac{\\Lambda+2{\\lambda\_{W}^{\\mathrm{norm}}}}{ay-\\Lambda^{2}-2{\\lambda\_{W}^{\\mathrm{norm}}}\\Lambda}\\,d\\Lambda, |  | (99) |

and differentiation with respect to the initial strength gives

|     |     |     |     |
| --- | --- | --- | --- |
|  | ∂sg∂Λ0=−12​η​λW​Λ0+2​λWnorma​y−Λ02−2​λWnorm​Λ0<0\\frac{\\partial s\_{\\mathrm{g}}}{\\partial\\Lambda\_{0}}=-\\frac{1}{2\\eta\\lambda\_{W}}\\frac{\\Lambda\_{0}+2{\\lambda\_{W}^{\\mathrm{norm}}}}{ay-\\Lambda\_{0}^{2}-2{\\lambda\_{W}^{\\mathrm{norm}}}\\Lambda\_{0}}<0 |  | (100) |

throughout the reachable branch, where the denominator is the positive spectral drive. The sign predicts that improving early alignment with a reachable task direction should advance the later held-out transition even if the auxiliary intervention is removed before generalization begins.

### F.2 A temporary NTK intervention advances held-out generalization

We change early task-indexed tangent geometry while keeping the later objective identical, which tests the sign in Eq. ( [100](https://arxiv.org/html/2609.26679v1#A6.E100 "In F.1 Earlier task alignment shortens the predicted delay ‣ Appendix F Directional NTK-alignment intervention ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning")). We use addition modulo 9797 on the 4,7534{,}753 unordered input pairs. We train a bias-free →→→→97194\\!\\to\\!128\\!\\to\\!128\\!\\to\\!128\\!\\to\\!97 ReLU teacher with He-normal initialization. Its 50/5050/50 split uses NumPy seed 00 and contains 2,3762{,}376 training and 2,3772{,}377 held-out examples. We run 30,00030{,}000 full-batch SGD updates without momentum, using learning rate 180180, coupled weight decay 10−510^{-5}, ordinary PyTorch MSE between logits and one-hot targets averaged over samples and outputs, and model seed 00.

For examples 𝐱μ,𝐱ν\\mathbf{x}\_{\\mu},\\mathbf{x}\_{\\nu} with labels cμ,cνc\_{\\mu},c\_{\\nu}, we project the empirical NTK through the corresponding true-label logits,

|     |     |     |     |
| --- | --- | --- | --- |
|  | Kμ​νTL=⟨∇𝜽fcμ​(𝐱μ),∇𝜽fcν​(𝐱ν)⟩.K^{\\rm TL}\_{\\mu\\nu}=\\left\\langle\\nabla\_{\\boldsymbol{\\theta}}f\_{c\_{\\mu}}(\\mathbf{x}\_{\\mu}),\\nabla\_{\\boldsymbol{\\theta}}f\_{c\_{\\nu}}(\\mathbf{x}\_{\\nu})\\right\\rangle. |  | (101) |

The intervention uses a normalization order different from the diagnostic in Appendix [G.1](https://arxiv.org/html/2609.26679v1#A7.SS1 "G.1 Spectral diagnostics on addition modulo 97 ‣ Appendix G Experimental configurations and analysis conventions ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning"). We first cosine-normalize the sample kernel,

|     |     |     |     |
| --- | --- | --- | --- |
|  | K^μ​νTL=Kμ​νTLKμ​μTL​Kν​νTL,\\widehat{K}^{\\rm TL}\_{\\mu\\nu}=\\frac{K^{\\rm TL}\_{\\mu\\nu}}{\\sqrt{K^{\\rm TL}\_{\\mu\\mu}K^{\\rm TL}\_{\\nu\\nu}}}, |  | (102) |

and then average between label groups,

|     |     |     |     |
| --- | --- | --- | --- |
|  | K¯c​c′=1nc​nc′∑μ:cμ=c∑ν:cν=c′K^TLμ​ν.\\bar{K}\_{cc^{\\prime}}=\\frac{1}{n\_{c}n\_{c^{\\prime}}}\\sum\_{\\mu:c\_{\\mu}=c}\\sum\_{\\nu:c\_{\\nu}=c^{\\prime}}\\widehat{K}^{\\rm TL}\_{\\mu\\nu}. |  | (103) |

We compute the fixed symmetric 97×9797\\times 97 target matrix from the teacher’s training split and the differentiable student matrix from each student’s training split.

We train 4040 paired bias-free →→97194\\!\\to\\!256\\!\\to\\!97 ReLU students with He-normal initialization. Pair q∈{0,…,39}q\\in\\{0,\\ldots,39\\} uses model seed qq and an independently generated 50/5050/50 split with NumPy seed 10,000+q10{,}000+q; every split has 2,3762{,}376 training and 2,3772{,}377 held-out examples, and both members of the pair share the same initialization and data. Every student uses 30,00030{,}000 full-batch SGD updates without momentum, learning rate 3535, coupled weight decay 3×10−63\\times 10^{-6}, and ordinary PyTorch MSE between logits and one-hot targets averaged over samples and outputs. During updates 00 through 2,9992{,}999, the intervention member minimizes

|     |     |     |     |
| --- | --- | --- | --- |
|  | ℒint=ℒstudent+λNTK​‖K¯𝜽−K¯teacher‖F,λNTK=0.10.\\mathcal{L}\_{\\rm int}=\\mathcal{L}\_{\\rm student}+\\lambda\_{\\rm NTK}\\left\\\|\\bar{K}\_{\\boldsymbol{\\theta}}-\\bar{K}\_{\\rm teacher}\\right\\\|\_{F},\\qquad\\lambda\_{\\rm NTK}=0.10. |  | (104) |

We remove the auxiliary term at update 3,0003{,}000; the baseline never receives it. We record training and held-out accuracy and MSE every 100100 updates.

Figure 6: Temporary early NTK alignment advances held-out generalization. (a) Mean held-out accuracy for the paired baseline and intervention students. The shaded vertical region marks the first 3,0003{,}000 updates, when the NTK penalty is active; bands show one sample standard deviation across 4040 paired seeds. (b) Mean held-out MSE for the same runs. The mean first 50%50\\% held-out crossing moves from 20,20020{,}200 updates for the baseline to 13,527.513{,}527.5 for the intervention, a ratio of 1.4931.493.

The paired mean speedup is 1.4981.498, and the median paired speedup is 1.4891.489. The intervention students also maintain lower held-out MSE after the penalty has been removed. Because the two members of each pair share their data and initialization and use the same objective after update 3,0003{,}000, the shift is consistent with the predicted negative derivative in Eq. ( [100](https://arxiv.org/html/2609.26679v1#A6.E100 "In F.1 Earlier task alignment shortens the predicted delay ‣ Appendix F Directional NTK-alignment intervention ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning")). Matching the full 97×9797\\times 97 label kernel changes several Fourier components and may alter other parts of the trajectory, so the experiment tests the direction of the initial-alignment effect rather than the scalar one-mode closure in isolation.

## Appendix G Experimental configurations and analysis conventions

The accompanying experiment package centralizes the configurations below in the model, training, and analysis code. We report optimizer weight decay in the convention passed directly to PyTorch SGD. The anonymous artifact URL remains to be inserted in the reproducibility statement.

### G.1 Spectral diagnostics on addition modulo 97

For Figure [2](https://arxiv.org/html/2609.26679v1#S3.F2 "Figure 2 ‣ 3.1 Task-aligned NTK structure develops after fitting ‣ 3 Experiments ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning"), we use the 97⋅98/2=4,75397\\cdot 98/2=4{,}753 unordered input pairs allowed by commutativity. A NumPy generator with seed 00 permutes the pairs; the first 2,3762{,}376 are used for training and the remaining 2,3772{,}377 are held out. Each input concatenates two 9797-dimensional one-hot vectors, and the target is the 9797-dimensional one-hot encoding of the modular sum.

The model is a bias-free →→97194\\!\\to\\!256\\!\\to\\!97 one-hidden-layer ReLU MLP with He-normal initialization and model seed 00. We train for 20,00020{,}000 full-batch SGD updates with learning rate 4040, coupled weight decay 2.5×10−52.5\\times 10^{-5}, and no momentum. The loss is ordinary PyTorch MSE averaged over samples and all 9797 output coordinates, so 𝒩=2/(2376⋅97)=1/115,236\\mathcal{N}=2/(2376\\cdot 97)=1/115{,}236 and λWnorm=2.8809{\\lambda\_{W}^{\\mathrm{norm}}}=2.8809. Accuracy is recorded every 100100 updates.

For training examples 𝐱μ,𝐱ν\\mathbf{x}\_{\\mu},\\mathbf{x}\_{\\nu} with labels cμ,cνc\_{\\mu},c\_{\\nu}, the true-label-projected sample NTK is

|     |     |     |     |
| --- | --- | --- | --- |
|  | Kμ​νTL=⟨∇𝜽fcμ​(𝐱μ),∇𝜽fcν​(𝐱ν)⟩.K^{\\rm TL}\_{\\mu\\nu}=\\left\\langle\\nabla\_{\\boldsymbol{\\theta}}f\_{c\_{\\mu}}(\\mathbf{x}\_{\\mu}),\\nabla\_{\\boldsymbol{\\theta}}f\_{c\_{\\nu}}(\\mathbf{x}\_{\\nu})\\right\\rangle. |  | (105) |

We first average these entries between groups of examples with the same output labels, producing a 97×9797\\times 97 matrix K~\\widetilde{K}. We then diagonal-normalize and symmetrize the label kernel,

|     |     |     |     |
| --- | --- | --- | --- |
|  | K¯c​c′=K~c​c′K~c​c​K~c′​c′.\\bar{K}\_{cc^{\\prime}}=\\frac{\\widetilde{K}\_{cc^{\\prime}}}{\\sqrt{\\widetilde{K}\_{cc}\\widetilde{K}\_{c^{\\prime}c^{\\prime}}}}. |  | (106) |

Panel (a) uses updates 00, 3,0003{,}000, 8,0008{,}000, and 20,00020{,}000. Eigensystems are computed at updates 00, 100100, 500500, 1,0001{,}000, 3,0003{,}000, 6,0006{,}000, 8,0008{,}000, 10,00010{,}000, and 20,00020{,}000. Eigenvectors are matched backward through these checkpoints by maximum absolute overlap, with signs chosen continuously. Panel (c) displays four tracked leading eigenvectors at updates 00, 1,0001{,}000, 6,0006{,}000, and 20,00020{,}000.

### G.2 High-resolution MLP phase sweep on addition modulo 23

We use all 232=52923^{2}=529 ordered input pairs. Each input concatenates two 2323-dimensional one-hot vectors, and the target is a 2323-dimensional one-hot encoding of the modular sum. NumPy seed 20272027 selects the training differences

|     |     |     |     |
| --- | --- | --- | --- |
|  | 𝒟={0,1,2,3,4,5,6,7,12,13,14,16,17,19,20,22},\\mathcal{D}=\\{0,1,2,3,4,5,6,7,12,13,14,16,17,19,20,22\\}, |  | (107) |

and we define

|     |     |     |     |
| --- | --- | --- | --- |
|  | 𝒯𝒟={(a,b):a−bmod23∈𝒟}.\\mathcal{T}\_{\\mathcal{D}}=\\{(a,b):a-b\\bmod 23\\in\\mathcal{D}\\}. |  | (108) |

The split contains 368368 training and 161161 held-out pairs and is invariant under simultaneous shifts (a,b)↦(a+r,b+r)(a,b)\\mapsto(a+r,b+r).

At every grid point we initialize the same bias-free →→2346\\!\\to\\!256\\!\\to\\!23 one-hidden-layer ReLU MLP with He-normal initialization and model seed 00. Training uses ordinary PyTorch MSE averaged over the 368×23368\\times 23 residual coordinates, full-batch SGD, coupled L2L\_{2} weight decay, and no momentum. Thus 𝒩=2/(368⋅23)=1/4232\\mathcal{N}=2/(368\\cdot 23)=1/4232. Each run lasts 24,00024{,}000 updates and is evaluated every 100100 updates. The grid contains 8484 logarithmically spaced learning rates from 0.10.1 to 100100 and 9090 logarithmically spaced weight decays from 5×10−65\\times 10^{-6} to 2×10−32\\times 10^{-3}, for 7,5607{,}560 runs.

We classify runs using a 90%90\\% accuracy threshold. _Grokking_ requires final training and held-out accuracy above 90%90\\%; _memorization_ requires final training accuracy above 90%90\\% but held-out accuracy below it; _forgetting_ means that training accuracy crossed 90%90\\% earlier but ends below it; all remaining runs are _no fitting_. For Figure [3](https://arxiv.org/html/2609.26679v1#S3.F3 "Figure 3 ‣ 3.2 Grokking boundaries in the MLP optimizer plane ‣ 3 Experiments ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning") b, we retain only runs that ultimately grok and record the first update at which held-out accuracy reaches 10%10\\%. This lower threshold marks the beginning of the final MLP generalization rise and does not affect the phase labels.

The representative runs in Figure [7](https://arxiv.org/html/2609.26679v1#A7.F7 "Figure 7 ‣ G.2 High-resolution MLP phase sweep on addition modulo 23 ‣ Appendix G Experimental configurations and analysis conventions ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning") are fixed cells from the same dense sweep. Their (η,λW)(\\eta,\\lambda\_{W}) values are (0.87048,3.2930×10−5)(0.87048,3.2930\\times 10^{-5}) for memorization, (2.36316,1.18329×10−4)(2.36316,1.18329\\times 10^{-4}) for grokking, (1.43426,1.24846×10−3)(1.43426,1.24846\\times 10^{-3}) for forgetting, and (28.6967,1.89561×10−4)(28.6967,1.89561\\times 10^{-4}) for no fitting. The grokking example first reaches 90%90\\% held-out accuracy at update 8,3008{,}300. The forgetting example reaches peak training accuracy 0.9480.948 and ends at 0.2040.204.

Figure 7: Representative trajectories for the four MLP phase outcomes. We show training and held-out accuracy for fixed cells from the modulo-2323 sweep. The dotted horizontal line is the 90%90\\% threshold used to classify the phase map.

#### G.2.1 Boundary extraction and calibration

For each usable weight-decay column, we define the memorization–grokking boundary as the geometric midpoint between the last memorizing learning rate and the first grokking learning rate. We exclude the detached high-step island. The finite-time boundary uses 5353 midpoints over 2×10−5≤λW≤7×10−42\\times 10^{-5}\\leq\\lambda\_{W}\\leq 7\\times 10^{-4} with equal weight in log learning rate.

To evaluate Eq. ( [15](https://arxiv.org/html/2609.26679v1#S2.E15 "In Corollary 1 (Finite-training grokking boundary). ‣ 2.3 Grokking boundaries in the (𝜂,𝜆_𝑊) plane ‣ 2 Post-fit residuals drive tangent feature learning ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning")), define the scaled coordinates

|     |     |     |     |
| --- | --- | --- | --- |
|  | Λ^=𝒩​Λ,q^=𝒩2​a​y.\\widehat{\\Lambda}=\\mathcal{N}\\Lambda,\\qquad\\widehat{q}=\\mathcal{N}^{2}ay. |  | (109) |

This removes 𝒩\\mathcal{N} from the integrand while leaving the optimizer prefactor 1/(2​λW​T)1/(2\\lambda\_{W}T) unchanged. The calibrated finite-time curve is

|     |     |     |     |
| --- | --- | --- | --- |
|  | ηfit​(λW)=Cη2​λW​T​∫Λ^0Λ^gΛ^+2​λWq^−Λ^2−2​λW​Λ^​𝑑Λ^\\boxed{\\eta\_{\\rm fit}(\\lambda\_{W})=\\frac{C\_{\\eta}}{2\\lambda\_{W}T}\\int\_{\\widehat{\\Lambda}\_{0}}^{\\widehat{\\Lambda}\_{g}}\\frac{\\widehat{\\Lambda}+2\\lambda\_{W}}{\\widehat{q}-\\widehat{\\Lambda}^{2}-2\\lambda\_{W}\\widehat{\\Lambda}}\\,d\\widehat{\\Lambda}} |  | (110) |

with one global kinetic factor CηC\_{\\eta}. If omitted modes and slow variation in the projected coupling rescale the scalar velocity as d​Λ^/d​t≃κ​F​(Λ^)d\\widehat{\\Lambda}/dt\\simeq\\kappa F(\\widehat{\\Lambda}), then crossing times are multiplied by 1/κ1/\\kappa and Cη=1/κC\_{\\eta}=1/\\kappa. The MSE normalization 𝒩\\mathcal{N} is known exactly and is not part of this calibration. The quantities Λ^0\\widehat{\\Lambda}\_{0}, Λ^g\\widehat{\\Lambda}\_{g}, and q^\\widehat{q} are inferred jointly from the optimizer-plane boundary rather than measured from NTK trajectories. We therefore treat them as effective coordinates of the scalar boundary model: the fit tests the predicted dependence of the crossing time on learning rate and weight decay, but does not determine the absolute magnitude of post-fit NTK growth.

Panel (a) of Table [2](https://arxiv.org/html/2609.26679v1#A7.T2 "Table 2 ‣ G.3 Transformer phase sweep on addition modulo 23 ‣ Appendix G Experimental configurations and analysis conventions ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning") collects the values used for every overlay in Figure [3](https://arxiv.org/html/2609.26679v1#S3.F3 "Figure 3 ‣ 3.2 Grokking boundaries in the MLP optimizer plane ‣ 3 Experiments ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning") a. The mode-reachability cutoff is derived from the fitted reduced coordinates rather than fitted independently. The training-fit and stability overlays use the functional forms of Eqs. ( [17](https://arxiv.org/html/2609.26679v1#S2.E17 "In 2.3 Grokking boundaries in the (𝜂,𝜆_𝑊) plane ‣ 2 Post-fit residuals drive tangent feature learning ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning")) and ( [18](https://arxiv.org/html/2609.26679v1#S2.E18 "In 2.3 Grokking boundaries in the (𝜂,𝜆_𝑊) plane ‣ 2 Post-fit residuals drive tangent feature learning ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning")), with their locations calibrated separately to the observed boundaries. In particular, the stability calibration is summarized by an effective scale Λ^stab\\widehat{\\Lambda}\_{\\rm stab} and is not obtained from a direct measurement of 𝒩​Λmax​(K)\\mathcal{N}\\Lambda\_{\\max}(K).

Over the small-decay interval used for the inverse-decay comparison, the empirical boundary has log–log slope −1.1681-1.1681, while the calibrated finite-time curve has slope −0.9043-0.9043. These slopes are diagnostics of the boundary shape and are not additional fit parameters.

### G.3 Transformer phase sweep on addition modulo 23

We use the same task, difference split, split seed, training size, and held-out size as in Appendix [G.2](https://arxiv.org/html/2609.26679v1#A7.SS2 "G.2 High-resolution MLP phase sweep on addition modulo 23 ‣ Appendix G Experimental configurations and analysis conventions ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning"), but present each input as a sequence of two 2323-dimensional one-hot tokens. A learned affine encoder maps each token to dmodel=32d\_{\\rm model}=32. The model contains one pre-norm Transformer block with four attention heads, a ReLU feed-forward width of 6464, residual connections, learned embeddings for the two positions, a final LayerNorm, and an affine →2364\\!\\to\\!23 decoder. We use model seed 00 at every grid point.

Training uses full-batch SGD with coupled L2L\_{2} weight decay and no momentum. The loss is PyTorch MSE between softmax probabilities and one-hot targets, averaged over samples and output coordinates, so 𝒩=1/4232\\mathcal{N}=1/4232. Each run lasts 60,00060{,}000 updates and is evaluated every 250250 updates. The phase file contains 4242 uniformly logarithmically spaced learning rates from 5×10−35\\times 10^{-3} to 10310^{3} and 4545 uniformly logarithmically spaced weight decays from 10−710^{-7} to 10−310^{-3}, for 1,8901{,}890 independently trained networks. We use the same 90%90\\% phase definitions as for the MLP.

For every usable weight-decay column, we extract the geometric midpoint between the last memorizing and first grokking learning rate. We fit the inverse-decay form η=A/λW\\eta=A/\\lambda\_{W} over 10−5≤λW≤3×10−410^{-5}\\leq\\lambda\_{W}\\leq 3\\times 10^{-4}. The current phase grid yields the values in panel (b) of Table [2](https://arxiv.org/html/2609.26679v1#A7.T2 "Table 2 ‣ G.3 Transformer phase sweep on addition modulo 23 ‣ Appendix G Experimental configurations and analysis conventions ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning").

|     |     |     |     |
| --- | --- | --- | --- |
| Curve | Relation | Calibration values | Fit range / quality |
| (a) MLP |
| Finite-time boundary | Eq. ( [110](https://arxiv.org/html/2609.26679v1#A7.E110 "In G.2.1 Boundary extraction and calibration ‣ G.2 High-resolution MLP phase sweep on addition modulo 23 ‣ Appendix G Experimental configurations and analysis conventions ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning")) | Cη=24.840743C\_{\\eta}=24.840743, Λ^0=0.9281806\\widehat{\\Lambda}\_{0}=0.9281806, Λ^g=0.9284920\\widehat{\\Lambda}\_{g}=0.9284920, q^=0.8633973\\widehat{q}=0.8633973; κ=Cη−1=0.0402564\\kappa=C\_{\\eta}^{-1}=0.0402564 | 5353 midpoints, 2×10−5≤λW≤7×10−42\\times 10^{-5}\\leq\\lambda\_{W}\\leq 7\\times 10^{-4}; log-MSE 0.0268000.026800; log-RMSE 0.1637060.163706 |
| Inverse-decay asymptote | η=A/λW\\eta=A/\\lambda\_{W} | A=1.0916395×10−4A=1.0916395\\times 10^{-4} | 3434 midpoints, 2×10−5≤λW≤2×10−42\\times 10^{-5}\\leq\\lambda\_{W}\\leq 2\\times 10^{-4}; log-RMSE 0.1199020.119902 |
| Mode reachability | Eq. ( [16](https://arxiv.org/html/2609.26679v1#S2.E16 "In 2.3 Grokking boundaries in the (𝜂,𝜆_𝑊) plane ‣ 2 Post-fit residuals drive tangent feature learning ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning")) | λWmode=7.0000×10−4\\lambda\_{W}^{\\rm mode}=7.0000\\times 10^{-4}, derived from q^\\widehat{q} and Λ^g\\widehat{\\Lambda}\_{g} | no independent fit |
| Training fit | Eq. ( [17](https://arxiv.org/html/2609.26679v1#S2.E17 "In 2.3 Grokking boundaries in the (𝜂,𝜆_𝑊) plane ‣ 2 Post-fit residuals drive tangent feature learning ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning")) | λWfit=8.3821151×10−4\\lambda\_{W}^{\\rm fit}=8.3821151\\times 10^{-4} | location minimizes final-training-accuracy misclassification |
| Frozen-kernel stability | Eq. ( [18](https://arxiv.org/html/2609.26679v1#S2.E18 "In 2.3 Grokking boundaries in the (𝜂,𝜆_𝑊) plane ‣ 2 Post-fit residuals drive tangent feature learning ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning")) | effective Λ^stab=0.1056011\\widehat{\\Lambda}\_{\\rm stab}=0.1056011; low-decay cutoff η≃18.9282\\eta\\simeq 18.9282 | location calibrated to observed instability boundary |
| (b) Transformer |
| Inverse-decay boundary | η=A/λW\\eta=A/\\lambda\_{W} | A=1.6875621×10−5A=1.6875621\\times 10^{-5} | 1616 boundary points, 10−5≤λW≤3×10−410^{-5}\\leq\\lambda\_{W}\\leq 3\\times 10^{-4}; log-RMSE 0.1070160.107016 |

Table 2: Boundary calibrations for the optimizer-plane experiments. Panel (a) lists the MLP phase-diagram overlays; panel (b) gives the inverse-decay fit for the Transformer memorization–grokking boundary. The theory specifies the functional forms of the MLP curves, while their locations are determined either by the reduced finite-time fit or by separate effective calibrations to the observed boundaries.

For the transition-time scaling panel, we retain runs that ultimately grok and record the first held-out 80%80\\% crossing. The 80%80\\% threshold is chosen to lie in the final generalization rise rather than near chance accuracy, 1/23≈4.35%1/23\\approx 4.35\\%. On the current 1,8901{,}890-run grid, the grokking cohort has median η​λW​s80=0.920\\eta\\lambda\_{W}s\_{80}=0.920.

### G.4 Reduced-mode potential illustration

For Figure [5](https://arxiv.org/html/2609.26679v1#A3.F5 "Figure 5 ‣ C.2 Proof of Theorem ‣ Appendix C One-mode neural-tangent-hierarchy closure ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning"), we integrate Eq. ( [12](https://arxiv.org/html/2609.26679v1#S2.E12 "In 2.2 Residuals drive task-aligned tangent growth ‣ 2 Post-fit residuals drive tangent feature learning ‣ A Spectral Theory of Grokking: Weight Decay induces Feature Learning")) with dimensionless normalized decay λWnorm=0.15{\\lambda\_{W}^{\\mathrm{norm}}}=0.15, coupling a=1a=1, initial strength Λ⁡(0)=0.25\\Lambda(0)=0.25, fourth-order Runge–Kutta step 0.0020.002, and final time 1818. The task-aligned trajectory uses y=1y=1, while the unsupported control uses y=0y=0. These values are chosen only to display the qualitative reduced dynamics and are not fitted to an empirical run.
