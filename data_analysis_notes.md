
- Graphs should be zoomed in (especially for balanced) - can't see any difference
- What kind of results have other studies found? Are our results at all significant? Do we need more epochs (will this help? How many do ther studies use?)? 
- Adding plots of loss
- For imbalanced why is weighted ce so bad? It is meant to perform better here. Is there a bug in our code? Why else might this happen? (we will probably need to do more experiments for this)
- Need to quantify how each loss function performed. The graphs are good but need solid numbers to report - what are the most important/telling statistics? (e.g. label smoothing acheieved a mean accuracy of 89%, ce acheived a training loss of 0.1)
- We should look for similar studies to compare to and talk about in background


Very relevant:
https://ieeexplore.ieee.org/abstract/document/9323583
https://link.springer.com/article/10.1007/s11042-024-19543-8
https://d1wqtxts1xzle7.cloudfront.net/100504839/NN_ImgProc-libre.pdf?1680279663=&response-content-disposition=inline%3B+filename%3DLoss_Functions_for_Image_Restoration_Wit.pdf&Expires=1778589234&Signature=KlNkmSr0mYKX-lMMnw0qP4wsf6ntvYfysM02w4ckl33d7Ac1a6QPyW4tOatTVq0ny0WbVQ0iLwr1j6fXz3xsEuitZALHVO5mVWPPTg5GovuyjMyh-ESfRoSD71tV5ovVDItbZY9EKK9sRKZPHdKp5A7OB53y0uC96Iiu4XM8bp8Xy0sSwdeTMCGoqc5D4Pk5x0~~YY3yuHShHTAPVSrl0IBqAKuELG~5J4DX6dCznVAXMv94JXUpM6wbFi1Nv8K1FZTVlk6tgmUTonJOFr9pVMVQf-pQmT8S0weqhmL9d0qwpw6YRgyAcWmsS0zS~18IhsS1-Vs5gfJ8wLKJD6nykg__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA

Somewhat related:
https://ieeexplore.ieee.org/abstract/document/9086269
https://ieeexplore.ieee.org/abstract/document/9323583
https://arxiv.org/abs/1702.05659
https://onlinelibrary.wiley.com/doi/full/10.1155/2021/6660961
https://openaccess.thecvf.com/content_cvpr_2018/html/Wan_Rethinking_Feature_Distribution_CVPR_2018_paper.html
https://www.sciencedirect.com/science/article/abs/pii/S1047320319301336



Review of loss functions:
[arXiv:2504.04242v1\[cs.LG\] ](https://arxiv.org/html/2504.04242v1)
https://link.springer.com/article/10.1007/s10462-025-11198-7
