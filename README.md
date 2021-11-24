# Master
Code and data used in my master thesis. 

## The project, in brief
Medical deep learning pipelines are known to exhibit poor generalizability. Despite exhibiting great performance on holdout sets, they tend to fail entirely when deployed in practical settings. My thesis aims to explore ways to mitigate this problem, and in a broader sense methods to increase generalizability. The approach can be outlined as follows:
 * Find a way to sufficiently model the variation that can be present in a given domain
 * Use this model of natural variation to modify a training image 
 * Interleave a contrastive learning step between training the main model, and minimize the difference between the expected change and actual change in output

Since this particular project relates to segmenting polyps found in colonoscopy images, it is necessary to develop a model of natural variation which can represent the variability one might expect to see in polyps. Of course, finding a model to generate all the different polyps one might see and will ever see in all the different intestinal environments is a wholly intractable problem, and a consequence some simplification is necessary. One could for example solely alter texture, shape, or  location. This project limits the variability to the polyp itself, and in particular the shape and size of the polyps. To this end, a GAN-inpainter is used. The inpainter is trained to paint polyps in some region defined by a mask. Variability in the training data can then be induced by slightly modifying the masks already present in the dataset. 

This now needs to be incorporated in the training process of the segmentation model as contrastive learning. To do this, I use a custom loss which measures the discrepancy between the expected change in the segmentation and the actual change in the segmentation. Optimally, the change in the mask should lead to a correpsonding change in the segmentation, but poorly generalized pipelines evidently exhibit high degrees of sensitivty when perturbed off the training manifold. As a result, this is not often the case in poorly generalized predictors. Thus, this discrepancy should be minimized. 
