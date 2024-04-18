# Recent Models and Advancements
1. Started with Nearest Neighbour Upscaling, Interpolation, Bicubic Interpolation
2. CNN Based Techniques - SRCNN and VDSR
3. GANs - SRGAN and ESRGAN
4. Transformers and Attention (Need to Explore)

1. Loss Functions : pixel-wise loss, content loss, and adversarial loss, Perceptual Loss
2. Metrics : MSE (Bad), SSIM and MS-SSIM
3. [Benchmarking Datasets](https://www.kaggle.com/datasets/jesucristo/super-resolution-benchmarks)

# Conferences and Challenges
1. CVPR (NTIRE - Research Challenges)
1. [VIP Cup](https://signalprocessingsociety.org/community-involvement/video-image-processing-cup)
1. ICIP
1. ECCV
1. TPAMI

# Problems Currently Existing
1. Difficult to Train (Zero Shot Learning)
1. Uses Old Metrics (Can we Imitate Human Perception)
1. Context Awareness
1. Network Interpolation
2. Often results in artifacts, blurred details and overly smooth images
    + Non Specific Blurring needs to be addressed
3. There are infinitely many high-res images that can be constructed from one low res image
4. Single Image Super resolution - corrupted by blur and noise with unknown standard deviation requires an optimal parameter selection strategy

# Idea
1. Prompt based Image Super-Resolution (Generative Fill Method)
1. Zero shot learning (Fine tuning Quickly for different applications)
1. New Metrics (Human Perception)
1. Training with Low Data
1. Medical Image Biasing

## Bottleneck : Data Processing Inequality
> We cannot get out any more information than we put in. Theoretically, it makes super-resolution imporrible. But, NNs can hallucinate information from training sets

# Areas
1. Single Image Super-Resolution 
2. Multiple Image Super Resolution : Multiple Images to Create one Image
3. Video Super-Resolution

## The Benefit
+ Transferring Low Amount of Pixels to then Reconstruct a higher quality image for the user (Reducing Space and Design Efforts)
+ Medical Imaging : To get clearer Details

## Applications
> Adding Task Specific Priors will help in that application as well

+ Natural Scenes
+ Medical 
+ Forensics Imaging
+ Satellite and Remote Sensing

---
# Resources
1. [A List of Models](https://blog.paperspace.com/image-super-resolution/)
1. [Awesome Collection of Papers](https://github.com/ChaofWang/Awesome-Super-Resolution)
1. [List of Models](https://github.com/togheppi/pytorch-super-resolution-model-collection/blob/master/model.py)
1. [Another repo](https://github.com/idealo/image-super-resolution)
1. [Iterative Refinement](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement)

---
# Think about Later

```markdown
# Project Plan
1. Decide on Topic
2. See the existing Papers
3. See conferences on that topic / recent advances
4. Decide where to advance / novelty

## Conferences
- RecSys (STIRS)
- CodaLab

## To Do
- Submit Paper
- Present in Conferences (How to send to conference papers)

## More
- Understanding Academia

# To DO
Comparision with standard datasets
Zero shot model - Take an existing model and train on small data

Continue with this for capstone project

Write a paper for good conference

# General Research
- Understand Research, conferences, Academia etc
- See muthu and rhea papers
- See research challenges

Chaotic Algorithms

RecSys challenge, other codalab challenges to submit papers to + Conferences (See muthu and rhea papers + Academia papers)

how to publish research paper, understand conferences etc.

Github - https://github.com/topics/super-resolution
https://github.com/idealo/image-super-resolution
https://github.com/JingyunLiang/SwinIR
https://github.com/researchmm/TTSR

https://github.com/deepak112/Keras-SRGAN

image upscaler chrome extensions - how do htey work
```