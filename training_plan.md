
For each run with these seeds: 45, 2, 255, 18, 33
Epochs: 20

## Training Aim to Finish by Wednesday

Balanced experiments: (Ryan)
- (DONE) Ce
- (DONE) Focal Loss
- (DONE) Label Smoothing

Imbalanced:
- CE (Libby, Ryan for last 2 seeds)
- Weighted CE (Libby, Ryan for last 2 seeds)
- Label Smoothing (Libby, Ryan for last 2 seeds)
- (DONE) Focal Loss (Oliver)
- Class Balanced (Yuri)

Folder structure:

    Balanced
        - Ce
            - Seed 45
            - Seed 2
        - Focal Loss
        - Label Smoothing
    Imbalanced
        - Ce

In each run training with one seed, one loss, and either balanced/imalanced
Data should be saved in appropriate folders automatically

# Metrics Things to Look At:

- convergence rates
- accuracy, precision, recall, general
- logits for confidence over time in training analysis (could think more about)
- confusion matrix analysis
- metrics
    - F1
    - top 1/3
    - 
    - 

# Intersting Things

- focal loss for imabalanced data makes it worse than cross entropy
- examinig the logits
    - accuracy vs confidence over epochs (different prioritisations for diffferent loss functions)
    - plot the distribution of confidences over epochs
- statistical quantification of all observations

# Plotting

Can plot all the balanced and imbalanced graphs together

Graphs to plot:
    - top 1/3
    - average for each class
        - accuracy over time
        - precision over time
        - recall over time
        - loss over time
    - testing confusion matrices over epochs
    - accuracy vs confidence over time
    - distribution of confidence over time
    - F1 measure over epochs
    - average recall across 

Teamwork:
- Libby: analyse plots for insights & method drafting
- Ryan: balanced & imbalanced F1 and top3 (accuracy, precision & recall)
- Yuri: statistical measures of significance & gifs & class balanced data
- Oliver: accuracy/confidence + confidence distribution over time