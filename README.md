## Official repository for Reproduction of CVPR2023 paper: "Towards Artistic Image Aesthetics Assessment: a Large-scale Dataset and a New Method"

## Introduction

In our reproduction project, we revisited the CVPR2023 paper entitled "Towards Artistic Image Aesthetics Assessment: a Large-scale Dataset and a New Method" which introduces a significant dataset, BAID, and a novel method, SAAN (Style-specific Art Assessment Network). This paper addresses the subjective and complex task of artistic image aesthetic assessment (AIAA) by proposing a large-scale dataset of 60,337 artistic images and a new method that integrates style-specific and generic aesthetic information. Our objective was to replicate the results presented in the paper to validate the efficacy of SAAN and explore its performance on different datasets, rewriting the code for efficiency and readability, and perform an in-depth hyperparameter sensitivity analysis.

## Methodology

### Existing Code Evaluation

- **Description**: We began by evaluating the existing code from [source/repository]. Our evaluation process involved...
- **Findings**: During the code evaluation, we noticed...

### New Data Evaluation

- **Data Collection**: We sourced additional datasets from [data source] to assess the model's robustness and generalizability. The new datasets include...
- **Results on New Data**: The model's performance on new datasets showed...

#### New Code Variant: Code Revamping
To address the challenge of executing the SAAN model across different platforms, we introduced an intelligent path management system. We crafted a configuration file that discerns whether the code is running on Kaggle's cloud compute or a local machine. This adaptive config file dynamically assigns file paths used throughout the codebase, significantly streamlining the process of dataset and resource handling. This innovation eliminates the need for manual path adjustments, thereby reducing the setup time and potential for unpredictable holdups and human error. Here are the main aspects of this new configuration:

- **Automatic Path Detection**: A configuration file is now included that auto-determines the running environment and adjusts file paths utilized in the code accordingly.
- **Seamless Transition**: Switch between local and Kaggle environments without manual path corrections, significantly reducing setup times and potential errors.

Given the extensive computational demands of training the SAAN model — taking several days even on high-end GPUs as mentioned in the original README by the authors of the paper — it's impractical to rely on a single uninterrupted training session, especially on platforms like Kaggle, which have session time constraints. To this end, we've overhauled the weight management system with the following capabilities:

- **Weight Loading**: The model now supports the loading of pre-trained weights. This feature is particularly crucial as it allows for continuity in training beyond a single session. Researchers can now pick up from where the last successful training epoch ended without the necessity to start afresh.
- **Mid-Training Weight Saving**: We've instituted a robust checkpointing mechanism that saves the model's weights at regular intervals during the training process. This incremental saving is a safeguard against data loss from potential crashes, memory overflows, or the exceeding of Kaggle's 12-hour session limit. The model's training can be resumed from the last saved checkpoint, conserving both time and computational resources.

- **Impact of Refactoring**:


### Hyperparameter Sensitivity Analysis

- **Approach**: We systematically varied hyperparameters such as [list a few hyperparameters] to understand their impact on model performance.
- **Key Insights**: The sensitivity analysis revealed that...

## Results

Provide a brief overview of the results, perhaps with tables or figures if appropriate.

### Reproduced Results

- **Original vs. Reproduced**: Compared to the original implementation, our reproduced results are...
- **Challenges Encountered**: We faced challenges such as...

### New Data Results

- **Consistency Across Datasets**: The model's consistency across various datasets was evaluated, revealing...

### New Code Variant Results

- **Performance Comparison**: With the refactored code, the performance [improved/remained consistent] because...

### Hyperparameter Analysis

- **Critical Hyperparameters**: Our analysis highlighted [hyperparameter] as particularly influential because...

## Discussion

Discuss any discrepancies, unexpected findings, or conclusions drawn from the reproduction effort.

