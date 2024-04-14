## Official repository for Reproduction of CVPR2023 paper: "Towards Artistic Image Aesthetics Assessment: a Large-scale Dataset and a New Method"

## Introduction

In our reproduction project, we revisited the CVPR2023 paper entitled "Towards Artistic Image Aesthetics Assessment: a Large-scale Dataset and a New Method" which introduces a significant dataset, BAID, and a novel method, SAAN (Style-specific Art Assessment Network). This paper addresses the subjective and complex task of artistic image aesthetic assessment (AIAA) by proposing a large-scale dataset of 60,337 artistic images and a new method that integrates style-specific and generic aesthetic information. Our objective was to replicate the results presented in the paper to validate the efficacy of SAAN and explore its performance on different datasets, rewriting the code for efficiency and readability, and perform an in-depth hyperparameter sensitivity analysis.

## Methodology

### New Code Variant: Code Revamping
To address the challenge of executing the SAAN model across different platforms, we introduced an intelligent path management system. We crafted a configuration file that discerns whether the code is running on Kaggle's cloud compute or a local machine. This adaptive config file dynamically assigns file paths used throughout the codebase, significantly streamlining the process of dataset and resource handling. This innovation eliminates the need for manual path adjustments, thereby reducing the setup time and potential for unpredictable holdups and human error. Here are the main aspects of this new configuration:

- **Automatic Path Detection**: A configuration file is now included that auto-determines the running environment and adjusts file paths utilized in the code accordingly.
- **Seamless Transition**: Switch between local and Kaggle environments without manual path corrections, significantly reducing setup times and potential errors.

Given the extensive computational demands of training the SAAN model — taking several days even on high-end GPUs as mentioned in the original README by the authors of the paper — it's impractical to rely on a single uninterrupted training session, especially on platforms like Kaggle, which have session time constraints. To this end, we've overhauled the weight management system with the following capabilities:

- **Weight Loading**: The model now supports the loading of pre-trained weights. This feature is particularly crucial as it allows for continuity in training beyond a single session. Researchers can now pick up from where the last successful training epoch ended without the necessity to start afresh.
- **Mid-Training Weight Saving**: We've instituted a robust checkpointing mechanism that saves the model's weights at regular intervals during the training process. This incremental saving is a safeguard against data loss from potential crashes, memory overflows, or the exceeding of Kaggle's 12-hour session limit. The model's training can be resumed from the last saved checkpoint, conserving both time and computational resources.


## Testing on New Dataset ##
We decided to run an experiment on a different dataset of new images. For this, we downloaded 25 copyright free images of (oil) paintings (unsplashed.com) and put these images in a google form. For each image, the person filling in the form gets two tasks: 
- Answer *Yes* or *No* to the question: Do you consider this artwork to be pleasing and good-looking?
- Give a rating of the artwork on a scale of 1 to 10

The first bullet point is used to try and replicate the way votes were counted in the original paper. For every Yes vote in the form, we would count it as one vote.
The second bullet point was used as a backup benchmark and to see what people actually thought of the paintings. 

After creating the form, we distributed them to random people to get their opinions. The distribution consists mainly of people not necessarily associated with any knowledge about art (family members, study association members, friends and CS students). We managed to get 64 people to fill in the form rating all of the 25 images, after which we transferred all the results to a spreadsheet file.

In the spreadsheet we calculated the amount of Yes votes of the first question compared to the amount of total votes for each image and calculated a grade based on this. Besides this, we also calculated the average of the second question for each image.  We then made a .CSV file for both of these grades: form_data1.csv and form_data2.csv.
We ran both of these files on the model and got the following results:

RESULTS HERE


The accuracy and scores of the model were very low.
We quickly realized that the grades we got from the form were way higher than the grades of the images in the BAID dataset that were used to train the model, and we had to make some changes. We used the same sigmoid-like function used in the paper to get new scores based on the amount of *Yes* votes: 

![sigmoid function](https://github.com/samuelribaric/BAID-Reproducibility/assets/57133973/fa6577b5-358c-4c4f-a8a5-10d6a3bb7363)

For our results, the variables can be explained as follows:
- V_mi is the average number of Yes votes for the 25 images, which turned out to be 46.04 (a lot of people liked the majority of the paintings).
- V_i is the amount of Yes votes of image i.

We then used the equations to get the new scores for all the images, which were way more concurrent to the BAID dataset regarding the grades.
We stored the scores in a new .CSV file: form_data3.csv. And we ran this file getting the following results:

RESULT HERE


As you can see, the model still performs relatively badly compared to testing on any test set of the original BAID dataset, but it is already better than the files we used previously.

Although the model seems to not work as well on this dataset, this could largely be due to problems with the dataset itself and how the experiment was conducted. Instead of people voting for an image like done with the BAID dataset, people get the choice of finding a painting good-looking or not. People most likely tend to vote ‘Yes’ quicker when not necessarily comparing paintings, but just getting the choice of liking or not liking a painting, in turn skewing the grades to be higher. If the 25 images were presented in a manner more correlated to the way it was done with the BAID dataset where people could vote on the paintings, results might have differed. 

Besides this, the sample size of 25 images and 64 votes on the form is still very small in comparison to other datasets. Unfortunately, we did not have more than 64 people rate our 25 images at the time (which we already found quite a lot of people wanting to help with our experiment). This is something that also has influence on the apparent inaccuracy of the model.

The spreadsheet with all of the results and additional calculations can be found [here](https://docs.google.com/spreadsheets/d/1R3sYHwE8HyBdyatLHbCgOrEzwhEw8QQJ-6G7sv1jh-8/edit?usp=sharing)!

## Ablation Study ##

We thought it would be interesting to see what would happen when we did an ablation study for this project. In the paper it is mentioned how they combine multiple components like a Style Feature Extractor (VGG) and an AestheticFeature Extractor (Restnet). We decided to take these components out of the training loop and see what the results would be. The performance can be seen in the results section, right below this section. It immediately strikes out that the model performs really bad when Resnet is taken out of the loop. However, when VGG is taken out of the loop it performs similarly as when it was still in the loop. This was a really interesting result and this could mean that VGG is not really neccessary for this model to perform well on the given dataset. We also ran these models on our own dataset. The meaning of each of the form datas can be found in the new dataset part, right above this part. Because it was only 25 pictures, the results are not significant at all and the results are also really bad. However we thought it was fun to check what the model would do and the results can be seen in the results section below.

<img src="https://github.com/samuelribaric/BAID-Reproducibility/assets/44850442/6da4e2c9-0bef-4a95-8ddb-a14ec440cd4d" alt="11064" width="200"/>

This is a picture of an example image out of the dataset. When rated by people, it got a rating of 4.352386. When ran on the model, the rating returned was a: I DO NOT HAVE THE RATINGS TO MY DISPOSAL IT IS IMAGE 11064

## Results

This section provides an overview of the outcomes from our project efforts, including reproducing the original paper's results, evaluating new datasets, and conducting an ablation study with modified model configurations.
### Reproduced Results
#### Initial Test with Weights from Authors

| Metric                     | Value                  |
|----------------------------|------------------------|
| Significance Statistic     | 0.472                  |
| Significance p-value       | 0.0                    |
| Pearson Statistic          | 0.467                  |
| Pearson p-value            | 0.0                    |
| Accuracy                   | 0.767                  |

#### Epoch 39 (First training)

| Metric                     | Value                  |
|----------------------------|------------------------|
| Significance Statistic     | 0.462                  |
| Significance p-value       | 0.0                    |
| Pearson Statistic          | 0.468                  |
| Pearson p-value            | 0.0                    |
| Accuracy                   | 0.780                  |

#### Epoch 99 (Finished Training)

| Metric                     | Value                  |
|----------------------------|------------------------|
| Significance Statistic     | 0.462                  |
| Significance p-value       | 0.0                    |
| Pearson Statistic          | 0.467                  |
| Pearson p-value            | 0.0                    |
| Accuracy                   | 0.779                  |

### Ablation Study: Performance without ResNet

| Dataset    | Significance Statistic | Significance p-value  | Pearson Statistic  | Pearson p-value     | Accuracy |
|------------|------------------------|-----------------------|--------------------|---------------------|----------|
| BAID       | 0.099                  | 2.77e-15              | 0.103              | 1.54e-16            | 0.238    |
| form_data1 | -0.318                 | 0.121                 | -0.294             | 0.154               | 0.240    |
| form_data2 | -0.305                 | 0.139                 | -0.278             | 0.179               | 0.080    |
| form_data3 | -0.318                 | 0.121                 | -0.295             | 0.152               | 0.320    |

### Performance without VGG

| Dataset    | Significance Statistic | Significance p-value | Pearson Statistic  | Pearson p-value     | Accuracy |
|------------|------------------------|----------------------|--------------------|---------------------|----------|
| BAID       | 0.281                  | 4.32e-116            | 0.275              | 1.04e-111           | 0.761    |
| form_data1 | 0.069                  | 0.745                | -0.015             | 0.942               | 0.200    |
| form_data2 | 0.099                  | 0.637                | 0.082              | 0.696               | 0.040    |
| form_data3 | 0.069                  | 0.745                | -0.015             | 0.945               | 0.360    |


## Model Performance on Artwork

We tested our model with selected artworks to compare the initial aesthetic scores with the scores predicted by our model. Below are the results that illustrate how our model performs with real-world artistic images.

### Artwork 1

<img src="https://github.com/samuelribaric/BAID-Reproducibility/docs/15418.jpg" alt="Artwork Analysis 1" width="600">

- **Initial Score:** 5.6506
- **Model's Prediction:** 4.303

### Artwork 2

<img src="https://github.com/samuelribaric/BAID-Reproducibility/docs/61963.jpg" alt="Artwork Analysis 2" width="600">

- **Initial Score:** 5.3349
- **Model's Prediction:** 4.660

### Artwork 3

<img src="https://github.com/samuelribaric/BAID-Reproducibility/docs/49509.jpg" alt="Artwork Analysis 3" width="600">

- **Initial Score:** 7.8236
- **Model's Prediction:** 5.466


## Discussion

### Unusual Findings and Their Implications

#### Earlier Training Stages Yielded Better Results
One notable observation from our reproduction study was that earlier training epochs, particularly Epoch 39, showed slightly better performance in terms of accuracy compared to the final model at Epoch 99. This could suggest potential overfitting as the training progressed, or it might indicate that the model reaches an optimal state of learning earlier than anticipated. This finding prompts a reevaluation of the training duration and may lead to more efficient training strategies by implementing early stopping or adjusting learning rates dynamically.

## Detailed Comparison of Ablation Studies

### Overview of Ablation Study Results

Our ablation study results provided intriguing insights, particularly when contrasted with those from the original paper. In the original study, the removal of various model components generally led to a decrease in performance, but the overall impact was relatively moderate. This suggests that while each component contributes to the model's effectiveness, the architecture is robust enough to maintain reasonable performance even when individual elements are disabled.

### Specific Findings from Our Study

#### Impact of Removing VGG and ResNet
- **Without VGG**: The performance of our model showed minimal impact when the VGG, responsible for style-specific feature extraction, was removed. This was somewhat unexpected given the importance typically attributed to style in artistic assessments, but it indicates that the remaining components of the model can compensate effectively for the loss of style-specific input.
  
- **Without ResNet**: Contrary to the minimal impact seen with the removal of VGG, eliminating ResNet resulted in a significant drop in all performance metrics. This highlights ResNet's critical role in extracting generic aesthetic features that are essential for the model's overall performance. Our results underscore the importance of this component more dramatically than the original study, suggesting that our model may be more reliant on generic aesthetic features, or possibly that our dataset emphasizes aspects of aesthetics that are particularly sensitive to the contributions of ResNet.

### Comparative Analysis with Original Study
The original paper's ablation results showed a systematic decline across metrics when removing components, but none as dramatic as our findings with the removal of ResNet. This could indicate differences in model dependency on certain features, or it might reflect variations in dataset characteristics where certain features play a more pivotal role.

#### Insights into Model Resilience and Component Dependency
- **Model Resilience**: The original SAAN model demonstrated a degree of resilience, managing to retain much of its effectiveness despite the removal of significant components. This suggests a well-integrated architecture where components can somewhat compensate for each other.
  
- **Component Dependency**: Our findings suggest a higher dependency on ResNet within our model's architecture compared to the original study. This dependency could be reflective of the nuances in how aesthetic features are processed and utilized within different implementations or adaptations of the model.

### Broader Implications
- **Understanding Component Roles**: These ablation studies are crucial for understanding the distinct roles and importance of various model components. They inform not only the design of more efficient and targeted models but also help in fine-tuning existing models for specific applications or datasets.

- **Future Model Improvements**: The insights from these studies could guide future modifications to the SAAN model, such as exploring alternative or additional components that could enhance style or generic aesthetic feature extraction without compromising other aspects of performance.

- **Expanding Ablation Studies**: Future work should consider more granular ablation studies that not only remove entire branches but also test the impact of varying layers or subsets of features within those branches. This could provide deeper insights into the model's operational dynamics and offer more precise guidance for enhancements.

### Concluding Thoughts
Both our study and the original paper highlight the complex interplay between different components in determining the performance of aesthetic assessment models. As we continue to push the boundaries of what these models can achieve, it becomes increasingly important to understand not just what each part does, but how they work together to produce the final assessment outcome. Our findings add a valuable perspective to the ongoing discussion about the optimization and application of aesthetic assessment models in real-world scenarios.



### Implications of New Data Results

#### Challenges with New Datasets
Testing the model on new datasets composed of images rated via a Google Form revealed lower performance. This could be attributed to several factors:
- **Dataset Quality and Size**: The new dataset was smaller and not curated with the same rigor as the BAID dataset.
- **Voting Bias**: The voting mechanism allowed participants to rate images without comparison, potentially skewing the results towards more favorable outcomes.

This highlights the challenges in generalizing the trained models across different datasets and underscores the importance of dataset design in training robust models.

### Reflection on Methodological Variations

#### Code Revamping Impact
The introduction of a dynamic path management and weight loading system proved crucial for adapting the SAAN model to different computational environments and for facilitating continuous training sessions. These improvements not only enhanced the model's usability across platforms but also prevented potential data loss, thereby making the research process more resilient and efficient.

### Concluding Thoughts

The discrepancies and unexpected findings from our reproduction effort suggest that while the SAAN model is robust under certain conditions, its performance varies significantly with changes in dataset characteristics and model architecture. These insights are invaluable for future research in the field of artistic image aesthetics assessment, providing a clearer path towards refining and adapting these models for broader applications.

Further studies should focus on exploring the impact of different architectural components more systematically and expanding the model's testing across diverse datasets to better understand its generalizability and limitations.



