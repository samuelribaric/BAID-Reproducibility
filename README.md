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

- **Impact of Refactoring**:


#### New Dataset
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

##Ablation Study##


## Results

This section provides an overview of the outcomes from our project efforts, including reproducing the original paper's results, evaluating new datasets, and conducting an ablation study with modified model configurations.
### Reproduced Results

#### Initial Test with Weights from Authors

| Metric                     | Value                  |
|----------------------------|------------------------|
| Significance Statistic     | 0.4721614384229233     |
| Significance p-value       | 0.0                    |
| Pearson Statistic          | 0.466993115611886      |
| Pearson p-value            | 0.0                    |
| Accuracy                   | 0.7673073917799657     |

#### Epoch 39 (First training)

| Metric                     | Value                  |
|----------------------------|------------------------|
| Significance Statistic     | 0.46154713882667087    |
| Significance p-value       | 0.0                    |
| Pearson Statistic          | 0.4676182473919227     |
| Pearson p-value            | 0.0                    |
| Accuracy                   | 0.7798093452101891     |

#### Epoch 99 (Finished Training)

| Metric                     | Value                  |
|----------------------------|------------------------|
| Significance Statistic     | 0.46196337871776366    |
| Significance p-value       | 0.0                    |
| Pearson Statistic          | 0.4673758142171795     |
| Pearson p-value            | 0.0                    |
| Accuracy                   | 0.7788716987029223     |

### Ablation Study: Performance without ResNet

#### Test on Various Datasets

| Dataset        | Significance Statistic | Significance p-value  | Pearson Statistic  | Pearson p-value     | Accuracy   |
|----------------|------------------------|-----------------------|--------------------|---------------------|------------|
| BAID           | 0.0985474602591078     | 2.769451136158405e-15 | 0.1029179598080305 | 1.5432773274448345e-16 | 0.23816221284575714 |
| form_data1.csv | -0.31844457134806753   | 0.12080277186000737   | -0.2940150947209376| 0.15370386745737993 | 0.24       |
| form_data2.csv | -0.3046153846153846    | 0.13872654983076974   | -0.2776495897322667| 0.1790218015069261  | 0.08       |
| form_data3.csv | -0.31844457134806753   | 0.12080277186000737   | -0.2953613439152013| 0.1517407447601908  | 0.32       |

### Performance without VGG

#### Test on Various Datasets

| Dataset        | Significance Statistic | Significance p-value | Pearson Statistic  | Pearson p-value     | Accuracy   |
|----------------|------------------------|----------------------|--------------------|---------------------|------------|
| BAID           | 0.28059646018021484    | 4.315794067754801e-116| 0.2753714573832638 | 1.0384372460027474e-111 | 0.7610564150648539 |
| form_data1.csv | 0.0685406695283628     | 0.7447714241446419   | -0.015240836022584673 | 0.9423577832017125 | 0.2        |
| form_data2.csv | 0.09923076923076923    | 0.6369836836671996   | 0.08233920024385069| 0.6955871335049199  | 0.04       |
| form_data3.csv | 0.0685406695283628     | 0.7447714241446419   | -0.01461946420573074 | 0.9447042784323637 | 0.36       |

## Discussion

### Unusual Findings and Their Implications

#### Earlier Training Stages Yielded Better Results
One notable observation from our reproduction study was that earlier training epochs, particularly Epoch 39, showed slightly better performance in terms of accuracy compared to the final model at Epoch 99. This could suggest potential overfitting as the training progressed, or it might indicate that the model reaches an optimal state of learning earlier than anticipated. This finding prompts a reevaluation of the training duration and may lead to more efficient training strategies by implementing early stopping or adjusting learning rates dynamically.

### Insights from the Ablation Study

#### Performance without ResNet
The ablation study highlighted significant performance drops when the ResNet component was removed from the model architecture. This underscores ResNet's role in capturing complex patterns in artistic aesthetics, which are possibly pivotal for higher accuracy in assessments. The results demonstrate the importance of deep architectures in handling the nuanced task of image aesthetic assessment.

#### Significance and Pearson Metrics
The consistently low p-values observed in our tests underscore the statistical significance of the results obtained, confirming that the findings are unlikely due to random chance. Meanwhile, the Pearson correlation coefficients provide insights into the linear correlation between model predictions and human judgments, reflecting the model's effectiveness in mimicking human aesthetic assessments.

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



