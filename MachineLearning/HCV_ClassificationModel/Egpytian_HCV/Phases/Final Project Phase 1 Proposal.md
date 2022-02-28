> Team Members: Ahmad Abdalmageed, Mohamed Almotasem

# Final Project Phase 1 Proposal 

## Problem 

The highest prevalence of HCV infection is present in Egypt, with 92.5% of patients infected with genotype 4, 3.6% patients with genotype 1, 3.2% patients with multiple genotypes, and < 1% patients with other genotypes. Grading the Virus has an impact on the treatment given to the patient this is where Machine Learning Steps in.

## Dataset 

Recorded by Ain Shams University, this [Data](https://archive.ics.uci.edu/ml/datasets/Hepatitis+C+Virus+%28HCV%29+for+Egyptian+patients) provides records of patients who underwent treatment dosages for HCV about 18 months. Data consists of 29 Attributes and 1385 instances. The Target Variable is a Discretized grade of the patient's infection degree.

### Description 

**Age**: (Numerical)Age Recorded for each patient.

**Gender**: (Categorical) Gender of Each

**BMI (Body Mass Index)**: (Numerical)  Measure of body fat based on height and weight

**Fever** :(Categorical) Present or Absent 

**Nausea/Vomting**:(Categorical) Present or Absent

**Headache** :(Categorical) Present or Absent

**Diarrhea** : (Categorical) Present or Absent

**Fatigue & generalized bone ache**: (Categorical) Present or Absent 

**Jaundice**: (Categorical) Present or Absent

**Epigastric pain**: (Categorical) Present or Absent

**WBC (White blood cell)**: (Numerical) Count of WBCs

**RBC (Red blood cells)**: (Numerical) Count of WBCs

**HGB (Hemoglobin)**:(Numerical) The range of hemoglobin in blood, it's normal range is for men, 13.5 to 17.5 grams per deciliter. for women, 12.0 to 15.5 grams per deciliter.

**Plat (Platelets)**: (Numerical) The count of blood platelets found. A normal platelet count ranges from 150,000 to 450,000 platelets per microliter of blood.

**AST 1 (aspartate transaminase ratio)**: (Numerical) Measures the amount of AST enzyme in the blood. AST is normally found in red blood cells, liver, heart, muscle tissue, pancreas, and kidneys. Typically the range for normal AST is reported between 10 to 40 units per liter.

**ALT 1 (alanine transaminase ratio 1 week)**: (Numerical) Measures the amount of ALT enzyme in the blood. it is an enzyme found mostly in the cells of the liver and kidney the normal ALT range is between 7 to 56 units per liter.

**ALT 4 (alanine transaminase ratio 12 weeks)**: (Numerical)

**ALT 12 (alanine transaminase ratio 4 weeks)**:(Numerical)

**ALT 24 (alanine transaminase ratio 24 weeks)**: (Numerical)

**ALT 36 (alanine transaminase ratio 36 weeks)**: (Numerical)

**ALT 48 (alanine transaminase ratio 48 weeks)**: (Numerical)

**ALT (after 24 w alanine transaminase ratio 24 weeks)**: (Numerical)

**RNA Base**: (Numerical) The count of HCV RNA PCR test is used to determine whether the hepatitis C virus exists in your bloodstream.

**RNA 4**:(Numerical)

**RNA 12**: (Numerical)

**RNA EOT (end-of-treatment)**: (Numerical) 

**RNA EF (Elongation Factor)**: (Numerical)

**Baseline Histological Grading**: (Categorical) Is meant to reflect how quickly the disease is progressing to the end of it's current stage.

**Baseline Histological staging**: (Categorical) It's a measure of how far the disease has progressed in its natural history, where at the end stage the organ fails. there are 4 stages: Portal fibrosis - Peri portal fibrosis - Septal fibrosis - Cirrhosis.

## Proposed Methodology 

1. **Data Preprocessing**: Exploring the data for potential problems
   - Data Cleaning: Removing potential problems in the data and make the data more tidy, some of the attributes are categorical with values 1 and 2 which may induce error into the model.
2. **Exploratory Data Analysis**: Exploring the Data for Insightful Characteristics 
   - Plotting the Relations between the Attributes and the Target variables which would explain more about their Relationships and Correlations 
3. **Prototyping**: Instantiate a basic Model from potential ML Algorithms and find best Candidates
   - Run a very basic Classification Algorithms on Clean Data
   - Extract a few best Performing Models and move to the next phase
4. **Fine Tunning**: Taking the best performing models and tune their parameters to drive the model to a better performance.
5. **Feature Engineering**: Given the Data Exploration extract new information or edit the information given by our attributes to give better performance.

