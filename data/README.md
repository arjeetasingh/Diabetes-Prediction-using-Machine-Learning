# Data

`diabetes.csv` is the **PIMA Indians Diabetes** dataset.

* 768 women of Pima Indian heritage, age 21 or older
* Collected by the National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK)
* Target `Outcome`: 1 = diabetes within 5 years (268 patients, 34.9%), 0 = no diabetes (500 patients)

| Column | Meaning |
|---|---|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose 2 hours into an oral glucose tolerance test (mg/dL) |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin-fold thickness (mm) |
| Insulin | 2-hour serum insulin (mu U/ml) |
| BMI | Body mass index (kg/m2) |
| DiabetesPedigreeFunction | Score of diabetes family history |
| Age | Years |

## The hidden missing values

A `0` in **Glucose, BloodPressure, SkinThickness, Insulin and BMI** is physiologically impossible. It means the value was not measured. The file contains 5, 35, 227, 374 and 11 such zeros respectively. (A zero in `Pregnancies` is real.) The code in `diabetes_ml/preprocessing.py` converts these zeros to missing values and fills them inside each cross-validation fold.

## Source and citation

The dataset is publicly available and widely mirrored (UCI / Kaggle / GitHub copies are the same 768 rows). If you replace the file, keep the same column names.

Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., Johannes, R.S. (1988).
*Using the ADAP learning algorithm to forecast the onset of diabetes mellitus.*
Proceedings of the Symposium on Computer Applications and Medical Care, 261-265.

Check the terms of the copy you download before redistributing it.
