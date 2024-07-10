# Machine Learning Project in Julia
##  Relationships between the timbre of songs and their year

This project is a Julia scripts where we implement the simplest ML models; ***ANN, SVM, kNN, Decision Tree and an Ensemble model of the previous***,  using different songs between 1996 and 2010 selected from: 
[Dataset](https://archive.ics.uci.edu/dataset/203/yearpredictionmsd). 

We will use 90 dimensions related with the timbre mean (12) and timbre covariance (78) of each song for the **input of the models**. Our **target** data is to detect the release date of the song.

***This is part of a Final Project of a Master in AI subject***, all the info and documentation is on the pdf.

## Implementation
For implement the code in this repository is needed to download the data: [Dataset](https://archive.ics.uci.edu/dataset/203/yearpredictionmsd) and locate the `YearPredictionMSD.txt` on a folder `datasets`.
