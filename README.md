# LT2222 V23 Assignment 3

file: a3_features.py 
In order to run the program, one has to place the file into a folder containing the data from Enron. 
Then following the steps: 
1. Open terminal
2. cd to the folder containing the python code and the data. 
3. Run the code in the following way: python3 a3_features.py //path to the data// //outputdatafile// //outputauthorsfile// dims //test size//

3.1 For example: python3 a3_features.py /Users/stashakkarainen/Desktop/lt2222-v23-a3/enron_sample output.csv authors.csv 100 --t 20
2 files will be created named "output.csv" and "authors.csv". The first one contains the vectors of emails. The second one contains the names of the people who wrote those emails. 100 stands for dims, and --t 20 means that the test size is 20. 


file: a3_model.py
In order to run the program, one has to place the file into a folder containing the data from Enron. 

Then following the steps: 
1. Open terminal
2. cd to the folder containing the python code and the data. 
3. Run the code in the following way: python3 a3_model.py //outputdatafile// //outputauthorsfile//. 

3.1 For example: python3 a3_model.py outputdata.csv authors.csv.
4. The program also takes values such as the size of the hidden layer, the nonlinearity function, the learning rate and the number of epochs. 
5. In the end, one gets a confusion matrix with the predicted values as X and the true values as Y. 




PART 4.
I believe that there are 2 sides of the situation. Personally, I don't think that it is ethically acceptable to use such data as it reveals someone's personal information without them actively knowing it. While stripping the headers and the sign-offs, I had to take a look into most of the emails, and many of them contained personal business not only the employees but the customers as well, which, to an extent, is not acceptable. 

However, if the following corpus has been released by the court during/after the hearing, then I believe that from the law-defining side of the issue there is no actual problem in using the aforementioned corpus.

PART BONUS A
A file named "f3.py" has been added to the repostitory. This is the code for the bonus part A. the code works just like the code in a3_model.py
