# DigitRecognizer

Hand writing recognition is an important challenge in tech industry and holds variety of uses.
The goal of the Digit Recognition problem is to identify accurately the digits through 0 to 9 from a dataset containing thousands of handwritten images.

Currently the dataset used to train and test the images are from Handwritten Digit Recognition dataset from MNIST ("Modified National Institute of Standards and Technology") http://yann.lecun.com/exdb/mnist/

The data files has train.csv and test.csv which contains pixel information for the handwritten digits from 0-9. Each image has 784 pixels, which can be regarded as 28 x 28 pixel image. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel. The value of pixel ranges from 0 to 255 denoting the gray scale value for the image.

In order to train and test the pixels, firstly the images are converted to LIBSVM format. In order to convert the data to LIBSVM, ConvertTestData.java and ConvertTrainData.java are used.

The repository provides the implementation in JAVA for the following algorithm: 
1. Random Forest,
2. Decision Tree,
3. NaiveBayes,
4. Kmeans.

Accuracy achieved for the models corresponding to different paramenters could be checked in Algorithm Analysis.xlsx.
