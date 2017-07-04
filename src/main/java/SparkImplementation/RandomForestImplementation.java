/**
 * Implementation of Digit recoginzer using Random forest implementation.
 */
package SparkImplementation;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;

import scala.Tuple2;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.util.Utils;

/*
 * In this code, numTrees, maxBins, maxDepth and impurity are tunable parameters.
 * Currently parameters are tuned to result best accuracy for the given dataset
 * Dataset used is from Kaggle
 * https://www.kaggle.com/c/digit-recognizer/data
 * 
 */
public class RandomForestImplementation {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		if(args.length != 2) {
            System.err.println("usage: SparkImplementation.RandomForestImplementation <input-file-data> <output-file-directory>");
            System.exit(1);
        }
        // Create Java Spark Context
        SparkConf conf = new SparkConf().setAppName("RandomForestDigitRecognition");
        JavaSparkContext jsc = new JavaSparkContext(conf);
        
        // Load  input data.
        String inputFile = args[0];
        String outputFile=args[1];
        
        JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(jsc.sc(), inputFile).toJavaRDD();

        // Split the data into training and test sets (30% held out for testing)        
        JavaRDD<LabeledPoint>[] splits = data.randomSplit(new double[]{0.7, 0.3});
        JavaRDD<LabeledPoint> trainingData = splits[0];
        JavaRDD<LabeledPoint> testData = splits[1];

        // Train a RandomForest model.
        // Empty categoricalFeaturesInfo indicates all features are continuous.
     
        Integer numClasses = 10;
        HashMap<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
        Integer numTrees = 28; 
        String featureSubsetStrategy = "auto"; // Let the algorithm choose.
        String impurity = "gini";
        Integer maxDepth = 7;
        Integer maxBins = 32;
        Integer seed = Utils.random().nextInt();

        final RandomForestModel model = RandomForest.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins, seed);


        // Evaluate model on test instances and compute test error
        
        JavaPairRDD<Double, Double> predictionAndLabel =
        testData.mapToPair(new PairFunction<LabeledPoint, Double, Double>() {
         @Override
         public Tuple2<Double, Double> call(LabeledPoint p) {
           return new Tuple2<>(model.predict(p.features()), p.label());
         }
        });
        
        // Save the predicted output in text file at output location mentioned in argument
        predictionAndLabel.saveAsTextFile(outputFile);
        
        // To calculate the accuracy, based on ouput and label
        Double testErr =
        		  1.0 * predictionAndLabel.filter(new Function<Tuple2<Double, Double>, Boolean>() {
        		    @Override
        		    public Boolean call(Tuple2<Double, Double> pl) {
        		      return !pl._1().equals(pl._2());
        		    }
        		  }).count() / testData.count();
        		
        System.out.println("Test Error: " + testErr);
        System.out.println("Accuracy: "+ (1-testErr));
        System.out.println("Learned classification forest model:\n" + model.toDebugString());
        		
        // Save and load model
        model.save(jsc.sc(), "/user/user01/tmp/myRandomForestClassificationModel");
        RandomForestModel sameModel = RandomForestModel.load(jsc.sc(),
        		  "/user/user01/tmp/myRandomForestClassificationModel");
        
        jsc.close();

	}

}
/*
 * The below code is done in scala, 
 * In this code, numTrees, maxBins, maxDepth and impurity are tunable parameters.  
 * 
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.util.MLUtils

// Load and parse the data file.
val data = MLUtils.loadLibSVMFile(sc, "/FileStore/tables/djkggqkz1498071259645/train_LIBSVM_format_data-4ff9b.txt")
// Split the data into training and test sets (30% held out for testing)
val splits = data.randomSplit(Array(0.7, 0.3))
val (trainingData, testData) = (splits(0), splits(1))

// Train a RandomForest model.
// Empty categoricalFeaturesInfo indicates all features are continuous.
val numClasses = 10
val categoricalFeaturesInfo = Map[Int, Int]()
val numTrees = 28 
val featureSubsetStrategy = "auto" // Let the algorithm choose.
val impurity = "entropy"
val maxDepth = 7
val maxBins = 32

val model = RandomForest.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
  numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

// Evaluate model on test instances and compute test error
val labelAndPreds = testData.map { point =>
  val prediction = model.predict(point.features)
  (point.label, prediction)
}
val testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / testData.count()
println("Test Error = " + testErr)
println("Accu
println("Learned classification forest model:\n" + model.toDebugString)

// to print the pridiction and label
learnAndPreds.collect().foreach(println)

// Save and load model
model.save(sc, "/tmp/myRandomForestClassificationModel")
val sameModel = RandomForestModel.load(sc, "/tmp/myRandomForestClassificationModel")
 */
