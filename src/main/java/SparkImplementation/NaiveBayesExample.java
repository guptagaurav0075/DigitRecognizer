package SparkImplementation;

import scala.Tuple2;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.SparkConf;

/*
 * Dataset used is from Kaggle
 * https://www.kaggle.com/c/digit-recognizer/data
 * 
 * Scala implementation is also illustrated in comment block after NaiveBayesExample class completion
 */

public class NaiveBayesExample {
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		if(args.length != 2) {
			System.err.println("usage: SparkImplementation.NaiveBayesImplementation <input-file-training-data> <output-file-directory>");
			System.exit(1);
		}
		String inputFile = args[0];
		String outputFile=args[1];
		
		// Create Java Spark Context
		SparkConf sparkConf = new SparkConf().setAppName("JavaNaiveBayesExample");
		JavaSparkContext jsc = new JavaSparkContext(sparkConf);
		// Load  input data.
        JavaRDD<LabeledPoint> inputData = MLUtils.loadLibSVMFile(jsc.sc(), inputFile).toJavaRDD();
		JavaRDD<LabeledPoint>[] tmp = inputData.randomSplit(new double[]{0.6, 0.4});
		JavaRDD<LabeledPoint> training = tmp[0]; // training set
		JavaRDD<LabeledPoint> test = tmp[1]; // test set
		
        final NaiveBayesModel model = NaiveBayes.train(training.rdd(), 1.0);
		
        JavaPairRDD<Double, Double> predictionAndLabel =
        		test.mapToPair(new PairFunction<LabeledPoint, Double, Double>() {
        		    @Override
        		    public Tuple2<Double, Double> call(LabeledPoint p) {
        		      return new Tuple2<>(model.predict(p.features()), p.label());
        		    }
        		  });
     
        // Save the predicted output in text file at output location mentioned in argument
        predictionAndLabel.saveAsTextFile(outputFile);
        
        // To calculate the accuracy, based on ouput and label
        double accuracy = predictionAndLabel.filter(new Function<Tuple2<Double, Double>, Boolean>() {
        		  @Override
        		  public Boolean call(Tuple2<Double, Double> pl) {
        		    return pl._1().equals(pl._2());
        		  }
        		}).count() / (double) test.count();

        System.out.println("Test Error: " + (1-accuracy));
        System.out.println("Accuracy: "+ accuracy);
        		
        // Save and load model
        model.save(jsc.sc(), "/user/user01/labs/Spark/DigitReconizer/myNaiveBayesModel");
		NaiveBayesModel sameModel = NaiveBayesModel.load(jsc.sc(), "/user/user01/labs/Spark/DigitReconizer/myNaiveBayesModel");
	
		jsc.stop();
  
	}
	/*
	 * import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.util.MLUtils

// Load and parse the data file.
val data = MLUtils.loadLibSVMFile(sc, "
/FileStore/tables/djkggqkz1498071259645/train_LIBSVM_format_data-4ff9b.txt")

// Split data into training (60%) and test (40%).
val Array(training, test) = data.randomSplit(Array(0.6, 0.4))

val model = NaiveBayes.train(training, lambda = 1.0, modelType = "multinomial")

val predictionAndLabel = test.map(p => (model.predict(p.features), p.label))
val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / test.count()

// Save and load model
model.save(sc, "target/tmp/myNaiveBayesModel")
val sameModel = NaiveBayesModel.load(sc, "target/tmp/myNaiveBayesModel")
	 */
}
