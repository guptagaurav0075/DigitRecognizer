package SparkImplementation;

/**
 * 
 */
import java.util.HashMap;
import java.util.Map;
import scala.Tuple2;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import java.util.Arrays;
import java.util.Iterator;
import org.apache.spark.util.Utils;
import org.apache.spark.api.java.*;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.SparkConf;


public class KmeansExample {
	public static void main(String[] args) {

	    if(args.length != 2) {
            System.err.println("usage: SparkImplementation.DecisionTreeImplementation <input-file-data> <output-file-directory>");
            System.exit(1);
        }
        // Create Java Spark Context
	    SparkConf conf = new SparkConf().setAppName("K-means Example");
	    JavaSparkContext sc = new JavaSparkContext(conf);
        
        // Load  input data.
        String inputFile = args[0];
        String outputFile=args[1];
        
        //JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(jsc.sc(), inputFile).toJavaRDD();
        JavaRDD<String> data = sc.textFile(inputFile);

        // Split the data into training and test sets (30% held out for testing)        
        JavaRDD<String>[] splits = data.randomSplit(new double[]{0.7, 0.3});
        JavaRDD<Vector> trainingData = splits[0].map(
      	      new Function<String, Vector>() {
      	        public Vector call(String s) {
      	          String[] sarray = s.split(" ");
      	          double[] values = new double[sarray.length];
      	          for (int i = 0; i < sarray.length; i++)
      	            values[i] = Double.parseDouble(sarray[i]);
      	          return Vectors.dense(values);
      	        }
      	      }
      	    );
        JavaRDD<Vector> testData = splits[1].map(
      	      new Function<String, Vector>() {
      	        public Vector call(String s) {
      	          String[] sarray = s.split(" ");
      	          double[] values = new double[sarray.length];
      	          for (int i = 0; i < sarray.length; i++)
      	            values[i] = Double.parseDouble(sarray[i]);
      	          return Vectors.dense(values);
      	        }
      	      }
      	    );
        
	    

	    // Cluster the data into two classes using KMeans
	    int numClusters = 10;
	    int numIterations = 20;
	    KMeansModel clusters = KMeans.train(trainingData.rdd(), numClusters, numIterations);

	    // Evaluate clustering by computing Within Set Sum of Squared Errors
	    double WSSSE = clusters.computeCost(trainingData.rdd());
	    System.out.println("Within Set Sum of Squared Errors = " + WSSSE);
	    JavaRDD<Integer> output = clusters.predict(testData);
	    output.saveAsTextFile(outputFile);
	  }
}

