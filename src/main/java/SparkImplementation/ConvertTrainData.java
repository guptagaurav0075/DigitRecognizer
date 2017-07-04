package SparkImplementation;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

/**
 * This file is used to convert training data into libsvm format.
 */

public class ConvertTrainData {
	public static void main(String[] args) throws IOException{
		// TODO Auto-generated method stub
		// check command-line args
        if(args.length != 1) {
            System.err.println("usage: SparkImplementation.ConvertTrainData <input-file>");
            System.exit(1);
        }
        String inputFile = args[0];
		BufferedReader br = new BufferedReader(new FileReader(inputFile));
		String line = br.readLine();
        FileWriter out = new FileWriter("train-LIBSVM-format-data.txt");
		while(line!=null){
			if(!Character.isLetter(line.charAt(0))){
				String values[] = line.split(",");
				String outputLine ="";
				outputLine = values[0];
				for(int i=1; i<values.length; i++){
					if(Integer.parseInt(values[i])!=0)
						outputLine += " "+(i-1)+":"+values[i];
				}
				out.write(outputLine+"\n");
				System.out.println(outputLine);
			}
			line=br.readLine();
		}
		System.out.println("Done creating a training file");
		br.close();
		out.close();
	}
}