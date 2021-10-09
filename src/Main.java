import java.awt.image.*;
import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Random;

import javax.imageio.ImageIO;

import Exceptions.TooFewLayersException;
import NetworkClasses.Network;
import Other.Constants;
import Other.DataGenerator;
import Other.Util;
import Other.FunctionClasses.Activation.ActivationOptions;
import Other.FunctionClasses.Cost.CostFunction;
import Other.FunctionClasses.Cost.CostOptions;

public class Main {
    public static HashMap<OutputProfiles, double[][][]> trainingData = new HashMap<>();

    // TODO: implement momentum cleanly
    // TODO: Use MNIST handwriting to test NN
    public static void main(String[] args) throws TooFewLayersException {
        // Data prep
        prepareData();

        // Training
        System.out.println("Training...");
        Network n = new Network(new int[] { 81, 57, 4 }, new ActivationOptions[] { ActivationOptions.SIGMOID,
                ActivationOptions.SIGMOID, ActivationOptions.LEAKY_RE_LU }, CostOptions.QUADRATIC);
        for (int i = 0; i < Constants.batchSize * 1000; i++) {
            OutputProfiles op = OutputProfiles.getRandomProfile();
            n.pulseWithInput(Util.flattenArr(trainingData.get(op)[new Random().nextInt(100000)]));
            n.learnFrom(op.getProfile());
            if (i % Constants.batchSize == 0)
                n.updateLayers();
        }
        System.out.println("Training Complete!");

        // Network write/read
        n.saveToFile(new File("src\\Saved Networks\\MyNetwork.txt"));
        System.out.println("Reading...");
        n = Util.readFromFile(new File("src\\Saved Networks\\MyNetwork.txt"));
        System.out.println("Read Complete");

        // Testing
        testNN(n, new File("src\\A.png"));
        testNN(n, new File("src\\B.png"));
        testNN(n, new File("src\\C.png"));
        testNN(n, new File("src\\D.png"));
    }

    /**
     * Tests the neural network passed in by "n" against the associated image within
     * the File passed in by "fileToRead". For this case, only 9x9 neural networks
     * can be tested
     * 
     * @param n
     * @param fileToRead
     */
    public static void testNN(Network n, File fileToRead) {
        try {
            BufferedImage image = ImageIO.read(fileToRead);
            double[][] data = new double[image.getHeight()][image.getWidth()];
            for (int i = 0; i < data.length; i++) {
                for (int j = 0; j < data[i].length; j++) {
                    int rgb = image.getRGB(j, i);
                    int r = (rgb >> 16) & 0xFF;
                    int g = (rgb >> 8) & 0xFF;
                    int b = (rgb & 0xFF);
                    data[j][i] = Math.abs((((r + g + b) / 3) / 255) - 1);
                }
            }
            n.initializeNetwork(Util.flattenArr(data));
            double[] result = n.pulseWithResult();
            System.out.println("This is a: " + OutputProfiles.getBestProfile(result, n.getCostFunction()));
            System.out.print("NN Output: ");
            Arrays.stream(result).forEach(e -> System.out.print(new DecimalFormat("0.00").format(e) + " "));
            System.out.println();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Prepares and populates the "trainingData HashSet with randomized matrixes as
     * created by the DataGenerator class"
     * 
     * @see DataGenerator
     */
    public static void prepareData() {
        int dataNum = 100000;
        double[][][] aData = new double[dataNum][][];
        for (int i = 0; i < aData.length; i++) {
            aData[i] = DataGenerator.generateA();
        }
        trainingData.put(OutputProfiles.SQUARE, aData);

        double[][][] bData = new double[dataNum][][];
        for (int i = 0; i < bData.length; i++) {
            bData[i] = DataGenerator.generateB();
        }
        trainingData.put(OutputProfiles.DIAMOND, bData);

        double[][][] cData = new double[dataNum][][];
        for (int i = 0; i < cData.length; i++) {
            cData[i] = DataGenerator.generateC();
        }
        trainingData.put(OutputProfiles.PLUS, cData);

        double[][][] dData = new double[dataNum][][];
        for (int i = 0; i < dData.length; i++) {
            dData[i] = DataGenerator.generateD();
        }
        trainingData.put(OutputProfiles.CROSS, dData);

    }

    /**
     * An enum that holds "optimal" neural network outputs for multiple images,
     * along with methods that facilitate data input/comparison
     */
    enum OutputProfiles {
        SQUARE(new double[] { 1, 0, 0, 0 }), DIAMOND(new double[] { 0, 1, 0, 0 }), PLUS(new double[] { 0, 0, 1, 0 }),
        CROSS(new double[] { 0, 0, 0, 1 });

        private double[] profile;

        public double[] getProfile() {
            return profile;
        }

        private OutputProfiles(double[] profile) {
            this.profile = profile;
        }

        /**
         * Selects a random enum value and returns it
         * 
         * @return a random value from the possible values OutputProfiles can hold
         */
        public static OutputProfiles getRandomProfile() {
            int randSel = new Random().nextInt(4);
            if (randSel == 0) {
                return OutputProfiles.SQUARE;
            } else if (randSel == 1) {
                return OutputProfiles.DIAMOND;
            } else if (randSel == 2) {
                return OutputProfiles.PLUS;
            } else {
                return OutputProfiles.CROSS;
            }
        }

        /**
         * Compares the result from a neural network pulse to values within
         * OutputProfiles to get the profile the network is most confident in
         * identifying
         * 
         * @param result is the result of pulsing a neural network that needs to be
         *               compared to the values in OutputProfiles
         * @return is the OutputProfile that the neural network is most confident is the
         *         "true" classification of the input
         */
        public static OutputProfiles getBestProfile(double[] result, CostFunction cf) {
            OutputProfiles[] desiredValues = OutputProfiles.values();
            double costArr[] = new double[desiredValues.length];
            for (int i = 0; i < costArr.length; i++) {
                costArr[i] = cf.getFunction().calculateOriginal(result, desiredValues[i].getProfile());
            }
            double min = Double.POSITIVE_INFINITY;
            int minIndex = -1;
            for (int i = 0; i < costArr.length; i++) {
                if (costArr[i] < min) {
                    min = costArr[i];
                    minIndex = i;
                }
            }
            return desiredValues[minIndex];
        }
    }
}