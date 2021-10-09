package Other;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Scanner;
import java.util.stream.Stream;

import Exceptions.TooFewLayersException;
import NetworkClasses.Layer;
import NetworkClasses.Network;
import NetworkClasses.Neuron;
import Other.FunctionClasses.Activation.ActivationFunction;
import Other.FunctionClasses.Cost.CostFunction;

public class Util {

    /**
     * Turns a 2D array into a readable String
     * 
     * @param arr is the array to stringify
     * @return the stringified array
     */
    public static String stringify2DArr(double[][] arr) {
        String ret_val = "";
        for (int i = 0; i < arr.length; i++) {
            ret_val += i + ": " + Arrays.toString(arr[i]) + "\n";
        }
        return ret_val;
    }

    /**
     * Calculates the average error within an array of errors represented by doubles
     * 
     * @param error is the error array to evaluate
     * @return the average of the error array
     */
    public static double getAvgError(double[] error) {
        double sum = 0;
        for (int i = 0; i < error.length; i++) {
            sum += error[i];
        }
        return (sum / error.length);
    }

    /**
     * Flattens a 2D matrix into a 1D array
     * 
     * @param arr the 2D double matrix to flatten
     * @return the flattened matrix
     */
    public static double[] flattenArr(double[][] arr) {
        double[] ret_val = Stream.of(arr).flatMapToDouble(Arrays::stream).toArray();
        return ret_val;
    }

    /**
     * Calculates the partial derivative of the cost with respect to the raw
     * activation as a function called z (w*a+b)
     * 
     * @param a  activation values of the current layer
     * @param b  the partial derivative of the cost with respect to the current
     *           layer activations
     * @param af the activation function used by the layer
     * @return the partial derivative of the cost with respect to the raw activation
     *         as a function called z (w*a+b)
     */
    public static double[] calculatedCdz(double[] a, double[] b, ActivationFunction af) {
        double[] ret_val = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            ret_val[i] = af.getFunction().calculateDerivative(a[i]) * b[i];
        }
        return ret_val;
    }

    /**
     * Calculates the partial derivative of the cost function with respect to the
     * activations of the previous layer
     * 
     * @param a the weight matrix of the previous layer
     * @param b the dCdz calculation for the current layer as returned by the
     *          "calculatedCdZ" method in the Util class
     * @return the partial derivative of the cost function with respect to the
     *         activations of the previous layer
     * @see Util
     */
    public static double[] calculatedCda(double[][] a, double[] b) {
        double[] ret_val = new double[a.length];
        for (int i = 0; i < ret_val.length; i++) {
            ret_val[i] = dotProduct(a[i], b);
        }
        return ret_val;
    }

    /**
     * Calculates the outer product of two 1D vectors
     * 
     * @param a vector a
     * @param b vector b
     * @return 2D matrix product
     * @see https://en.wikipedia.org/wiki/Outer_product
     */
    public static double[][] outerProduct(double[] a, double[] b) {
        double[][] ret_val = new double[b.length][a.length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < b.length; j++) {
                ret_val[j][i] = a[i] * b[j];
            }
        }
        return ret_val;
    }

    /**
     * Calculates the dot product of two 1D vectors
     * 
     * @param a vector a
     * @param b vector b
     * @return scalar product
     * @see https://en.wikipedia.org/wiki/Dot_product
     */
    public static double dotProduct(double[] a, double[] b) {
        double ret_val = 0;
        for (int i = 0; i < a.length; i++) {
            ret_val += a[i] * b[i];
        }
        return ret_val;
    }

    /**
     * Uses L2 Regularization to prevent a few neurons from dominating the ultimate
     * output of the neural network by proportionally decreasing their influence
     * based on a L2 Regularization constant that can be found in the Constants file
     * 
     * @param arr unregularized matrix
     * @return regularized matrix
     * @see Constants
     */
    public static double[][] l2RegularizeMatrix(double[][] arr) {
        for (int i = 0; i < arr.length; i++) {
            for (int j = 0; j < arr[i].length; j++) {
                arr[i][j] = arr[i][j] - (arr[i][j] * Constants.L2regConstant);
            }
        }
        return arr;
    }

    /**
     * Reads a text file of weights, biases, and other information to reconstruct a
     * pre-trained neural network
     * 
     * @param f the file to read from
     * @return a trained neural network
     */
    public static Network readFromFile(File f) {
        try {
            Queue<Layer> layerQueue = new LinkedList<>();
            BufferedReader b = new BufferedReader(new FileReader(f));
            String costFunctionString = b.readLine();
            int layerAmount = Integer.parseInt(b.readLine());
            for (int i = 0; i < layerAmount; i++) {
                Scanner paramReader = new Scanner(b.readLine());
                Integer nodeNum = paramReader.nextInt();
                Integer nextNodeNum = paramReader.nextInt();
                String activationFunctionString = paramReader.next();
                Queue<Neuron> neuronQueue = new LinkedList<>();
                for (int j = 0; j < nodeNum; j++) {
                    Neuron n = new Neuron(false);
                    Scanner weightBiasReader = new Scanner(b.readLine());
                    double[] weights = new double[nextNodeNum];
                    for (int k = 0; k < nextNodeNum; k++) {
                        weights[k] = weightBiasReader.nextDouble();
                    }
                    double bias = Double.parseDouble(b.readLine());
                    n.setWeights(weights);
                    n.setBias(bias);
                    neuronQueue.add(n);
                    weightBiasReader.close();
                }
                layerQueue.add(new Layer(neuronQueue,
                        ActivationFunction.convertStringToObject(activationFunctionString), nextNodeNum));
                paramReader.close();
            }
            b.close();
            return new Network(layerQueue, CostFunction.convertStringToObject(costFunctionString));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (TooFewLayersException e) {
            e.printStackTrace();
        }
        return null;
    }
}