package NetworkClasses;

import java.util.HashSet;
import java.util.LinkedList;
import java.util.Queue;

import Other.Constants;
import Other.Util;
import Other.FunctionClasses.Activation.ActivationFunction;

public class Layer {
    private Queue<Neuron> neuronQueue = new LinkedList<>();
    private HashSet<double[][]> dWSet = new HashSet<>();
    private HashSet<double[]> dBSet = new HashSet<>();
    private int[] matrixDims = new int[2];
    private boolean hasWeights = false;
    private boolean hasBiases = false;
    private ActivationFunction af;
    private int nextNodesNum = 0;

    public Layer(int nodesNum, int nextNodesNum, ActivationFunction af) {
        for (int i = 0; i < nodesNum; i++) {
            neuronQueue.add(new Neuron(true, nextNodesNum));
        }
        this.af = af;
        this.nextNodesNum = nextNodesNum;
    }

    public Layer(int nodesNum, ActivationFunction af) {
        for (int i = 0; i < nodesNum; i++) {
            neuronQueue.add(new Neuron(true));
        }
        this.af = af;
    }

    public Layer(Queue<Neuron> neuronSet, ActivationFunction af, int nextNodesNum) {
        this.neuronQueue = neuronSet;
        this.af = af;
        this.nextNodesNum = nextNodesNum;
    }

    public Queue<Neuron> getNeurons() {
        return neuronQueue;
    }

    /**
     * Returns all of the values of the neurons in this layer as a double array
     * (vector)
     * 
     * @return the double[] of values of the neurons from this layer
     */
    public double[] getValuesAsVector() {
        double[] ret_val = new double[neuronQueue.size()];
        int counter = 0;
        for (Neuron n : neuronQueue) {
            ret_val[counter] = n.getVal();
            counter++;
        }
        return ret_val;
    }

    /**
     * Returns all of the weights per neuron in this layer as a 2D double array
     * (matrix)
     * 
     * @return the double[][] of weights. Rows represent neuron weight vectors and
     *         each element is a specific weight intended for the connection of a
     *         neuron from one layer to the next
     */
    public double[][] getWeightsAsMatrix() {
        double[][] ret_val = new double[neuronQueue.size()][];
        int counter = 0;
        for (Neuron n : neuronQueue) {
            ret_val[counter] = n.getWeights();
            counter++;
        }
        return ret_val;
    }

    public void setNextNodesNum(int nextNodesNum) {
        this.nextNodesNum = nextNodesNum;
    }

    public int getNextNodesNum() {
        return nextNodesNum;
    }

    public ActivationFunction getActivationFunction() {
        return af;
    }

    /**
     * Uses the values of this layer to edit the values of the next layer within the
     * network. Since this method is not meant to be changed it is assumed that the
     * size of the vectors within the weight matrix of the current layer match up
     * with the amount of Neurons in the next layer. The values of the next layer is
     * determined by a function of the weights and activations of the previous layer
     * and the biases of the next one.
     * 
     * @param nextLayer the layer who's values are edited
     */
    public void activateLayer(Layer nextLayer) {
        int counter = 0;
        for (Neuron b : nextLayer.getNeurons()) {
            double sum = 0;
            for (Neuron a : this.neuronQueue) {
                sum += (a.getVal() * (a.getWeights()[counter]));
            }
            counter++;
            sum = af.getFunction().calculateOriginal(sum + b.getBias());
            b.setVal(sum);
        }
    }

    /**
     * Adds the weight matrix changes calculated by backpropagation to a running
     * total
     * 
     * @param dW the changes to the weight matrix to be added
     */
    public void addWeightDeltas(double[][] dW) {
        if (dWSet.isEmpty()) {
            matrixDims[0] = dW.length;
            matrixDims[1] = dW[0].length;
        }
        dWSet.add(dW);
        hasWeights = true;
    }

    /**
     * Adds the bias matrix changes calculated by backpropagation to a running total
     * 
     * @param dB the changes to the bias matrix to be added
     */
    public void addBiasDeltas(double[] dB) {
        dBSet.add(dB);
        hasBiases = true;
    }

    /**
     * Takes the running totals of the dW and dB sets and averages them. This
     * average (multiplied by a constant that is the learning rate) is then used to
     * affect the weights and biases of this layer
     */
    public void adjustWB() {
        double[][] dW = new double[matrixDims[0]][matrixDims[1]];
        for (double[][] a : dWSet) {
            for (int i = 0; i < a.length; i++) {
                for (int j = 0; j < a[i].length; j++) {
                    dW[i][j] += a[i][j];
                }
            }
        }
        for (int i = 0; i < dW.length; i++) {
            for (int j = 0; j < dW[i].length; j++) {
                dW[i][j] /= dWSet.size();
            }
        }

        double[] dB = new double[neuronQueue.size()];
        for (double[] a : dBSet) {
            for (int i = 0; i < a.length; i++) {
                dB[i] += a[i];
            }
        }
        for (int i = 0; i < dB.length; i++) {
            dB[i] /= dBSet.size();
        }
        int counter = 0;
        if (hasBiases) {
            for (Neuron n : neuronQueue) {
                n.setBias(n.getBias() + (dB[counter] * Constants.learningRate));
                counter++;
            }
        }
        counter = 0;
        if (hasWeights) {
            dW = Util.l2RegularizeMatrix(dW); // L2 Regularization
            for (Neuron n : neuronQueue) {
                for (int i = 0; i < dW[counter].length; i++) {
                    n.setWeight(i, n.getWeights()[i] - (dW[counter][i] * Constants.learningRate));
                }
                counter++;
            }
        }
        dWSet.clear();
        dBSet.clear();
        hasWeights = false;
        hasBiases = false;
    }

    @Override
    public String toString() {
        String ret_val = "";
        for (Neuron n : neuronQueue) {
            ret_val += n.toString();
        }
        return ret_val + "\n";
    }
}