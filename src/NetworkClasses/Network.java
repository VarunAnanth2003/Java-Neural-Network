package NetworkClasses;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Random;
import java.util.Stack;

import Exceptions.TooFewLayersException;
import Other.Util;
import Other.FunctionClasses.Activation.ActivationFunction;
import Other.FunctionClasses.Activation.ActivationOptions;
import Other.FunctionClasses.Cost.CostFunction;
import Other.FunctionClasses.Cost.CostOptions;

public class Network {

    private Queue<Layer> layerQueue = new LinkedList<>();
    private Layer inputLayer;
    private int numLayers;
    private CostFunction cf;

    public Network(int[] layerNodeCounts, ActivationOptions[] afArr, CostOptions costOp) throws TooFewLayersException {
        numLayers = layerNodeCounts.length;
        if (numLayers < 2) {
            throw new TooFewLayersException();
        }
        for (int i = 0; i < layerNodeCounts.length - 1; i++) {
            int currentNodeAmount = layerNodeCounts[i];
            int nextNodeAmount = layerNodeCounts[i + 1];
            layerQueue.add(new Layer(currentNodeAmount, nextNodeAmount, new ActivationFunction(afArr[i])));
        }
        layerQueue.add(new Layer(layerNodeCounts[layerNodeCounts.length - 1],
                new ActivationFunction(afArr[layerNodeCounts.length - 1])));
        inputLayer = layerQueue.peek();
        cf = new CostFunction(costOp);
    }

    public Network(Queue<Layer> layerQueue, CostFunction cf) throws TooFewLayersException {
        if (layerQueue.size() < 2) {
            throw new TooFewLayersException();
        }
        numLayers = layerQueue.size();
        this.layerQueue = layerQueue;
        inputLayer = layerQueue.peek();
        this.cf = cf;
    }

    /**
     * Populates the first layer of the network with values (between 0.0 and 1.0)
     * that can be pulsed through the network for a result
     * 
     * @param initialValues the values to initialize the function with. SHould be
     *                      the same length as the number of neurons in the initial
     *                      layer
     */
    public void initializeNetwork(double[] initialValues) {
        int counter = 0;
        for (Neuron n : inputLayer.getNeurons()) {
            try {
                n.setVal(initialValues[counter]);
                counter++;
            } catch (ArrayIndexOutOfBoundsException e) {
                e.printStackTrace();
                System.err.println("Too many initial values for base network layer.");
            }
        }
    }

    /**
     * Populates the first layer of the network with <b>[RANDOM]</b> values (between
     * 0.0 and 1.0) that can be pulsed through the network for a result
     */
    public void initializeNetwork() {
        for (Neuron n : inputLayer.getNeurons()) {
            try {
                n.setVal(new Random().nextDouble());
            } catch (ArrayIndexOutOfBoundsException e) {
                e.printStackTrace();
                System.err.println("Too many initial values for base network layer.");
            }
        }
    }

    /**
     * This method will iterate through each layer and use the weights and
     * activations of that layer along with the biases of the next to compute the
     * activations of the next layer and will repeat this process until the last
     * layer, at which point the final layer of Neurons will collectively hold the
     * vector of activations that is the ultimate result of this network
     */
    public void pulse() {
        Layer currentLayer = null;
        Layer nextLayer = null;
        for (int i = 0; i < numLayers - 1; i++) {
            currentLayer = layerQueue.poll();
            nextLayer = layerQueue.peek();
            currentLayer.activateLayer(nextLayer);
            layerQueue.add(currentLayer);
        }
        layerQueue.add(layerQueue.poll());
    }

    /**
     * This method will iterate through each layer and use the weights and
     * activations of that layer along with the biases of the next to compute the
     * activations of the next layer and will repeat this process until the last
     * layer, at which point the final layer of Neurons will collectively hold the
     * vector of activations that is the ultimate result of this network. This
     * vector will also be returned as a double[]
     */
    public double[] pulseWithResult() {
        Layer currentLayer = null;
        Layer nextLayer = null;
        for (int i = 0; i < numLayers - 1; i++) {
            currentLayer = layerQueue.poll();
            nextLayer = layerQueue.peek();
            currentLayer.activateLayer(nextLayer);
            layerQueue.add(currentLayer);
        }
        layerQueue.add(layerQueue.poll());
        double[] ret_val = new double[nextLayer.getNeurons().size()];
        int counter = 0;
        for (Neuron n : nextLayer.getNeurons()) {
            ret_val[counter] = n.getVal();
            counter++;
        }
        return ret_val;
    }

    /**
     * Populates the first layer of the network with values (between 0.0 and 1.0)
     * that are then pulsed through the network for a result.This method will
     * iterate through each layer and use the weights and activations of that layer
     * along with the biases of the next to compute the activations of the next
     * layer and will repeat this process until the last layer, at which point the
     * final layer of Neurons will collectively hold the vector of activations that
     * is the ultimate result of this network
     */
    public void pulseWithInput(double[] initialValues) {
        int counter = 0;
        for (Neuron n : inputLayer.getNeurons()) {
            try {
                n.setVal(initialValues[counter]);
                counter++;
            } catch (ArrayIndexOutOfBoundsException e) {
                e.printStackTrace();
                System.err.println("Too many initial values for base network layer.");
            }
        }
        Layer currentLayer = null;
        Layer nextLayer = null;
        for (int i = 0; i < numLayers - 1; i++) {
            currentLayer = layerQueue.poll();
            nextLayer = layerQueue.peek();
            currentLayer.activateLayer(nextLayer);
            layerQueue.add(currentLayer);
        }
        layerQueue.add(layerQueue.poll());
    }

    /**
     * Uses an expected value along with backpropagation calculus to adjust the
     * weights and biases of the network to optimize the cost function towards zero.
     * <b> This is where the learning happens! </b>
     * 
     * @param expected the expected output of the neural network for a certain
     *                 training example
     */
    public void learnFrom(double[] expected) {
        Stack<Layer> stackLayers = new Stack<>();
        for (Layer l : layerQueue) {
            stackLayers.push(l);
        }
        Layer curLayer = stackLayers.pop();

        double[] dCda = cf.getFunction().calculateDerivative(curLayer.getValuesAsVector(), expected);
        do {
            double[] dCdz = Util.calculatedCdz(curLayer.getValuesAsVector(), dCda, curLayer.getActivationFunction());
            double[][] dCdW = Util.outerProduct(dCdz, stackLayers.peek().getValuesAsVector());
            curLayer.addBiasDeltas(dCdz);
            stackLayers.peek().addWeightDeltas(dCdW);
            dCda = Util.calculatedCda(stackLayers.peek().getWeightsAsMatrix(), dCdz);
            curLayer = stackLayers.pop();
        } while (!stackLayers.empty());

    }

    /**
     * Adjusts the weights and biases for all layers in this network. Can be
     * implemented with a batch size of 1 for Stochastic gradient descent. Other
     * uses will either be mini batch or batch gradient descent
     */
    public void updateLayers() {
        for (Layer l : layerQueue) {
            l.adjustWB();
        }
    }

    public Queue<Layer> getLayers() {
        return layerQueue;
    }

    /**
     * Saves this network's weights and biases to a text file at the location passed
     * into this method. The formatting for the data can be found by viewing a file
     * or taking a look at the README of this project
     * 
     * @param f the file location where the network will be saved
     */
    public void saveToFile(File f) {
        try {
            f.createNewFile();
            FileWriter w = new FileWriter(f);
            w.write(this.cf.getCo() + "\n");
            w.write(layerQueue.size() + "\n");
            for (Layer l : layerQueue) {
                w.write(l.getNeurons().size() + " " + l.getNextNodesNum() + " ");
                w.write(l.getActivationFunction().getAo() + "\n");
                for (Neuron n : l.getNeurons()) {
                    Arrays.stream(n.getWeights()).forEach(value -> {
                        try {
                            w.write(value + " ");
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                    });
                    w.write("\n" + n.getBias() + "\n");
                }
            }
            w.flush();
            w.close();
            System.out.println("Saved to " + f.getAbsolutePath());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Saves this network's weights and biases to a text file at a default location
     * (Saved Networks) with time in milliseconds since epoch as the name. The
     * formatting for the data can be found by viewing a file or taking a look at
     * the README of this project
     * 
     * @param f the file location where the network will be saved
     */
    public void saveToFile() {
        System.out.println("Saving...");
        try {
            File f = new File("Saved Networks\\" + System.currentTimeMillis() + ".txt");
            f.createNewFile();
            FileWriter w = new FileWriter(f);
            w.write(this.cf.getCo() + "\n");
            w.write(layerQueue.size() + "\n");
            for (Layer l : layerQueue) {
                w.write(l.getNeurons().size() + " " + l.getNextNodesNum() + " ");
                w.write(l.getActivationFunction().getAo() + "\n");
                for (Neuron n : l.getNeurons()) {
                    Arrays.stream(n.getWeights()).forEach(value -> {
                        try {
                            w.write(value + " ");
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                    });
                    w.write("\n" + n.getBias() + "\n");
                }
            }
            w.flush();
            w.close();
            System.out.println("Saved to " + f.getAbsolutePath());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public CostFunction getCostFunction() {
        return cf;
    }

    @Override
    public String toString() {
        String ret_val = "";
        for (Layer l : layerQueue) {
            ret_val += l.toString();
            ret_val += ("\n---\n");
        }
        return ret_val;
    }
}