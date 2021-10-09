package Other;

public final class Constants {
    /**
     * The minimum range at which neuron weights and biases are randomized
     */
    public static final double minNeuronRange = -0.5;
    /**
     * The maximum range at which neuron weights and biases are randomized
     */
    public static final double maxNeuronRange = 0.5;
    /**
     * The constant by which the weight matrix is multiplied by to avoid overshoot
     * of gradient descent
     */
    public static final double learningRate = 0.1;
    /**
     * L2 regulation constant
     */
    public static final double L2regConstant = 0.02;
    /**
     * Can be used to implement mini-batch gradient descent. The amount of times the
     * network will add on weight and bias backpropagation results before taking the
     * average and actually applying them to the weights and biases throughout the
     * network
     */
    public static final int batchSize = 256;
}