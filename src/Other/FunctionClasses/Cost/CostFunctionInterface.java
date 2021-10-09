package Other.FunctionClasses.Cost;

public interface CostFunctionInterface {
    /**
     * Calculates the cost of a network based on the output of the network and the
     * desired output. Varies based on which CostOption parameter was used to create
     * the CostFunction class this method is tied to
     * 
     * @param result        the result of the neural network pulse
     * @param desiredValues the desired values of a pulse for this training example
     * @return the average cost of all neurons
     */
    public double calculateOriginal(double[] result, double[] desiredValues);

    /**
     * Calculates the derivative vector cost of a network based on the output of the
     * network and the desired output. Varies based on which CostOption parameter
     * was used to create the CostFunction class this method is tied to
     * 
     * @param result        the result of the neural network pulse
     * @param desiredValues the desired values of a pulse for this training example
     * @return the derivative vector cost of all neurons
     */
    public double[] calculateDerivative(double[] result, double[] desiredValues);
}