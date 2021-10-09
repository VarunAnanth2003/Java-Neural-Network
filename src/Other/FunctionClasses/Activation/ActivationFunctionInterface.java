package Other.FunctionClasses.Activation;

public interface ActivationFunctionInterface {
    /**
     * Calculates the final activation of a neuron based off of the pre-activation
     * value. The exact equation used to determine this final activation value
     * varies depending on which ActivationOption parameter was used to create the
     * ActivationFunction class this method is tied to
     * 
     * @param input the pre-activation value of the neuron
     * @return the final activation value of the neuron
     */
    public double calculateOriginal(double input);

    /**
     * Calculates the derivative of the final activation of a neuron based off of
     * the pre-activation value. The exact equation used to determine this final
     * activation value varies depending on which ActivationOption parameter was
     * used to create the ActivationFunction class this method is tied to
     * 
     * @param input the pre-activation value of the neuron
     * @return the derivative of the final activation value of the neuron
     */
    public double calculateDerivative(double input);
}