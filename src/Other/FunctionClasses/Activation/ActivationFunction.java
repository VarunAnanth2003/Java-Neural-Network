package Other.FunctionClasses.Activation;

public class ActivationFunction {
    private ActivationFunctionInterface selectedFunction;
    private ActivationOptions ao;

    /**
     * When an enum value from ActivationOptions is passed in, the selectedFunction
     * parameter will take on the characteristics of the equation that the enum
     * value is tied to. See the code for a breakdown of which enum value relates to
     * which equation (although it should be apparent)
     * 
     * @param f the enum value corresponding to a specific function
     * @see ActivationFunctionInterface
     * @see ActivationOptions
     */
    public ActivationFunction(ActivationOptions f) {
        ao = f;
        switch (f) {
            case SIGMOID:
                selectedFunction = new ActivationFunctionInterface() {

                    @Override
                    public double calculateOriginal(double input) {
                        return 1 / (1 + Math.exp(-input));
                    }

                    @Override
                    public double calculateDerivative(double input) {
                        return (calculateOriginal(input) * (1 - calculateOriginal(input)));
                    }

                };
                break;
            case RE_LU:
                selectedFunction = new ActivationFunctionInterface() {

                    @Override
                    public double calculateOriginal(double input) {
                        return input > 0 ? input : 0;
                    }

                    @Override
                    public double calculateDerivative(double input) {
                        return input > 0 ? 1 : 0;
                    }

                };
                break;
            case LEAKY_RE_LU:
                selectedFunction = new ActivationFunctionInterface() {

                    @Override
                    public double calculateOriginal(double input) {
                        return input > 0 ? input : 0.1 * input;
                    }

                    @Override
                    public double calculateDerivative(double input) {
                        return input > 0 ? 1 : 0.1;
                    }

                };
                break;
            default:
        }
    }

    public ActivationFunctionInterface getFunction() {
        return selectedFunction;
    }

    public String getAo() {
        return ao.toString();
    }

    /**
     * Converts a passed in string to a valid activation function
     * 
     * @param s String that relates to enum value to pass in
     * @return a valid activation function with an included equation in relation to
     *         the passed in String if the string is a valid enum member)
     */
    public static ActivationFunction convertStringToObject(String s) {
        return new ActivationFunction(ActivationOptions.valueOf(s));
    }
}