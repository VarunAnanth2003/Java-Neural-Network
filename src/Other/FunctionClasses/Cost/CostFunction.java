package Other.FunctionClasses.Cost;

public class CostFunction {
    private CostFunctionInterface selectedFunction;
    private CostOptions co;

    /**
     * When an enum value from CostOptions is passed in, the selectedFunction
     * parameter will take on the characteristics of the equation that the enum
     * value is tied to. See the code for a breakdown of which enum value relates to
     * which equation (although it should be apparent)
     * 
     * @param f the enum value corresponding to a specific function
     * @see CostFunctionInterface
     * @see CostOptions
     */
    public CostFunction(CostOptions f) {
        co = f;
        switch (f) {
            case QUADRATIC:
                selectedFunction = new CostFunctionInterface() {

                    @Override
                    public double calculateOriginal(double[] result, double[] desiredValues) {
                        double ret_val = 0;
                        for (int i = 0; i < result.length; i++) {
                            try {
                                ret_val += Math.pow(result[i] - desiredValues[i], 2);
                            } catch (IndexOutOfBoundsException e) {
                                System.err.println("Desired values array does not match the network result array");
                                e.printStackTrace();
                            }
                        }
                        return ret_val;
                    }

                    @Override
                    public double[] calculateDerivative(double[] result, double[] desiredValues) {
                        double[] ret_val = new double[result.length];
                        for (int i = 0; i < result.length; i++) {
                            try {
                                ret_val[i] = 2 * (result[i] - desiredValues[i]);
                            } catch (IndexOutOfBoundsException e) {
                                System.err.println("Desired values array does not match the network result array");
                                e.printStackTrace();
                            }
                        }
                        return ret_val;
                    }

                };
                break;
            default:
        }
    }

    public CostFunctionInterface getFunction() {
        return selectedFunction;
    }

    public String getCo() {
        return co.toString();
    }

    /**
     * Converts a passed in string to a valid cost function
     * 
     * @param s String that relates to enum value to pass in
     * @return a valid cost function with an included equation in relation to the
     *         passed in String if the string is a valid enum member)
     */
    public static CostFunction convertStringToObject(String s) {
        return new CostFunction(CostOptions.valueOf(s));
    }
}