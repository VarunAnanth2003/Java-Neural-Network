package Other;

import java.util.Random;

/**
 * Generates 9x9 arrays for training data
 */
public class DataGenerator {
    private static Random r = new Random();

    /**
     * Generates a square in a 9x9 matrix
     * 
     * @return the matrix for training
     */
    public static double[][] generateA() {
        double a = (r.nextDouble() * (0.25)) + (0.75);
        double[][] ret_val = { { a, a, a, a, a, a, a, a, a }, { a, 0, 0, 0, 0, 0, 0, 0, a },
                { a, 0, 0, 0, 0, 0, 0, 0, a }, { a, 0, 0, 0, 0, 0, 0, 0, a }, { a, 0, 0, 0, 0, 0, 0, 0, a },
                { a, 0, 0, 0, 0, 0, 0, 0, a }, { a, 0, 0, 0, 0, 0, 0, 0, a }, { a, 0, 0, 0, 0, 0, 0, 0, a },
                { a, a, a, a, a, a, a, a, a } };
        return ret_val;
    }

    /**
     * Generates a diamond in a 9x9 matrix
     * 
     * @return the matrix for training
     */
    public static double[][] generateB() {
        double a = (r.nextDouble() * (0.25)) + (0.75);
        double[][] ret_val = { { 0, 0, 0, 0, a, 0, 0, 0, 0 }, { 0, 0, 0, a, 0, a, 0, 0, 0 },
                { 0, 0, a, 0, 0, 0, a, 0, 0 }, { 0, a, 0, 0, 0, 0, 0, a, 0 }, { a, 0, 0, 0, 0, 0, 0, 0, a },
                { 0, a, 0, 0, 0, 0, 0, a, 0 }, { 0, 0, a, 0, 0, 0, a, 0, 0 }, { 0, 0, 0, a, 0, a, 0, 0, 0 },
                { 0, 0, 0, 0, a, 0, 0, 0, 0 } };
        return ret_val;
    }

    /**
     * Generates a plus in a 9x9 matrix
     * 
     * @return the matrix for training
     */
    public static double[][] generateC() {
        double a = (r.nextDouble() * (0.25)) + (0.75);
        double[][] ret_val = { { 0, 0, 0, 0, a, 0, 0, 0, 0 }, { 0, 0, 0, 0, a, 0, 0, 0, 0 },
                { 0, 0, 0, 0, a, 0, 0, 0, 0 }, { 0, 0, 0, 0, a, 0, 0, 0, 0 }, { a, a, a, a, a, a, a, a, a },
                { 0, 0, 0, 0, a, 0, 0, 0, 0 }, { 0, 0, 0, 0, a, 0, 0, 0, 0 }, { 0, 0, 0, 0, a, 0, 0, 0, 0 },
                { 0, 0, 0, 0, a, 0, 0, 0, 0 } };
        return ret_val;
    }

    /**
     * Generates a in a cross9x9 matrix
     * 
     * @return the matrix for training
     */
    public static double[][] generateD() {
        double a = (r.nextDouble() * (0.25)) + (0.75);
        double[][] ret_val = { { a, 0, 0, 0, 0, 0, 0, 0, a }, { 0, a, 0, 0, 0, 0, 0, a, 0 },
                { 0, 0, a, 0, 0, 0, a, 0, 0 }, { 0, 0, 0, a, 0, a, 0, 0, 0 }, { 0, 0, 0, 0, a, 0, 0, 0, 0 },
                { 0, 0, 0, a, 0, a, 0, 0, 0 }, { 0, 0, a, 0, 0, 0, a, 0, 0 }, { 0, a, 0, 0, 0, 0, 0, a, 0 },
                { a, 0, 0, 0, 0, 0, 0, 0, a }, };
        return ret_val;
    }
}