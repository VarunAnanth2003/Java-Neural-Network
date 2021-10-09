package Exceptions;

/**
 * Simple custom exception for when there are too few (< 2) layers in a network
 */
public class TooFewLayersException extends Exception {
    public TooFewLayersException() {
        System.err.println("Too few layers in this network");
    }
}