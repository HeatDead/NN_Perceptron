package org.example;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.List;

public class Main {
    public static void main(String[] args) throws Exception {
        double[][] inputs = {
                {1,1,1, //0
                 1,0,1,
                 1,0,1,
                 1,0,1,
                 1,1,1},
                {0,1,0, //1
                 1,1,0,
                 0,1,0,
                 0,1,0,
                 1,1,1},
                {0,1,1, //2
                 1,0,1,
                 0,0,1,
                 0,1,0,
                 1,1,1},
                {1,1,1, //3
                 0,0,1,
                 1,1,1,
                 0,0,1,
                 1,1,1}};

        double[][] targets = {{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}};

        NeuralNetwork nn = new NeuralNetwork(15,2,4);

        nn.l_rate = 1;

        nn.fit(inputs, targets, 3000);

        List<Double> output;
        double [][] input = {
                {1,1,1, //0
                        1,0,1,
                        1,0,1,
                        1,0,1,
                        1,1,1},
                {0,1,0, //1
                1,1,0,
                0,1,0,
                0,1,0,
                1,1,1},{0,1,1, //2
                1,0,1,
                0,0,1,
                0,1,0,
                1,1,1},{
                0.5,0.7,1, //3
                0,0,1,
                0.3,1,1,
                0,0,1,
                1,1,1}
        };
        nn.drawLoss();

        for(double d[]:input)
        {
            output = (List<Double>) nn.predict(d, false);
            for (Double db : output)
                System.out.printf("%.4f  ", db);
            System.out.println("");
        }
    }
}