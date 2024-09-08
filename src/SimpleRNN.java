/**
 * Minimal character-level Vanilla RNN model. Written by Mark Wong (@yefangwong) BSD License
 *
 * Copyright (c) 2024, Mark Wong
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted provided
 * that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
 * following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and
 * the following disclaimer in the documentation and/or other materials provided with the distribution.
 * 3. Neither the name of Mark Wong nor the names of its contributors may be used to endorse or promote products
 * derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * ---
 *
 * This software was developed based on code referenced from karpathy/min-char-rnn.py, available at
 * https://gist.github.com/karpathy/d4dee566867f8291f086. The original code is licensed under the
 * BSD license.
 */

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class SimpleRNN {
    private static final int HIDDEN_SIZE = 100; // 隱藏層大小
    private static final int SEQ_LENGTH = 3; // 序列長度
    private static final double LEARNING_RATE = 0.1; // 學習率

    private double[][] wxh; // 輸入層到隱藏層的權重矩陣
    private double[][] whh; // 隱藏層到隱藏層的權重矩陣
    private double[][] why; // 隱藏層到輸出層的權重矩陣
    private double[] bh;    // 隱藏層的 bias
    private double[] by;    // 輸出層的 bias

    private int vocabSize;
    private Map<Character, Integer> charToIdx;
    private Map<Integer, Character> idxToChar;

    public SimpleRNN(String data) {
        char[]  chars = data.toCharArray();

        int dataSize = chars.length;

        // 建立 char 到 index 的映射
        charToIdx = new HashMap<Character, Integer>();
        idxToChar = new HashMap<Integer, Character>();
        int index = 0;
        for (char c : chars) {
            if (charToIdx.containsKey(c) == false) {
                charToIdx.put(c, index);
                idxToChar.put(index, c);
                index++;
            }
        }

        vocabSize = charToIdx.size();
        System.out.println("Data has " + dataSize + " character, " + vocabSize + " unique.");
        System.out.println(charToIdx.toString());
        System.out.println(idxToChar.toString());

        // 初始化權重
        wxh = randomMatrix(HIDDEN_SIZE, vocabSize);   // 輸入層到隱藏層權重
        whh = randomMatrix(HIDDEN_SIZE, HIDDEN_SIZE); // 隱藏層到隱藏層權重
        why = randomMatrix(vocabSize, HIDDEN_SIZE);   // 隱藏層到輸出層權重
        bh = new double[HIDDEN_SIZE];                 // 隱藏層 bias
        by = new double[vocabSize];                   // 輸出層 bias
    }

    private double[][] randomMatrix(int rows, int cols) {
        Random rand = new Random();
        double[][] matrix = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i][j] = rand.nextGaussian() * 0.01;
            }
        }
        return matrix;
    }

    private void train(String data, int iterations) {
        int n = 0;
        int p = 0;
        double smoothLoss = -Math.log(1.0 / vocabSize) * SEQ_LENGTH;
        double[] hPrev = new double[HIDDEN_SIZE];

        while(n <= iterations) {
            hPrev = new double[HIDDEN_SIZE]; // 重置 RNN 記憶體
            p = 0;

            int[] inputs = new int[SEQ_LENGTH];
            int[] targets = new int[SEQ_LENGTH];
            for (int i = 0; i < SEQ_LENGTH - 1; i++) {
                inputs[i] = charToIdx.get(data.charAt(p + i));
                System.out.println("input char:" + data.charAt(p + i + 1));
                targets[i] = charToIdx.get(data.charAt(p + i + 1));
                System.out.println("output char:" + data.charAt(p + i + 1));
                System.out.println(String.format("inputs[%d]:%s", i,  inputs[i]));
                System.out.println(String.format("targets[%d]:%s", i,  targets[i]));
            }

            double loss = 0;
            // 前向傳播
            for (;;) {
                break;
            }

            n++;
        }

        System.out.println("initial smoothLoss:" + smoothLoss);
    }

    public static void main(String[] args) {
        String data = "牛肉麵";
        SimpleRNN rnn = new SimpleRNN(data);
        rnn.train(data, 2);
    }
}