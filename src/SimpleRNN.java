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

import java.io.*;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class SimpleRNN {
    private static final int HIDDEN_SIZE = 100; // 隱藏層大小
    private static final int SEQ_LENGTH = 1; // 序列長度
    private static final double LEARNING_RATE = 0.01; // 學習率

    private double[][] wxh; // 輸入層到隱藏層的權重矩陣
    private double[][] whh; // 隱藏層到隱藏層的權重矩陣
    private double[][] why; // 隱藏層到輸出層的權重矩陣
    private double[] bh;    // 隱藏層的 bias
    private double[] by;    // 輸出層的 bias

    private int vocabSize;
    private Map<Character, Integer> charToIdx;
    private Map<Integer, Character> idxToChar;

    public SimpleRNN(String data) {
        if (data.isEmpty()) return; // 表示是推理模式

        char[] chars = data.toCharArray();

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
        long startTime = System.currentTimeMillis(); // 紀錄開始
        int n = 0;
        int p = 0;
        double smoothLoss = -Math.log(1.0 / vocabSize) * SEQ_LENGTH;
        double[] hPrev = new double[HIDDEN_SIZE]; // 重置 RNN 記憶體

        System.out.println("initial smoothLoss:" + smoothLoss);
        while(n <= iterations) {
            //System.out.println("iter:" + n + " starts --------------------------------");
            if ((p + SEQ_LENGTH >= data.length() || n == 0)) {
                hPrev = new double[HIDDEN_SIZE]; // reset RNN memory
                p = 0; // go from start of data
            }

            int[] inputs = new int[SEQ_LENGTH];
            int[] targets = new int[SEQ_LENGTH];
            for (int i = 0; i < SEQ_LENGTH; i++) {
                inputs[i] = charToIdx.get(data.charAt(p + i));
                targets[i] = charToIdx.get(data.charAt(p + i + 1));
            }
            double loss = 0;

            // for debug
            //System.out.println("inputs:" + Arrays.toString(inputs));
            //System.out.println("targets:" + Arrays.toString(targets));

            // 前向傳播 (得到預測機率)
            ForwardResult result = forward(inputs, hPrev);
            // 取這一輪的最後一個時間步做為下一輪的初始 hPrev
            hPrev = result.h[result.h.length - 1];

            // 計算 loss (Cross Entropy)
            for (int t = 0; t < SEQ_LENGTH; t++) {
                loss += computeLoss(result.y[t], targets[t]);
            }

            // 更新 smoothLoss
            smoothLoss = smoothLoss * 0.99 + loss * 0.001;

            if (n % 100 == 0)
                System.out.println("Iteration: " + n + ", Loss: " + loss + ", Smooth Loss: " + smoothLoss);

            // 反向傳播
            BackwardResult grad = backward(inputs, targets, result);

            // 更新參數
            updateParameters(grad);

            p += 1; // move data pointer
            n++; // iteration counter
        }

        long endTime = System.currentTimeMillis(); // 紀錄結束時間 (毫秒)
        double elapsedTime = (endTime - startTime) / 1000.0; // 轉換為秒
        System.out.println("Training time: " + elapsedTime + " seconds");
    }

    private BackwardResult backward(int[] inputs, int[] targets, ForwardResult forwardResult) {
        int T = inputs.length;
        int H = HIDDEN_SIZE;
        int V = vocabSize;

        BackwardResult grad = new BackwardResult();
        grad.dwxh = new double[H][V];
        grad.dwhh = new double[H][H];
        grad.dwhy = new double[V][H];
        grad.dbh = new double[H];
        grad.dby = new double[V];

        double[] dhnext = new double[H];

        for (int t = T - 1; t >= 0; t--) {
            double[] dy = Arrays.copyOf(forwardResult.y[t], V);
            dy[targets[t]] -= 1.0;

            // 計算輸出層梯度
            grad.dwhy = add(grad.dwhy, outer(dy, forwardResult.h[t]));
            grad.dby = add(grad.dby, dy);

            // 計算隱藏層梯度
            double[] dh = matrixVectorMultiply(transpose(why), dy);
            dh = add(dh, dhnext);

            /**
             * dhraw 是經過 tanh 激活函数的導數修正後的誤差訊號。
             * 在反向傳播中，隱藏層的誤差 dh 需要乘以 tanh 函数的導數tanh(h)，
             * 以反映激活函数對誤差的影響，從而得到對隱藏層输入的真實梯度 dhraw。
             * 这个 dhraw 用於計算输入層到隱藏層權重(wxh)、隱藏層到隱藏層權重(whh)和隱藏層偏置(bh)的梯度。
             */
            double[] dhraw = multiply(dh, dtanh(forwardResult.h[t]));

            // 計算輸入層和隱藏層梯度
            grad.dwxh = add(grad.dwxh, outer(dhraw, idxToOneHot(inputs[t])));
            grad.dwhh = add(grad.dwhh, outer(dhraw, forwardResult.h[t]));
            grad.dbh = add(grad.dbh, dhraw);

            dhnext = matrixVectorMultiply(transpose(whh), dhraw);
        }
        return grad;
    }

    private void updateParameters(BackwardResult grad) {
        // 更新權重和偏差
        wxh = subtract(wxh, scale(grad.dwxh, LEARNING_RATE));
        whh = subtract(whh, scale(grad.dwhh, LEARNING_RATE));
        why = subtract(why, scale(grad.dwhy, LEARNING_RATE));
        bh = subtract(bh, scale(grad.dbh, LEARNING_RATE));
        by = subtract(by, scale(grad.dby, LEARNING_RATE));
    }

    private double[][] outer(double[] a, double[] b) {
        double[][] result = new double[a.length][b.length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < b.length; j++) {
                result[i][j] = a[i] * b[j];
            }
        }
        return result;
    }

    private double[] dtanh(double[] x) {
        double[] result = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            result[i] = 1 - x[i] * x[i];
        }
        return result;
    }

    private double[][] scale(double[][] matrix, double factor) {
        double[][] result = new double[matrix.length][matrix[0].length];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                result[i][j] = matrix[i][j] * factor;
            }
        }
        return result;
    }

    private double[] scale(double[] vector, double factor) {
        double[] result = new double[vector.length];
        for (int i = 0; i < vector.length; i++) {
            result[i] = vector[i] * factor;
        }
        return result;
    }

    private double[][] subtract(double[][] a, double[][] b) {
        double[][] result = new double[a.length][a[0].length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                result[i][j] = a[i][j] - b[i][j];
            }
        }
        return result;
    }

    private double[] subtract(double[] a, double[] b) {
        double[] result = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] - b[i];
        }
        return result;
    }

    private double[][] transpose(double[][] matrix) {
        double[][] result = new double[matrix[0].length][matrix.length];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                result[j][i] = matrix[i][j];
            }
        }
        return result;
    }

    private double[] multiply(double[] a, double[] b) {
        double[] result = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] * b[i];
        }
        return result;
    }

    private double computeLoss(double[] probabilities, int targetIndex) {
        return -Math.log(probabilities[targetIndex] + 1e-9);// 1e-9 是防止 log(0) 出錯
    }

    // 前項傳播方法
    private ForwardResult forward(int[] inputs, double[] hPrev) {
        int T = inputs.length;
        int H = whh.length;
        int V = whh[0].length;

        ForwardResult result = new ForwardResult();
        result.h = new double[T][H];
        result.y = new double[T][V];
        result.z = new double[T][V];

        for (int t = 0; t < inputs.length; t++) {
            // 計算隱藏層狀態 ht = tanh(xt * Wxh + ht-1 * Whh + hb)
            result.h[t] = tanh(add(matrixVectorMultiply(this.wxh, idxToOneHot(inputs[t])),
                        add(matrixVectorMultiply(this.whh, hPrev), this.bh)));

            // 計算輸出層的 yt = Why * ht + by
            result.z[t] = add(matrixVectorMultiply(this.why, result.h[t]), this.by);

            // 計算 softmax 輸出概率 pt
            result.y[t] = softmax(result.z[t]);
        }
        return result;
    }

    // 矩陣和向量相乘的輔助函數
    private double[] matrixVectorMultiply(double[][] matrix, double[] vector) {
        // A:[100][3] x:[1,0,0] -> A:100 X 3 , x:3 X 1, 100 X 1
        double[] result = new double[matrix.length];
        for (int i = 0; i < matrix.length; i++) {
            result[i] = 0;
            for (int j = 0; j < vector.length; j++) {
                result[i] += matrix[i][j] * vector[j];
            }
        }
        return result;
    }

    // 向量相加的輔助函數
    private double[] add(double[] a, double[] b) {
        double[] result = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] + b[i];
        }
        return result;
    }

    private double[][] add(double[][] a, double[][] b) {
        double[][] result = new double[a.length][a[0].length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                result[i][j] = a[i][j] + b[i][j];
            }
        }
        return result;
    }

    // 簡化版的 tanh 函數
    private double[] tanh(double[] x) {
        double[] result = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            result[i] = Math.tanh(x[i]);
        }
        return result;
    }

    private double[] idxToOneHot(int idxes) {
        double[] oneHot = new double[vocabSize];
        oneHot[idxes] = 1.0;
        return oneHot;
    }

    private int sampleFromProbabilities(double[] probabilities) {
        double randomValue = Math.random();
        double cumulativeProbability = 0.0;
        for (int i = 0; i < probabilities.length; i++) {
          cumulativeProbability += probabilities[i];
          if (randomValue <= cumulativeProbability) {
              return i;
          }
        }
        return probabilities.length - 1;
    }

    private int argmax(double[] array) {
        int maxIndex = 0;
        double max = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > max) {
                max = array[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    public static void main(String[] args) throws IOException, ClassNotFoundException {
        SimpleRNN rnn = null;

        if (args.length == 0 || (args[0].isEmpty() || args[0].contains("--train"))) {
            //String data = "鮭魚生魚片#";
            String data = "查詢所有保單數量->select * from table;#";
            rnn = new SimpleRNN(data);
            int iter = 11000;
            rnn.train(data, iter);
            rnn.generate(29, '查');
            //rnn.generate(4, '鮭');
            rnn.saveModel(String.format("rnn_model_%d.dat", iter));
        } else if (args[0].contains("--inference")) {
            rnn = new SimpleRNN("");
            rnn.loadModel("rnn_model_1200.dat");
            rnn.generate(3, '瑪');
        }
    }

    private void loadModel(String fileName) throws IOException, ClassNotFoundException {
        try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(fileName))) {
            wxh = (double[][]) in.readObject();
            whh = (double[][]) in.readObject();
            why = (double[][]) in.readObject();
            bh = (double[]) in.readObject();
            by = (double[]) in.readObject();
            charToIdx = (Map<Character, Integer>) in.readObject();
            idxToChar = (Map<Integer, Character>) in.readObject();
            vocabSize = charToIdx.size();
        }
    }

    private void saveModel(String fileName) throws IOException {
        try (ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(fileName))) {
            out.writeObject(wxh);
            out.writeObject(whh);
            out.writeObject(why);
            out.writeObject(bh);
            out.writeObject(by);
            out.writeObject(charToIdx);
            out.writeObject(idxToChar);
        }
    }

    // 使用模型生成長序列
    private void generate(int length, char seedChar) {
        double[] h = new double[HIDDEN_SIZE]; // 初始隱藏狀態
        double[] x = new double[vocabSize]; // One-Hot 編碼

        System.out.print(seedChar);

        x[charToIdx.get(seedChar)] = 1.0;

        int currentCharIdx = charToIdx.get(seedChar);
        for (int i = 0; i < length; i++) {
            ForwardResult result = forward(new int[]{currentCharIdx}, h);

            if (result == null) {
                System.out.println("Error: Forward propagation failed");
                return;
            }

            double[] probs = softmax(result.z[0]);
            System.out.println("\nSoftmax 機率分布:");
            for (int j = 0; j < probs.length; j++) {
                System.out.printf("%s : %.4f     ", idxToChar.get(j), probs[j]);
            }
            System.out.println("\n");
            int nextCharIdx = argmax(probs);
            if (nextCharIdx < 0 || nextCharIdx >= vocabSize) {
                System.out.println("Error: Invalid character index generated");
                return;
            }
            char nextChar = idxToChar.get(nextCharIdx);
            if (nextChar == '#') {
                break; // 停止輸出
            }
            System.out.print(nextChar);

            // 更新輸入和隱藏狀態
            x = new double[vocabSize];
            x[nextCharIdx] = 1.0;
            h = result.h[result.h.length - 1];
            currentCharIdx = nextCharIdx; // 使用當前字符的索引作為下一個輸入
        }
        System.out.println();
    }

    private double[] softmax(double[] x) {
        double[] result = new double[x.length];
        double sum = 0.0;
        for (double value : x) {
            sum += Math.exp(value);
        }
        for (int i = 0; i < x.length; i++) {
            result[i] = Math.exp(x[i]) / sum;
        }
        return result;
    }
}