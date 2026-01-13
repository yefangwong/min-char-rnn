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

import logging.HiddenStateLogger;
import logging.LossLogger;

import java.io.*;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class SimpleRNN {
    private static final int HIDDEN_SIZE = 100; // 隱藏層大小
    private static final boolean DEBUG = false;
    private static final double LEARNING_RATE = 0.00004; // 學習率

    // 實驗設定切換
    private static boolean isSuccessMode = true;
    // 基礎參數會根據模式調整
    private static int SEQ_LENGTH = isSuccessMode ? 5 : 4;
    private static int iterations = isSuccessMode ? 82000 : 64000;
    boolean useIdentityInit = false;

    private double[][] wxh; // 輸入層到隱藏層的權重矩陣
    private double[][] whh; // 隱藏層到隱藏層的權重矩陣
    private double[][] why; // 隱藏層到輸出層的權重矩陣
    private double[] bh;    // 隱藏層的 bias
    private double[] by;    // 輸出層的 bias

    // Adagrad 優化器的記憶變數
    private double[][] mWxh; // wxh 的梯度平方累積
    private double[][] mWhh; // whh 的梯度平方累積
    private double[][] mWhy; // why 的梯度平方累積
    private double[] mBh;    // bh 的梯度平方累積
    private double[] mBy;    // by 的梯度平方累積
    private static final double EPSILON = 1e-8; // 避免除零的小常數

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
        if (useIdentityInit) {
            whh = identityMatrix(HIDDEN_SIZE, HIDDEN_SIZE); // 隱藏層到隱藏層權重
        } else {
            whh = randomMatrix(HIDDEN_SIZE, HIDDEN_SIZE); // 隱藏層到隱藏層權重
        }
        why = randomMatrix(vocabSize, HIDDEN_SIZE);   // 隱藏層到輸出層權重
        bh = new double[HIDDEN_SIZE];                 // 隱藏層 bias
        by = new double[vocabSize];                   // 輸出層 bias

        // 初始化 Adagrad 記憶變數
        mWxh = new double[HIDDEN_SIZE][vocabSize];
        mWhh = new double[HIDDEN_SIZE][HIDDEN_SIZE];
        mWhy = new double[vocabSize][HIDDEN_SIZE];
        mBh = new double[HIDDEN_SIZE];
        mBy = new double[vocabSize];
    }

    private double[][] identityMatrix(int rows, int cols) {
        double[][] identityMatrix = new double[rows][cols];
        double scale = 0.9; // <--- 加入這個縮放因子，幫助抑制梯度爆炸

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                // Main diagonal elements
                if (i == j) {
                    identityMatrix[i][j] = scale;
                } else { // Off-diagonal elements
                    identityMatrix[i][j] = 0;
                }
            }
        }
        return identityMatrix;
    }

    private double[][] randomMatrix(int rows, int cols) {
        Random rand = new Random();
        /*double[][] matrix = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i][j] = rand.nextGaussian() * 0.01;
            }
        }
        return matrix;*/
        // 改用 Xavier 初始化 (適用於 tanh)
        double scale = Math.sqrt(6.0 / (rows + cols));
        return Arrays.stream(new double[rows][cols])
                .map(row -> Arrays.stream(row)
                        .map(v -> rand.nextGaussian() * scale)
                        .toArray())
                .toArray(double[][]::new);
    }

    private void train(String data, int iterations) {
        long startTime = System.currentTimeMillis(); // 紀錄開始
        int n = 0;
        int p = 0;
        double smoothLoss = -Math.log(1.0 / vocabSize) * SEQ_LENGTH;
        double[] hPrev = new double[HIDDEN_SIZE]; // 重置 RNN 記憶體

        // 梯度監控相關變數
        double maxGradientNorm = 0.0;
        double avgGradientNorm = 0.0;
        int gradientExplodeCount = 0;
        double clipNorm = 5.0; // 梯度爆炸閾值

        LossLogger logger = new LossLogger(
                "loss.csv",
                LossLogger.LogLevel.AUDIT,
                100
        );
        // 初始化 HiddenStateLogger
        HiddenStateLogger hsLogger = new HiddenStateLogger(
                "hidden_states.csv",
                HIDDEN_SIZE);

        int earlyStopCounter = 0;
        final double earlyStopThreshold = 0.01;
        final int earlyStopPatience = 500;

        System.out.println("initial smoothLoss:" + smoothLoss);
        while(n <= iterations) {
            //System.out.println("iter:" + n + " starts --------------------------------");
            // 檢查指針是否已滑到數據末尾，無法再取出一個完整序列。
            if ((p + SEQ_LENGTH > data.length() || n == 0)) {
                hPrev = new double[HIDDEN_SIZE]; // reset RNN memory
                p = 0; // go from start of data
            }

            int[] inputs = new int[SEQ_LENGTH];
            int[] targets = new int[SEQ_LENGTH];
            for (int i = 0; i < SEQ_LENGTH; i++) {
                // 檢查目標字符的索引 (p + i + 1) 是否超出總資料長度
                // 如果下一個字符 (p + i + 1) 不存在，則代表資料切片結束，必須跳出
                if (p + i + 1 > data.length()) {
                    break;
                }

                // 讀取當前輸入字符（p + i）
                inputs[i] = charToIdx.get(data.charAt(p + i));
                // 讀取下一個目標字符（p + i + 1）
                // 當 input 字符不是結尾字元
                if (p + i + 1 < data.length())
                    targets[i] = charToIdx.get(data.charAt(p + i + 1));
                else // 當 input 字符是結尾字元
                    targets[i] = charToIdx.get(data.charAt(p + i));// 令其為結尾字元，避免 java.lang.StringIndexOutOfBoundsException
            }
            double loss = 0;

            // for debug
            if (DEBUG /*&& (n % 100 == 0)*/) {
                System.out.println("inputs:" + Arrays.toString(inputs));
                System.out.println("targets:" + Arrays.toString(targets));
            }

            // 前向傳播 (得到預測機率)
            ForwardResult result = forward(inputs, hPrev);

            // 記錄隱藏狀態 (只在特定迭代記錄，避免檔案過大)
            if (n % 1000 == 0) { // 例如每 1000 次迭代記錄一次
                for (int t = 0; t < SEQ_LENGTH; t++) {
                    if (inputs[t] == 1) { // 確保 token 只輸出「魚」
                        char token = idxToChar.get(inputs[t]);
                        hsLogger.log(n, t, token, result.h[t]);
                    }
                }
            }

            // 取這一輪的最後一個時間步做為下一輪的初始 hPrev
            hPrev = result.h[result.h.length - 1];

            // 計算 loss (Cross Entropy)
            for (int t = 0; t < SEQ_LENGTH; t++) {
                loss += computeLoss(result.y[t], targets[t]);
            }

            // 平滑系數調大，曲線更穩
            final double smoothFactor = 0.999;
            final double lossFactor   = 0.001;

            // 更新 smoothLoss
            smoothLoss = smoothLoss * smoothFactor + loss * lossFactor;

            // Early stopping check
            if (smoothLoss < earlyStopThreshold) {
                earlyStopCounter++;
            } else {
                earlyStopCounter = 0;
            }

            if (earlyStopCounter >= earlyStopPatience) {
                System.out.println("Early stopping at iteration " + n + " because smooth_loss < " + earlyStopThreshold + " for " + earlyStopPatience + " steps.");
                break;
            }

            // 反向傳播
            BackwardResult grad = backward(inputs, targets, result);

            // 計算梯度範數並監控梯度爆炸
            double gradientNorm = grad.calculateGradientNorm();
            avgGradientNorm = avgGradientNorm * smoothFactor + gradientNorm * (1 - smoothFactor);

            // 全局梯度缩放 & 檢測梯度爆炸
            if (gradientNorm > clipNorm) {
                double scale = clipNorm / gradientNorm;
                grad.scaleGradients(scale);
                //System.out.println("Global gradient scaled at iteration " + n +
                //        ", original norm: " + gradientNorm +
                //        ", scaled factor: " + scale);
                gradientExplodeCount++;
                //System.out.println("Warning: Gradient explosion detected at iteration " + n +
                //        ", gradient norm: " + gradientNorm);
            }

            // 在更新參數前進行梯度裁剪
            double afterClipNorm = grad.calculateGradientNorm();

            // 如果梯度被裁剪，輸出裁剪前後的梯度範數
            if (Math.abs(gradientNorm  - afterClipNorm) > 1e-6) {
                //System.out.println("Gradient clipped at iteration " + n +
                //                  ", before: " + gradientNorm  +
                //                  ", after: " + afterClipNorm);
            }

            if (n % 100 == 0) {
                System.out.println("Iteration: " + n +
                                  ", Loss: " + loss +
                                  ", Smooth Loss: " + smoothLoss +
                                  ", Gradient Norm: " + afterClipNorm +
                                  ", Avg Gradient Norm: " + avgGradientNorm);
            }

            // 更新參數
            updateParameters(grad);

            p += 1; // move data pointer
            n++; // iteration counter
            logger.log(n, loss, smoothLoss);
        }

        // 關閉兩個 logger
        logger.close();
        hsLogger.close();

        long endTime = System.currentTimeMillis(); // 紀錄結束時間 (毫秒)
        double elapsedTime = (endTime - startTime) / 1000.0; // 轉換為秒
        System.out.println("Training time: " + elapsedTime + " seconds");
        System.out.println("Gradient statistics - Max Norm: " + maxGradientNorm +
                          ", Avg Norm: " + avgGradientNorm +
                          ", Explosion Count: " + gradientExplodeCount);
    }

    // 新增梯度裁剪方法
    private double[] clipGradient(double[] grad, double threshold) {
        double norm = Math.sqrt(Arrays.stream(grad).map(x->x*x).sum());
        return norm > threshold ?
                Arrays.stream(grad).map(x -> x*threshold/norm).toArray() :
                grad;
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
             * 這個 dhraw 用於計算輸入層到隱藏層權重(wxh)、隱藏層到隱藏層權重(whh)和隱藏層偏置(bh)的梯度。
             */
            double[] dhraw = multiply(dh, dtanh(forwardResult.h[t]));

            // 計算輸入層和隱藏層梯度
            grad.dwxh = add(grad.dwxh, outer(dhraw, idxToOneHot(inputs[t])));
            // 使用前一時間步的隱藏態 h[t-1]
            grad.dwhh = add(grad.dwhh, outer(dhraw, forwardResult.hPrev[t]));
            grad.dbh = add(grad.dbh, dhraw);

            dhnext = matrixVectorMultiply(transpose(whh), dhraw);

            // 增加局部梯度裁剪 (新增以下方法調用)
            dhraw = clipGradient(dhraw, 1.0);  // 新增局部裁剪
            dhnext = clipGradient(dhnext, 1.0); // 限制單個神經元梯度

            // 添加梯度監控 (新增以下三行)
            double gradNorm = Math.sqrt(Arrays.stream(dhnext).map(x -> x * x).sum());
            double maxGrad = Arrays.stream(dhnext).max().orElse(0);
            double minGrad = Arrays.stream(dhnext).min().orElse(0);

            if (gradNorm < 1e-6) {
                System.out.printf("t=%d  dhnext norm=%.6f  max=%.6f  min=%.6f%n", t, gradNorm, maxGrad, minGrad);
            }
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

    private void updateParameters(BackwardResult grad, int n) {
        double process = n / (double) iterations;
        double dynamicLR = LEARNING_RATE * (1.0 - process); // 線性衰減

        // 使用 Adagrad 優化器更新權重和偏差

        // 更新 wxh 及其記憶變數
        for (int i = 0; i < wxh.length; i++) {
            for (int j = 0; j < wxh[0].length; j++) {
                mWxh[i][j] += grad.dwxh[i][j] * grad.dwxh[i][j];
                wxh[i][j] -= dynamicLR * grad.dwxh[i][j] / Math.sqrt(mWxh[i][j] + EPSILON);
            }
        }

        // 更新 whh 及其記憶變數
        for (int i = 0; i < whh.length; i++) {
            for (int j = 0; j < whh[0].length; j++) {
                mWhh[i][j] += grad.dwhh[i][j] * grad.dwhh[i][j];
                whh[i][j] -= dynamicLR * grad.dwhh[i][j] / Math.sqrt(mWhh[i][j] + EPSILON);
            }
        }

        // 更新 why 及其記憶變數
        for (int i = 0; i < why.length; i++) {
            for (int j = 0; j < why[0].length; j++) {
                mWhy[i][j] += grad.dwhy[i][j] * grad.dwhy[i][j];
                why[i][j] -= dynamicLR * grad.dwhy[i][j] / Math.sqrt(mWhy[i][j] + EPSILON);
            }
        }

        // 更新 bh 及其記憶變數
        for (int i = 0; i < bh.length; i++) {
            mBh[i] += grad.dbh[i] * grad.dbh[i];
            bh[i] -= dynamicLR * grad.dbh[i] / Math.sqrt(mBh[i] + EPSILON);
        }

        // 更新 by 及其記憶變數
        for (int i = 0; i < by.length; i++) {
            mBy[i] += grad.dby[i] * grad.dby[i];
            by[i] -= dynamicLR * grad.dby[i] / Math.sqrt(mBy[i] + EPSILON);
        }
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
        int V = vocabSize;

        ForwardResult result = new ForwardResult();
        result.h = new double[T][H];
        result.hPrev = new double[T][H];
        result.y = new double[T][V];
        result.z = new double[T][V];

        double[] h_t_minus_1 = hPrev;

        for (int t = 0; t < inputs.length; t++) {
            // 儲存本步驟的前一隱藏態 h[t-1]
            result.hPrev[t] = Arrays.copyOf(h_t_minus_1, h_t_minus_1.length);

            // 計算隱藏層狀態 ht = tanh(xt * Wxh + ht-1 * Whh + hb)
            result.h[t] = tanh(add(matrixVectorMultiply(this.wxh, idxToOneHot(inputs[t])),
                        add(matrixVectorMultiply(this.whh, h_t_minus_1), this.bh)));

            // 更新前一狀態
            h_t_minus_1 = result.h[t];

            // 計算輸出層與 softmax
            result.z[t] = add(matrixVectorMultiply(this.why, result.h[t]), this.by);
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

    // 從檔案讀取訓練資料 (支援 Big5 編碼)
    private static String readDataFromFile(String filePath) throws IOException {
        return DataCleaner.cleanBibleText("resources" + File.separator + "hb5_utf8.txt");
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

        System.out.print(String.format("seedChar: %s", seedChar));

        x[charToIdx.get(seedChar)] = 1.0;

        int currentCharIdx = charToIdx.get(seedChar);
        StringBuffer sb = new StringBuffer();
        sb.append(seedChar);
        for (int i = 0; i < length; i++) {
            ForwardResult result = forward(new int[]{currentCharIdx}, h);

            if (result == null) {
                System.out.println("Error: Forward propagation failed");
                return;
            }

            double[] probs = result.y[0];
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
            sb.append(nextChar);
            System.out.print(nextChar);

            // 更新輸入和隱藏狀態
            x = new double[vocabSize];
            x[nextCharIdx] = 1.0;
            h = result.h[result.h.length - 1];
            currentCharIdx = nextCharIdx; // 使用當前字符的索引作為下一個輸入
        }
        System.out.println("\n");
        System.out.println(sb.toString());
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

    public static void main(String[] args) throws IOException, ClassNotFoundException {
        SimpleRNN rnn = null;

        if (args.length == 0 || (args[0].isEmpty() || args[0].contains("--train"))) {
            String data;
            if (args.length >= 2 && args[1].contains("--file")) {
                // 從檔案讀取訓練資料
                String filePath = "resources/hb5_utf8.txt";
                if (args.length >= 3) {
                    filePath = args[2];
                }
                System.out.println("Reading training data from file: " + filePath);
                data = readDataFromFile(filePath);
            } else {
                // 使用預設的訓練資料
                //data = "我只有一件事，就是忘記背後努力面前的，向著標竿直跑，要得 神在基督耶穌裏從上面召我來得的獎賞。#";
                data = "鮭魚生魚片#";
                //data = "AUTOPUBACCFLAG#AUTO#PUB#ACC#FLAG#";
            }

            rnn = new SimpleRNN(data);
            if (args.length >= 3 && args[3].contains("--iter")) {
                try {
                    iterations = Integer.parseInt(args[4]);
                } catch (NumberFormatException e) {
                    System.out.println("Invalid iteration number, using default: " + iterations);
                }
            }

            System.out.println("Training with " + data.length() + " characters, " + iterations + " iterations");
            // 使用訓練資料的第一個字符作為生成的種子
            char seedChar = data.charAt(0);
            rnn.train(data, iterations);
            rnn.saveModel(String.format("rnn_model_%d.dat", iterations));
        } else if (args[0].contains("--inference")) {
            String modelPath = "rnn_model_1200.dat";
            if (args.length >= 2) {
                modelPath = args[1];
            }

            if (args.length < 3) { // Expect at least --inference <model_path> <seed_char>
                System.out.println("Usage for inference: java SimpleRNN --inference <model_path> <seed_char> [generate_length]");
                return;
            }

            char seedChar = args[2].charAt(0);

            int genLength = 48; // Default generation length
            if (args.length >= 4) {
                try {
                    genLength = Integer.parseInt(args[3]);
                } catch (NumberFormatException e) {
                    System.out.println("Invalid generation length, using default: " + genLength);
                }
            }

            rnn = new SimpleRNN("");
            System.out.println("Loading model from: " + modelPath);
            rnn.loadModel(modelPath);
            System.out.println("Model loaded. Generating sequence from '" + seedChar + "' (length: " + genLength + ")");
            rnn.generate(genLength, seedChar);
        }
    }
}
