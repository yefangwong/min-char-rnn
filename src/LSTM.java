import java.io.*;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class LSTM {
    private static final int HIDDEN_SIZE = 100;
    private static final int SEQ_LENGTH = 25; // 維持 25 不動
    private static final double LEARNING_RATE = 0.1; 
    private static int iterations = 1200;

    private double[][] wxh; 
    private double[][] whh; 
    private double[] bh;    
    private double[][] why; 
    private double[] by;    

    private double[][] mWxh, mWhh, mWhy;
    private double[] mBh, mBy;
    private static final double EPSILON = 1e-8;

    private int vocabSize;
    private Map<Character, Integer> charToIdx;
    private Map<Integer, Character> idxToChar;

    public LSTM(String data) {
        if (data.isEmpty()) return;
        char[] chars = data.toCharArray();
        charToIdx = new HashMap<>();
        idxToChar = new HashMap<>();
        int index = 0;
        for (char c : chars) {
            if (!charToIdx.containsKey(c)) {
                charToIdx.put(c, index);
                idxToChar.put(index, c);
                index++;
            }
        }
        vocabSize = charToIdx.size();
        wxh = randomMatrix(4 * HIDDEN_SIZE, vocabSize);
        whh = randomMatrix(4 * HIDDEN_SIZE, HIDDEN_SIZE);
        bh = new double[4 * HIDDEN_SIZE];
        why = randomMatrix(vocabSize, HIDDEN_SIZE);
        by = new double[vocabSize];
        mWxh = new double[4 * HIDDEN_SIZE][vocabSize];
        mWhh = new double[4 * HIDDEN_SIZE][HIDDEN_SIZE];
        mWhy = new double[vocabSize][HIDDEN_SIZE];
        mBh = new double[4 * HIDDEN_SIZE];
        mBy = new double[vocabSize];
    }

    private double[][] randomMatrix(int rows, int cols) {
        Random rand = new Random();
        double scale = Math.sqrt(6.0 / (rows + cols));
        double[][] matrix = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) matrix[i][j] = rand.nextGaussian() * scale;
        }
        return matrix;
    }

    public ForwardResult forward(int[] inputs, double[] hPrev, double[] cPrev) {
        int T = inputs.length; // 步長 T 現在是動態的
        int H = HIDDEN_SIZE;
        int V = vocabSize;
        ForwardResult res = new ForwardResult(T, H, V);
        double[] h_t = Arrays.copyOf(hPrev, H);
        double[] c_t = Arrays.copyOf(cPrev, H);
        for (int t = 0; t < T; t++) {
            res.hPrev[t] = Arrays.copyOf(h_t, H);
            res.cPrev[t] = Arrays.copyOf(c_t, H);
            double[] gateRaw = add(matrixVectorMultiply(this.wxh, idxToOneHot(inputs[t])),
                               add(matrixVectorMultiply(this.whh, h_t), this.bh));
            for (int i = 0; i < H; i++) {
                res.is[t][i] = sigmoid(gateRaw[i]);
                res.fs[t][i] = sigmoid(gateRaw[i + H]);
                res.os[t][i] = sigmoid(gateRaw[i + 2 * H]);
                res.gs[t][i] = Math.tanh(gateRaw[i + 3 * H]);
                c_t[i] = res.fs[t][i] * res.cPrev[t][i] + res.is[t][i] * res.gs[t][i];
                h_t[i] = res.os[t][i] * Math.tanh(c_t[i]);
            }
            res.h[t] = Arrays.copyOf(h_t, H);
            res.c[t] = Arrays.copyOf(c_t, H);
            res.z[t] = add(matrixVectorMultiply(this.why, h_t), this.by);
            res.y[t] = softmax(res.z[t]);
        }
        return res;
    }

    public BackwardResult backward(int[] inputs, int[] targets, ForwardResult res) {
        int T = inputs.length;
        int H = HIDDEN_SIZE;
        int V = vocabSize;
        BackwardResult grad = new BackwardResult();
        grad.dwxh = new double[4 * H][V]; grad.dwhh = new double[4 * H][H];
        grad.dwhy = new double[V][H]; grad.dbh = new double[4 * H]; grad.dby = new double[V];
        double[] dhnext = new double[H], dcnext = new double[H];
        for (int t = T - 1; t >= 0; t--) {
            double[] dy = Arrays.copyOf(res.y[t], V); dy[targets[t]] -= 1.0;
            for (int i = 0; i < V; i++) {
                for (int j = 0; j < H; j++) grad.dwhy[i][j] += dy[i] * res.h[t][j];
                grad.dby[i] += dy[i];
            }
            double[] dh = matrixVectorMultiply(transpose(why), dy);
            for (int i = 0; i < H; i++) dh[i] += dhnext[i];
            double[] dgateRaw = new double[4 * H];
            for (int i = 0; i < H; i++) {
                double tanhCt = Math.tanh(res.c[t][i]);
                double dos = dh[i] * tanhCt * res.os[t][i] * (1.0 - res.os[t][i]);
                dgateRaw[i + 2 * H] = dos;
                double dc = dh[i] * res.os[t][i] * (1.0 - tanhCt * tanhCt) + dcnext[i];
                double dfs = dc * res.cPrev[t][i] * res.fs[t][i] * (1.0 - res.fs[t][i]);
                dgateRaw[i + H] = dfs;
                double dis = dc * res.gs[t][i] * res.is[t][i] * (1.0 - res.is[t][i]);
                dgateRaw[i] = dis;
                double dgs = dc * res.is[t][i] * (1.0 - res.gs[t][i] * res.gs[t][i]);
                dgateRaw[i + 3 * H] = dgs;
                dcnext[i] = dc * res.fs[t][i];
            }
            double[] x = idxToOneHot(inputs[t]);
            for (int i = 0; i < 4 * H; i++) {
                for (int j = 0; j < V; j++) grad.dwxh[i][j] += dgateRaw[i] * x[j];
                for (int j = 0; j < H; j++) grad.dwhh[i][j] += dgateRaw[i] * res.hPrev[t][j];
                grad.dbh[i] += dgateRaw[i];
            }
            dhnext = matrixVectorMultiply(transpose(whh), dgateRaw);
        }
        clipGradients(grad);
        return grad;
    }

    private void clipGradients(BackwardResult grad) {
        double threshold = 5.0;
        double norm = grad.calculateGradientNorm();
        if (norm > threshold) grad.scaleGradients(threshold / norm);
    }

    private void train(String data, int iterations) {
        int n = 0, p = 0;
        double smoothLoss = -Math.log(1.0 / vocabSize) * SEQ_LENGTH;
        double[] hPrev = new double[HIDDEN_SIZE], cPrev = new double[HIDDEN_SIZE];

        while (n <= iterations) {
            // 動態計算當前步長，確保不捨棄末尾
            int steps = Math.min(SEQ_LENGTH, data.length() - p - 1);
            
            if (steps <= 0) {
                p = 0;
                hPrev = new double[HIDDEN_SIZE];
                cPrev = new double[HIDDEN_SIZE];
                steps = Math.min(SEQ_LENGTH, data.length() - p - 1);
            }

            int[] inputs = new int[steps], targets = new int[steps];
            for (int t = 0; t < steps; t++) {
                inputs[t] = charToIdx.get(data.charAt(p + t));
                targets[t] = charToIdx.get(data.charAt(p + t + 1));
            }

            ForwardResult res = forward(inputs, hPrev, cPrev);
            hPrev = res.h[steps - 1];
            cPrev = res.c[steps - 1];

            double loss = 0;
            for (int t = 0; t < steps; t++) loss -= Math.log(res.y[t][targets[t]] + 1e-12);
            smoothLoss = smoothLoss * 0.999 + loss * 0.001;

            updateParameters(backward(inputs, targets, res));

            if (n % 100 == 0) System.out.printf("iter %d, loss: %f, smooth_loss: %f\n", n, loss, smoothLoss);
            if (n % 1000 == 0) generate(100, data.charAt(0));
            
            p += steps; 
            n++;
        }
    }

    private void updateParameters(BackwardResult grad) {
        for (int i = 0; i < wxh.length; i++) {
            for (int j = 0; j < wxh[0].length; j++) {
                mWxh[i][j] += grad.dwxh[i][j] * grad.dwxh[i][j];
                wxh[i][j] -= LEARNING_RATE * grad.dwxh[i][j] / Math.sqrt(mWxh[i][j] + EPSILON);
            }
        }
        for (int i = 0; i < whh.length; i++) {
            for (int j = 0; j < whh[0].length; j++) {
                mWhh[i][j] += grad.dwhh[i][j] * grad.dwhh[i][j];
                whh[i][j] -= LEARNING_RATE * grad.dwhh[i][j] / Math.sqrt(mWhh[i][j] + EPSILON);
            }
        }
        for (int i = 0; i < why.length; i++) {
            for (int j = 0; j < why[0].length; j++) {
                mWhy[i][j] += grad.dwhy[i][j] * grad.dwhy[i][j];
                why[i][j] -= LEARNING_RATE * grad.dwhy[i][j] / Math.sqrt(mWhy[i][j] + EPSILON);
            }
        }
        for (int i = 0; i < bh.length; i++) {
            mBh[i] += grad.dbh[i] * grad.dbh[i];
            bh[i] -= LEARNING_RATE * grad.dbh[i] / Math.sqrt(mBh[i] + EPSILON);
        }
        for (int i = 0; i < by.length; i++) {
            mBy[i] += grad.dby[i] * grad.dby[i];
            by[i] -= LEARNING_RATE * grad.dby[i] / Math.sqrt(mBy[i] + EPSILON);
        }
    }

    private void generate(int length, char seed) {
        double[] h = new double[HIDDEN_SIZE], c = new double[HIDDEN_SIZE];
        int idx = charToIdx.get(seed);
        System.out.print("Generated: " + seed);
        for (int i = 0; i < length; i++) {
            ForwardResult res = forward(new int[]{idx}, h, c);
            h = res.h[0]; c = res.c[0];
            double[] probs = res.y[0];
            double r = Math.random(), cum = 0;
            int nextIdx = 0;
            for (int j = 0; j < probs.length; j++) {
                cum += probs[j];
                if (r <= cum) { nextIdx = j; break; }
            }
            char nextChar = idxToChar.get(nextIdx);
            if (nextChar == '#') break; // 修正點：看到結束符號就停止
            System.out.print(nextChar);
            idx = nextIdx;
        }
        System.out.println();
    }

    private double sigmoid(double x) { return 1.0 / (1.0 + Math.exp(-x)); }
    private double[] softmax(double[] x) {
        double[] res = new double[x.length];
        double max = Double.NEGATIVE_INFINITY;
        for (double v : x) if (v > max) max = v;
        double sum = 0;
        for (int i = 0; i < x.length; i++) { res[i] = Math.exp(x[i] - max); sum += res[i]; }
        for (int i = 0; i < x.length; i++) res[i] /= sum;
        return res;
    }
    private double[] idxToOneHot(int idx) {
        double[] oneHot = new double[vocabSize]; oneHot[idx] = 1.0; return oneHot;
    }
    private double[] matrixVectorMultiply(double[][] matrix, double[] vector) {
        double[] res = new double[matrix.length];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < vector.length; j++) res[i] += matrix[i][j] * vector[j];
        }
        return res;
    }
    private double[][] transpose(double[][] matrix) {
        double[][] res = new double[matrix[0].length][matrix.length];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) res[j][i] = matrix[i][j];
        }
        return res;
    }
    private double[] add(double[] a, double[] b) {
        double[] res = new double[a.length];
        for (int i = 0; i < a.length; i++) res[i] = a[i] + b[i];
        return res;
    }

    public static void main(String[] args) {
        String data = "我只有一件事，就是忘記背後努力面前的，向著標竿直跑，要得 神在基督耶穌裏從上面召我來得的獎賞。#";
        LSTM lstm = new LSTM(data);
        lstm.train(data, iterations);
    }
}
