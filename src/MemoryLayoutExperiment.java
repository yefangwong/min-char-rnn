import java.util.Random;

public class MemoryLayoutExperiment {
    private static final int ROWS = 8000;
    private static final int COLS = 8000;
    private static final int ITERATIONS = 50;

    public static void main(String[] args) {
        System.out.println("Starting Memory Layout Experiment (Random Row Access)...");
        
        double[][] pointerJungle = new double[ROWS][COLS];
        double[] flatPlain = new double[ROWS * COLS];
        double[] vector = new double[COLS];
        double[] result = new double[ROWS];
        int[] rowOrder = new int[ROWS];
        
        Random rand = new Random();
        for (int i = 0; i < ROWS; i++) rowOrder[i] = i;
        // 打亂存取順序
        for (int i = 0; i < ROWS; i++) {
            int target = rand.nextInt(ROWS);
            int tmp = rowOrder[target];
            rowOrder[target] = rowOrder[i];
            rowOrder[i] = tmp;
        }

        for (int i = 0; i < COLS; i++) vector[i] = rand.nextDouble();
        for (int i = 0; i < ROWS; i++) {
            for (int j = 0; j < COLS; j++) {
                double val = rand.nextDouble();
                pointerJungle[i][j] = val;
                flatPlain[i * COLS + j] = val;
            }
        }

        // 實驗 1: 指標叢林 - 隨機行存取
        long startTime = System.currentTimeMillis();
        for (int n = 0; n < ITERATIONS; n++) {
            for (int r : rowOrder) {
                double sum = 0;
                double[] row = pointerJungle[r];
                for (int j = 0; j < COLS; j++) sum += row[j] * vector[j];
                result[r] = sum;
            }
        }
        long endTime = System.currentTimeMillis();
        System.out.printf("Test A (Pointer Jungle - Random) - Avg Time: %.2f ms\n", (endTime - startTime) / (double)ITERATIONS);

        // 實驗 2: 平坦平原 - 隨機行存取
        startTime = System.currentTimeMillis();
        for (int n = 0; n < ITERATIONS; n++) {
            for (int r : rowOrder) {
                double sum = 0;
                int offset = r * COLS;
                for (int j = 0; j < COLS; j++) sum += flatPlain[offset + j] * vector[j];
                result[r] = sum;
            }
        }
        endTime = System.currentTimeMillis();
        System.out.printf("Test B (Flat Plain - Random)    - Avg Time: %.2f ms\n", (endTime - startTime) / (double)ITERATIONS);
        
        System.out.println("Final check: " + result[rowOrder[0]]);
    }
}
