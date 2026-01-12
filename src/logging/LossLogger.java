package logging;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.time.Instant;
import java.util.Collection;
import java.util.Iterator;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;

/**
 * Financial-grade Loss Logger
 *
 * - Non-blocking training loop
 * - Async file writer
 * - Configurable log levels
 */
public class LossLogger {

    public enum LogLevel {
        NONE,
        BASIC,
        AUDIT
    }

    private final LogLevel logLevel;
    private final int logInterval;
    private final BlockingQueue<String> logQueue = new LinkedBlockingQueue<String>();

    // 使用 volatile 確保多執行緒間的可見性
    private volatile boolean running = true;

    private double lossSum = 0.0;
    private int lossCount = 0;

    private final Thread writerThread;

    public LossLogger(String filePath, LogLevel logLevel, int logInterval) {
        this.logLevel = logLevel;
        this.logInterval = logInterval;

        if (logLevel == LogLevel.NONE) {
            writerThread = null;
            return;
        }

        writerThread = new Thread(() -> writerLoop(filePath), "loss-logger-writer");
        writerThread.setDaemon(true);
        writerThread.start();

        writeHeader();
    }

    /**
     * Call this inside training loop
     */
    public void log(int step, double rawLoss, double smoothLoss) {
        if (logLevel == LogLevel.NONE) return;

        // Update accumulators (very cheap)
        lossSum += rawLoss;
        lossCount++;

        if (step % logInterval != 0) return;

        double avgLoss = lossSum / lossCount;
        lossSum = 0.0;
        lossCount = 0;

        String record;
        if (logLevel == LogLevel.BASIC) {
            record = step + "," + smoothLoss;
        } else {
            record = step + "," +
                     avgLoss + "," +
                     smoothLoss + "," +
                     Instant.now().toString();
        }

        logQueue.offer(record);
    }

    /**
     * Graceful shutdown (important for audit)
     */
    public void close() {
        running = false;
        if (writerThread != null) {
            writerThread.interrupt();
        }
    }

    /**
     * =====================================
     * Internal writer
     * =====================================
     */
    private void writeHeader() {
        if(logLevel == LogLevel.BASIC) {
            logQueue.offer("step,smooth_loss");
        } else if (logLevel == LogLevel.AUDIT) {
            logQueue.offer("step,avg_loss,smooth_loss,timestamp");
        }
    }

    private void writerLoop(String filePath) {
        try(BufferedWriter writer = new BufferedWriter(new FileWriter(filePath, false))) {
            while(running || !logQueue.isEmpty()) {
                String line = logQueue.poll();
                if (line != null) {
                    writer.write(line);
                    writer.newLine();
                } else {
                    Thread.sleep(5); // reduce CPU spin
                }
            }
            writer.flush();
        } catch (IOException | InterruptedException e) {
            // In finance system: never crash training because of logging
            System.err.println("[LossLogger] Logging error: " + e.getMessage());
        }
    }
}
