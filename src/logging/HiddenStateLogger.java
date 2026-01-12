package logging;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Logger for saving hidden states during training.
 *
 * - Asynchronous file writer to prevent blocking the training loop.
 * - Saves data in CSV format for easy analysis.
 * - Supports dynamic hidden state sizes.
 */
public class HiddenStateLogger {

    private final BlockingQueue<String> logQueue = new LinkedBlockingQueue<>();
    private volatile boolean running = true;
    private final Thread writerThread;

    /**
     * Constructs a HiddenStateLogger.
     * @param filePath The path to the log file.
     * @param hiddenSize The size of the hidden state vector (e.g., HIDDEN_SIZE).
     */
    public HiddenStateLogger(String filePath, int hiddenSize) {
        this.writerThread = new Thread(() -> writerLoop(filePath), "hidden-state-logger-writer");
        // Set as a daemon thread.
        // This ensures the JVM can exit even if this background thread is still running,
        // as long as all non-daemon threads (like the main thread) have finished.
        this.writerThread.setDaemon(true);
        writeHeader(hiddenSize);
        this.writerThread.start();
    }

    /**
     * Logs a single hidden state vector for a given iteration and time step.
     *
     * @param iteration The current training iteration (n).
     * @param timeStep The time step within the sequence (t).
     * @param token The input token character for this time step.
     * @param hiddenState The hidden state vector (h[t]).
     */
    public void log(int iteration, int timeStep, char token, double[] hiddenState) {
        if (!running) return;

        StringBuilder sb = new StringBuilder();
        sb.append(iteration).append(',')
                .append(timeStep).append(',')
                .append(token);

        for (double value : hiddenState) {
            sb.append(',').append(String.format("%.8f", value)); // Format for precision
        }

        logQueue.offer(sb.toString());
    }

    /**
     * Signals the logger to shut down gracefully, writing any remaining logs.
     */
    public void close() {
        running = false;
        writerThread.interrupt(); // Interrupt sleep to exit loop faster
        try {
            writerThread.join(); // Wait for the writer thread to finish
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            System.err.println("[HiddenStateLogger] Interrupted while waiting for writer to close.");
        }
    }

    private void writeHeader(int hiddenSize) {
        // Create header like: "iteration,time_step,token,h0,h1,h2,..."
        String header = "iteration,time_step,token," +
                IntStream.range(0, hiddenSize)
                        .mapToObj(i -> "h" + i)
                        .collect(Collectors.joining(","));
        logQueue.offer(header);
    }

    private void writerLoop(String filePath) {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath))) {
            while (running || !logQueue.isEmpty()) {
                try {
                    String line = logQueue.poll(5, java.util.concurrent.TimeUnit.MILLISECONDS);
                    if (line != null) {
                        writer.write(line);
                        writer.newLine();
                    }
                } catch (InterruptedException e) {
                    // This is expected when close() is called.
                    // Continue loop to drain the queue, then exit.
                }
            }
            writer.flush();
        } catch (IOException e) {
            System.err.println("[HiddenStateLogger] Critical logging error: " + e.getMessage());
        }
    }
}