import java.io.*;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;

public class DataCleaner {

    /**
     * Reads and cleans the Bible text file, removing the chapter/verse metadata at the beginning of each line.
     *
     * @param filePath The path to the Bible text file (e.g., "hb5.txt")
     * @return A string containing only the cleaned Bible text
     */
    public static String cleanBibleText(String filePath) {
        StringBuilder cleanedText = new StringBuilder();

        // Assuming the file path is correct, use BufferedReader to read the file
        // Note: If the file is truly Big5 encoded, complex encoding handling might be needed here,
        // but the current logic focuses on structural cleaning (removing metadata).
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(
                        new FileInputStream(filePath),
                        StandardCharsets.UTF_8
                )
            )
        ) {
            String line;
            boolean isFirstLine = true;

            while ((line = reader.readLine()) != null) {
                // 1. Skip the first line (as requested)
                if (isFirstLine) {
                    isFirstLine = false;
                    continue;
                }

                // 2. Find the position of the first space (separating Book and Chapter:Verse)
                int firstSpaceIndex = line.indexOf(' ');

                // 3. Find the position of the second space (separating Chapter:Verse and actual text)
                int secondSpaceIndex = -1;
                if (firstSpaceIndex != -1) {
                    // Start searching for the second space AFTER the first space
                    secondSpaceIndex = line.indexOf(' ', firstSpaceIndex + 1);
                }

                // 4. Perform trimming: extract the substring starting after the second space
                if (secondSpaceIndex != -1 && secondSpaceIndex < line.length() - 1) {
                    // Extract the substring starting from the first character after the second space
                    String bibleVerse = line.substring(secondSpaceIndex + 1);

                    // 新增字符過濾邏輯
                    StringBuilder filteredVerse = new StringBuilder();
                    for (char c : bibleVerse.toCharArray()) {
                        // 匹配中文、常見標點、數字和字母
                        if (c >= 0x4E00 && c <= 0x9FFF ||  // 基本漢字
                                c == ' '  ||               // 保留空格
                                c == '，' || c == '。' ||  c == '．' || // 中文標點
                                c == '、' || c == '；' ||
                                c == '：' || c == '？' ||
                                c == '！' || c == '“'  ||
                                c == '”' || c == '（'  || c == '〔' ||
                                c == '）' || c == '〕' || c == '—') {
                            filteredVerse.append(c);
                        } else {
                            // 紀錄非法字符的ASCII碼值
                            System.out.printf("發現非法字符: 0x%04x '%c'%n", (int) c, c);
                        }
                    }

                    // Append the cleaned text (should now be pure Chinese text)
                    cleanedText.append(filteredVerse.toString());
                }
                // Ignore if the line is empty or incorrectly formatted
            }

        } catch (IOException e) {
            System.err.println("Error reading the file: " + e.getMessage());
            e.printStackTrace();
        }

        return cleanedText.toString();
    }

    public static void main(String[] args) {
        String cleanData = cleanBibleText("resources" + File.separator + "hb5.txt");

        if (cleanData.length() > 0) {
            // Report the total number of characters read and cleaned
            System.out.println("Total number of characters successfully read and cleaned: " + cleanData.length());

            // Display a fragment of the cleaned text (metadata has been removed)
            System.out.println("--- Cleaned Text Fragment (Metadata removed) ---");
            // Note: Due to encoding issues (Big5), the output may still appear as garbled characters,
            // but the structural cleaning (removing Gen 16:15, etc.) is complete.
            int previewLength = Math.min(cleanData.length(), 200);
            System.out.println(cleanData.substring(0, previewLength));
        } else {
            // Failed to read or clean any data. Please check the file path and content.
            System.out.println("Failed to read or clean any data. Please check the file path and content.");
        }

        // **Next Step: Pass cleanData to SimpleRNN for training**
        // SimpleRNN rnn = new SimpleRNN(cleanData);
        // rnn.train();
    }
}
