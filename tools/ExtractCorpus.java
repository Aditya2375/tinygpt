import java.nio.file.*;
import java.util.*;
import java.util.zip.GZIPInputStream;
import sift.parse.HtmlDocument;
import sift.parse.HtmlParser;

/** Extracts plain article text from the sift crawl into one corpus file. */
public class ExtractCorpus {
    public static void main(String[] args) throws Exception {
        Path pages = Path.of(args[0]);
        Path out = Path.of(args[1]);
        List<Path> files = new ArrayList<>();
        try (var st = Files.list(pages)) {
            st.filter(p -> p.toString().endsWith(".html.gz")).sorted().forEach(files::add);
        }
        StringBuilder sb = new StringBuilder();
        int kept = 0;
        for (Path f : files) {
            byte[] raw;
            try (var in = new GZIPInputStream(Files.newInputStream(f))) {
                raw = in.readAllBytes();
            }
            HtmlDocument doc = HtmlParser.parse(new String(raw, java.nio.charset.StandardCharsets.UTF_8),
                    "https://simple.wikipedia.org/", "/wiki/", true);
            String t = doc.text.replaceAll("[ \t]+", " ").trim();
            if (t.length() < 200) continue;
            sb.append(t).append('\n');
            kept++;
        }
        Files.writeString(out, sb.toString());
        System.out.println("articles=" + kept + " chars=" + sb.length());
    }
}
