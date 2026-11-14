package gpt.train;

// Regenerates the published artifacts (weights.bin, train.json) from the saved
// training state without retraining. Used after fixing serialization bugs.
import gpt.model.Gpt;
import gpt.token.Bpe;
import java.nio.file.*;
import java.util.*;

public class Finalize {
    public static void main(String[] args) throws Exception {
        Path out = Path.of("data/out");
        Bpe bpe = Train.loadTokenizer(out.resolve("tokenizer.json"));
        String corpus = Files.readString(Path.of("data/corpus.txt"));
        int[] allIds = bpe.encode(corpus.substring(0, 2 * 1024 * 1024));
        int[] valIds = bpe.encode(corpus.substring(2 * 1024 * 1024, 2 * 1024 * 1024 + 100 * 1023));
        Gpt.Cfg cfg = new Gpt.Cfg(bpe.vocabSize, 64, 64, 4, 2);
        Gpt net = new Gpt(cfg, new Random(0));
        Train.loadState(net, out.resolve("state"));
        int params = net.all.stream().mapToInt(p -> p.w.length).sum();
        Train.finalizeArtifacts(net, bpe, cfg, out, allIds, valIds, 64, params);
    }
}
