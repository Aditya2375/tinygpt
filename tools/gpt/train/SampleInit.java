package gpt.train;

// Writes an untrained "step 0" sample, prepended into state/samples.txt so the
// demo's progression panel starts from random-init gibberish.
import gpt.model.Gpt;
import gpt.token.Bpe;
import java.nio.file.*;
import java.util.*;

public class SampleInit {
    public static void main(String[] args) throws Exception {
        Path out = Path.of("data/out");
        Bpe bpe = Train.loadTokenizer(out.resolve("tokenizer.json"));
        Gpt.Cfg cfg = new Gpt.Cfg(bpe.vocabSize, 64, 64, 4, 2);
        Gpt net = new Gpt(cfg, new Random(1234));
        String s = "step 0: " + Train.sample(net, bpe, "The ", 120, 0.8, 40, new Random(0));
        Path sp = out.resolve("state/samples.txt");
        String existing = Files.exists(sp) ? Files.readString(sp) : "";
        if (existing.contains("step 0:")) { System.out.println("already prepended"); return; }
        Files.writeString(sp, s + "\n---\n" + existing);
        System.out.println("prepended step-0 sample");
    }
}
