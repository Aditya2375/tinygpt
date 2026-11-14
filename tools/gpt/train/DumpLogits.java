package gpt.train;

// Loads the shipped float32 weights.bin into the Java model and prints logits
// for a fixed prompt - the reference output for the JS parity test.
import gpt.model.Gpt;
import gpt.token.Bpe;
import java.nio.*;
import java.nio.file.*;

public class DumpLogits {
    public static void main(String[] args) throws Exception {
        Path out = Path.of("data/out");
        Bpe bpe = Train.loadTokenizer(out.resolve("tokenizer.json"));
        Gpt.Cfg cfg = new Gpt.Cfg(bpe.vocabSize, 64, 64, 4, 2);
        Gpt net = new Gpt(cfg, new java.util.Random(0)); // values overwritten below
        ByteBuffer buf = ByteBuffer.wrap(Files.readAllBytes(out.resolve("weights.bin")))
                .order(ByteOrder.LITTLE_ENDIAN);
        for (Gpt.P p : net.all) for (int i = 0; i < p.w.length; i++) p.w[i] = buf.getFloat();
        int[] toks = bpe.encode("The city was");
        double[] logits = net.forward(toks, new Gpt.Act());
        int V = cfg.vocab, T = toks.length, off = (T - 1) * V;
        double norm = 0, max = 0;
        for (int t = 0; t < T; t++) for (int j = 0; j < V; j++) {
            double v = logits[t * V + j];
            norm += v * v; if (Math.abs(v) > max) max = Math.abs(v);
        }
        System.out.printf("tokens=%s%n", java.util.Arrays.toString(toks));
        System.out.printf("norm=%.9f max=%.9f%n", Math.sqrt(norm), max);
        StringBuilder sb = new StringBuilder("last16=");
        for (int j = 0; j < 16; j++) sb.append(String.format("%.9g%s", logits[off + j], j < 15 ? "," : ""));
        System.out.println(sb);
    }
}
