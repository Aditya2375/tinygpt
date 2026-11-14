package gpt.train;

import gpt.model.Gpt;
import gpt.token.Bpe;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import java.util.Random;

/**
 * Trains the small GPT in resumable slices (the sandbox kills long jobs).
 * Usage: Train <corpus> <outDir> <stepsThisRun> <batch>
 * State (weights, Adam moments, step counter, loss curve, samples) lives in
 * <outDir>/state and resumes across invocations; the demo artifacts
 * (weights.bin, tokenizer.json, train.json) are written on the final step.
 */
public final class Train {
    static final int TOTAL_STEPS = 900;
    static final double PEAK_LR = 3e-3, MIN_LR = 3e-4;
    static final int WARMUP = 60;

    public static void main(String[] args) throws Exception {
        Path corpusPath = Path.of(args[0]);
        Path out = Path.of(args[1]);
        int stepsThisRun = Integer.parseInt(args[2]);
        int B = Integer.parseInt(args[3]);
        Path state = out.resolve("state");
        Files.createDirectories(state);

        // Tokenizer: cached after the first slice.
        Path tokPath = out.resolve("tokenizer.json");
        Bpe bpe;
        if (Files.exists(tokPath)) {
            bpe = loadTokenizer(tokPath);
        } else {
            String corpus = Files.readString(corpusPath);
            String bpeSlice = corpus.substring(0, Math.min(corpus.length(), 512 * 1024));
            bpe = Bpe.train(bpeSlice, 1200);
            saveTokenizer(bpe, tokPath);
        }

        String corpus = Files.readString(corpusPath);
        int[] allIds = bpe.encode(corpus.substring(0, 2 * 1024 * 1024));
        int[] valIds = bpe.encode(corpus.substring(2 * 1024 * 1024, 2 * 1024 * 1024 + 100 * 1023));

        Gpt.Cfg cfg = new Gpt.Cfg(bpe.vocabSize, 64, 64, 4, 2);
        int params;
        Gpt net;
        int startStep = 1;
        Path metaPath = state.resolve("meta.properties");
        if (Files.exists(state.resolve("weights.bin"))) {
            net = new Gpt(cfg, new Random(0)); // overwritten by load
            loadState(net, state);
            Properties pr = new Properties();
            try (var in = Files.newInputStream(metaPath)) { pr.load(in); }
            startStep = Integer.parseInt(pr.getProperty("step")) + 1;
        } else {
            net = new Gpt(cfg, new Random(1234));
        }
        params = net.all.stream().mapToInt(p -> p.w.length).sum();
        if (startStep == 1) {
            System.out.printf("tokenizer: vocab=%d; train=%d tokens val=%d tokens; params=%d%n",
                    bpe.vocabSize, allIds.length, valIds.length, params);
        }

        final int T = cfg.ctx;
        Random sampler = new Random(99 + startStep); // deterministic per slice
        Path curvePath = state.resolve("curve.txt");
        long t0 = System.currentTimeMillis();
        int step = startStep;
        for (; step < startStep + stepsThisRun && step <= TOTAL_STEPS; step++) {
            double lr = step < WARMUP ? PEAK_LR * step / WARMUP
                    : MIN_LR + 0.5 * (PEAK_LR - MIN_LR) * (1 + Math.cos(Math.PI * (step - WARMUP) / (double) (TOTAL_STEPS - WARMUP)));
            net.zeroGrad();
            double loss = 0;
            for (int b = 0; b < B; b++) {
                int off = sampler.nextInt(allIds.length - T - 1);
                int[] toks = new int[T], tgts = new int[T];
                System.arraycopy(allIds, off, toks, 0, T);
                System.arraycopy(allIds, off + 1, tgts, 0, T);
                Gpt.Act act = new Gpt.Act();
                double[] logits = net.forward(toks, act);
                double[] probs = new double[logits.length];
                loss += Gpt.crossEntropy(logits, tgts, probs, T, cfg.vocab);
                double[] dlog = probs.clone();
                for (int t = 0; t < T; t++) dlog[t * cfg.vocab + tgts[t]] -= 1;
                double scale = 1.0 / (T * B);
                for (int i = 0; i < dlog.length; i++) dlog[i] *= scale;
                net.backward(act, dlog);
            }
            loss /= B;
            net.clipGrad(1.0);
            net.adamStep(lr, 0.9, 0.999, 1e-8, 0.01, step);
            if (step % 5 == 0) {
                Files.writeString(curvePath, step + " " + r4(loss) + "\n",
                        java.nio.file.StandardOpenOption.CREATE, java.nio.file.StandardOpenOption.APPEND);
            }
            if (step % 25 == 0 || step == 1) {
                System.out.printf("step %4d/%d  loss %.4f  lr %.2e  %ds%n",
                        step, TOTAL_STEPS, loss, lr, (System.currentTimeMillis() - t0) / 1000);
            }
            if (step == 300 || step == 450 || step == 600 || step == 750 || step == TOTAL_STEPS) {
                String s = "step " + step + ": " + sample(net, bpe, "The ", 120, 0.8, 40, new Random(step));
                Files.writeString(state.resolve("samples.txt"), s + "\n---\n",
                        java.nio.file.StandardOpenOption.CREATE, java.nio.file.StandardOpenOption.APPEND);
            }
        }
        int lastStep = step - 1;
        saveState(net, state);
        Properties pr = new Properties();
        pr.setProperty("step", String.valueOf(lastStep));
        try (var o = Files.newOutputStream(metaPath)) { pr.store(o, ""); }
        System.out.println("state saved at step " + lastStep);

        if (lastStep >= TOTAL_STEPS) {
            finalizeArtifacts(net, bpe, cfg, out, allIds, valIds, T, params);
        }
    }

    static void finalizeArtifacts(Gpt net, Bpe bpe, Gpt.Cfg cfg, Path out,
                                  int[] allIds, int[] valIds, int T, int params) throws Exception {
        double valLoss = 0;
        int vb = 32;
        Random vs = new Random(7);
        for (int b = 0; b < vb; b++) {
            int off = vs.nextInt(valIds.length - T - 1);
            int[] toks = new int[T], tgts = new int[T];
            System.arraycopy(valIds, off, toks, 0, T);
            System.arraycopy(valIds, off + 1, tgts, 0, T);
            double[] logits = net.forward(toks, new Gpt.Act());
            valLoss += Gpt.crossEntropy(logits, tgts, new double[logits.length], T, cfg.vocab);
        }
        valLoss /= vb;

        // Curve from state.
        List<String> lines = Files.readAllLines(out.resolve("state/curve.txt"));
        StringBuilder curveJson = new StringBuilder("[");
        double lastTrain = 0;
        for (int i = 0; i < lines.size(); i++) {
            if (i > 0) curveJson.append(',');
            String v = lines.get(i).split(" ")[1];
            curveJson.append(v);
            lastTrain = Double.parseDouble(v);
        }
        curveJson.append(']');

        String samples = Files.exists(out.resolve("state/samples.txt"))
                ? Files.readString(out.resolve("state/samples.txt")) : "";
        StringBuilder samplesJson = new StringBuilder("[");
        boolean first = true;
        for (String s : samples.split("---\n")) {
            s = s.strip();
            if (s.isEmpty()) continue;
            if (!first) samplesJson.append(',');
            samplesJson.append(quote(s));
            first = false;
        }
        samplesJson.append(']');

        ByteBuffer buf = ByteBuffer.allocate(params * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (Gpt.P p : net.all) for (double x : p.w) buf.putFloat((float) x);
        Files.write(out.resolve("weights.bin"), buf.array());

        String meta = "{\"config\":{\"vocab\":" + cfg.vocab
                + ",\"ctx\":" + cfg.ctx + ",\"dModel\":" + cfg.dModel
                + ",\"nHead\":" + cfg.nHead + ",\"nLayer\":" + cfg.nLayer
                + ",\"params\":" + params + "},"
                + "\"trainTokens\":" + allIds.length
                + ",\"steps\":" + TOTAL_STEPS
                + ",\"trainLoss\":" + r4(lastTrain) + ",\"valLoss\":" + r4(valLoss)
                + ",\"curve\":" + curveJson + ",\"samples\":" + samplesJson + "}\n";
        Files.writeString(out.resolve("train.json"), meta);
        System.out.printf("FINAL: train %.4f val %.4f; artifacts in %s%n", lastTrain, valLoss, out);
    }

    static void saveState(Gpt net, Path state) throws Exception {
        for (String f : new String[]{"weights.bin", "adam.bin"}) {
            // weights.bin holds w; adam.bin holds m then v, both float64 LE.
        }
        int n = net.all.stream().mapToInt(p -> p.w.length).sum();
        ByteBuffer w = ByteBuffer.allocate(n * 8).order(ByteOrder.LITTLE_ENDIAN);
        ByteBuffer adam = ByteBuffer.allocate(n * 16).order(ByteOrder.LITTLE_ENDIAN);
        for (Gpt.P p : net.all) for (double x : p.w) w.putDouble(x);
        for (Gpt.P p : net.all) { for (double x : p.m) adam.putDouble(x); }
        for (Gpt.P p : net.all) { for (double x : p.v) adam.putDouble(x); }
        Files.write(state.resolve("weights.bin"), w.array());
        Files.write(state.resolve("adam.bin"), adam.array());
    }

    static void loadState(Gpt net, Path state) throws Exception {
        ByteBuffer w = ByteBuffer.wrap(Files.readAllBytes(state.resolve("weights.bin"))).order(ByteOrder.LITTLE_ENDIAN);
        ByteBuffer adam = ByteBuffer.wrap(Files.readAllBytes(state.resolve("adam.bin"))).order(ByteOrder.LITTLE_ENDIAN);
        for (Gpt.P p : net.all) for (int i = 0; i < p.w.length; i++) p.w[i] = w.getDouble();
        for (Gpt.P p : net.all) for (int i = 0; i < p.m.length; i++) p.m[i] = adam.getDouble();
        for (Gpt.P p : net.all) for (int i = 0; i < p.v.length; i++) p.v[i] = adam.getDouble();
    }

    static void saveTokenizer(Bpe bpe, Path p) throws Exception {
        StringBuilder tj = new StringBuilder();
        tj.append("{\"vocabSize\":").append(bpe.vocabSize).append(",\"merges\":[");
        for (int i = 0; i < bpe.merges.length; i++) {
            if (i > 0) tj.append(',');
            tj.append('[').append(bpe.merges[i][0]).append(',').append(bpe.merges[i][1]).append(',').append(bpe.merges[i][2]).append(']');
        }
        tj.append("]}\n");
        Files.writeString(p, tj.toString());
    }

    static Bpe loadTokenizer(Path p) throws Exception {
        String s = Files.readString(p).strip();
        List<int[]> merges = new ArrayList<>();
        int i = s.indexOf("[", s.indexOf("\"merges\""));
        int depth = 0, num = 0; List<Integer> triple = new ArrayList<>();
        for (i = s.indexOf('[', s.indexOf("\"merges\"")); i < s.length(); i++) {
            char ch = s.charAt(i);
            if (ch == '[') depth++;
            else if (ch == ']') {
                depth--;
                if (depth == 1 && !triple.isEmpty()) { /* unreachable */ }
                if (depth == 0) break;
                if (triple.size() == 3) { merges.add(triple.stream().mapToInt(x -> x).toArray()); triple.clear(); }
            } else if (Character.isDigit(ch)) {
                int j = i;
                while (j < s.length() && Character.isDigit(s.charAt(j))) j++;
                triple.add(Integer.parseInt(s.substring(i, j)));
                i = j - 1;
            }
        }
        int[][] m = merges.toArray(new int[0][]);
        // Bpe constructor is private; rebuild via a tiny trainer-side factory.
        return Bpe.fromMerges(m);
    }

    static double r4(double x) { return Math.round(x * 10000.0) / 10000.0; }

    static String quote(String s) {
        StringBuilder sb = new StringBuilder(s.length() + 2).append('"');
        for (int i = 0; i < s.length(); i++) {
            char ch = s.charAt(i);
            switch (ch) {
                case '"' -> sb.append("\\\"");
                case '\\' -> sb.append("\\\\");
                case '\n' -> sb.append("\\n");
                case '\r' -> sb.append("\\r");
                case '\t' -> sb.append("\\t");
                default -> {
                    if (ch < 0x20) sb.append(String.format("\\u%04x", (int) ch));
                    else sb.append(ch);
                }
            }
        }
        return sb.append('"').toString();
    }

    public static String sample(Gpt net, Bpe bpe, String prompt, int nTokens,
                                double temperature, int topK, Random rng) {
        List<Integer> ids = new ArrayList<>();
        for (int id : bpe.encode(prompt)) ids.add(id);
        for (int gen = 0; gen < nTokens; gen++) {
            int start = Math.max(0, ids.size() - net.c.ctx);
            int[] ctx = new int[ids.size() - start];
            for (int i = 0; i < ctx.length; i++) ctx[i] = ids.get(start + i);
            double[] logits = net.forward(ctx, new Gpt.Act());
            int last = (ctx.length - 1) * net.c.vocab;
            int next = temperature <= 0 ? argmax(logits, last, net.c.vocab)
                    : sampleTopK(logits, last, net.c.vocab, temperature, topK, rng);
            ids.add(next);
        }
        int[] arr = ids.stream().mapToInt(Integer::intValue).toArray();
        return bpe.decode(arr);
    }

    static int argmax(double[] x, int off, int n) {
        int bi = 0;
        for (int i = 1; i < n; i++) if (x[off + i] > x[off + bi]) bi = i;
        return bi;
    }

    static int sampleTopK(double[] logits, int off, int n, double temperature, int k, Random rng) {
        Integer[] idx = new Integer[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        java.util.Arrays.sort(idx, (a, b) -> Double.compare(logits[off + b], logits[off + a]));
        double max = logits[off + idx[0]];
        double[] p = new double[k];
        double sum = 0;
        for (int i = 0; i < k; i++) { p[i] = Math.exp((logits[off + idx[i]] - max) / temperature); sum += p[i]; }
        double r = rng.nextDouble() * sum, acc = 0;
        for (int i = 0; i < k; i++) { acc += p[i]; if (r <= acc) return idx[i]; }
        return idx[k - 1];
    }
}
