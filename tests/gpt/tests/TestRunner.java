package gpt.tests;

import gpt.model.Gpt;
import gpt.token.Bpe;

public final class TestRunner {
    static int passed = 0, failed = 0;

    static void check(String name, boolean cond, String extra) {
        if (cond) { passed++; System.out.println("  ok  " + name); }
        else { failed++; System.out.println("FAIL  " + name + "  " + extra); }
    }
    static void check(String name, boolean cond) { check(name, cond, ""); }

    public static void main(String[] args) {
        tokenizer();
        gradientCheck();
        overfitOneBatch();
        determinism();
        System.out.printf("%d passed, %d failed%n", passed, failed);
        if (failed > 0) System.exit(1);
    }

    static void tokenizer() {
        Bpe b = Bpe.train("the cat sat on the mat\nthe cat ate the rat", 20);
        String s = "the cat sat on the mat";
        int[] ids = b.encode(s);
        check("bpe: round trip", b.decode(ids).equals(s));
        check("bpe: merges shrink length", ids.length < s.getBytes().length);
        Bpe b0 = Bpe.train("hello world", 0);
        int[] raw = b0.encode("hello");
        check("bpe: no merges = raw bytes", raw.length == 5);
        check("bpe: byte round trip", b0.decode(raw).equals("hello"));
    }

    /** Finite-difference check of the analytic backward on a tiny config. */
    static void gradientCheck() {
        Gpt.Cfg cfg = new Gpt.Cfg(17, 8, 8, 2, 2);
        java.util.Random rng = new java.util.Random(7);
        Gpt net = new Gpt(cfg, rng);
        int[] toks = {3, 5, 2, 9, 1, 4, 6, 2};
        int[] tgts = {5, 2, 9, 1, 4, 6, 2, 8};

        java.util.function.Function<double[], Double> lossOf = null;
        // analytic grads
        Gpt.Act act = new Gpt.Act();
        double[] logits = net.forward(toks, act);
        double[] probs = new double[logits.length];
        double loss = Gpt.crossEntropy(logits, tgts, probs, toks.length, cfg.vocab);
        double[] dlog = probs.clone();
        for (int t = 0; t < toks.length; t++) dlog[t * cfg.vocab + tgts[t]] -= 1;
        for (int i = 0; i < dlog.length; i++) dlog[i] /= toks.length;
        net.zeroGrad();
        net.backward(act, dlog);

        double eps = 1e-5, tol = 1e-4;
        int checked = 0, bad = 0;
        // check a spread of params across every tensor kind
        for (Gpt.P p : net.all) {
            int n = p.w.length;
            int[] idxs = {0, n / 3, n / 2, n - 1, rng.nextInt(n)};
            for (int i : idxs) {
                double orig = p.w[i];
                p.w[i] = orig + eps;
                double lp = loss(net, toks, tgts);
                p.w[i] = orig - eps;
                double lm = loss(net, toks, tgts);
                p.w[i] = orig;
                double num = (lp - lm) / (2 * eps);
                double ana = p.g[i];
                double absErr = Math.abs(num - ana);
                double err = absErr / Math.max(1e-8, Math.abs(num) + Math.abs(ana));
                checked++;
                if (err > tol && absErr > 1e-7) { bad++; if (bad < 4) System.out.println("    grad mismatch: num=" + num + " ana=" + ana + " rel=" + err); }
            }
        }
        check("gradient check: analytic ~= numeric (" + checked + " params)", bad == 0, bad + " bad");
    }

    static double loss(Gpt net, int[] toks, int[] tgts) {
        Gpt.Act a = new Gpt.Act();
        double[] logits = net.forward(toks, a);
        return Gpt.crossEntropy(logits, tgts, new double[logits.length], toks.length, net.c.vocab);
    }

    /** A tiny net must be able to memorize one batch: loss collapses. */
    static void overfitOneBatch() {
        Gpt.Cfg cfg = new Gpt.Cfg(24, 16, 32, 4, 2);
        Gpt net = new Gpt(cfg, new java.util.Random(3));
        int[] toks = new int[16], tgts = new int[16];
        java.util.Random r = new java.util.Random(9);
        for (int i = 0; i < 16; i++) { toks[i] = r.nextInt(24); tgts[i] = r.nextInt(24); }
        double first = loss(net, toks, tgts), last = first;
        for (int step = 1; step <= 300; step++) {
            Gpt.Act act = new Gpt.Act();
            double[] logits = net.forward(toks, act);
            double[] probs = new double[logits.length];
            last = Gpt.crossEntropy(logits, tgts, probs, 16, cfg.vocab);
            double[] dlog = probs.clone();
            for (int t = 0; t < 16; t++) dlog[t * cfg.vocab + tgts[t]] -= 1;
            for (int i = 0; i < dlog.length; i++) dlog[i] /= 16;
            net.zeroGrad();
            net.backward(act, dlog);
            net.clipGrad(1.0);
            net.adamStep(3e-3, 0.9, 0.999, 1e-8, 0.0, step);
        }
        check("overfit: loss collapses", last < first * 0.1, "first=" + first + " last=" + last);
    }

    static void determinism() {
        Gpt.Cfg cfg = new Gpt.Cfg(24, 8, 16, 2, 1);
        int[] toks = {1, 2, 3, 4, 5, 6, 7, 8};
        Gpt a = new Gpt(cfg, new java.util.Random(42));
        Gpt b = new Gpt(cfg, new java.util.Random(42));
        double[] la = a.forward(toks, new Gpt.Act());
        double[] lb = b.forward(toks, new Gpt.Act());
        boolean same = java.util.Arrays.equals(la, lb);
        check("same seed, same logits", same);
    }
}
