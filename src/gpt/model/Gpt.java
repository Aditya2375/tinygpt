package gpt.model;

import java.util.ArrayList;
import java.util.List;

/**
 * A small GPT, from scratch. Token + positional embeddings, pre-LN transformer
 * blocks (causal multi-head attention, GELU MLP), final LN, tied output head.
 * Plain double arrays, no libraries; the backward pass is hand-derived and
 * gradient-checked in tests.
 */
public final class Gpt {

    public static final class Cfg {
        public int vocab, ctx, dModel, nHead, nLayer;
        public Cfg(int vocab, int ctx, int dModel, int nHead, int nLayer) {
            this.vocab = vocab; this.ctx = ctx; this.dModel = dModel;
            this.nHead = nHead; this.nLayer = nLayer;
            if (dModel % nHead != 0) throw new IllegalArgumentException("dModel % nHead != 0");
        }
        public int headDim() { return dModel / nHead; }
    }

    /** A weight tensor with its gradient and Adam moments, flat row-major. */
    public static final class P {
        public final double[] w, g, m, v;
        public P(int n) { w = new double[n]; g = new double[n]; m = new double[n]; v = new double[n]; }
    }

    public final Cfg c;
    public final P tokEmb, posEmb, lnF;            // lnF: [2*dModel] gamma||beta
    public final P[][] blocks;                     // per layer: ln1, qkvW, qkvB, projW, projB, ln2, fc1W, fc1B, fc2W, fc2B
    public final List<P> all = new ArrayList<>();

    private static final int LN1 = 0, QKVW = 1, QKVB = 2, PROJW = 3, PROJB = 4,
            LN2 = 5, FC1W = 6, FC1B = 7, FC2W = 8, FC2B = 9;

    public Gpt(Cfg c, java.util.Random rng) {
        this.c = c;
        int d = c.dModel;
        tokEmb = reg(new P(c.vocab * d));
        posEmb = reg(new P(c.ctx * d));
        lnF = reg(new P(2 * d));
        blocks = new P[c.nLayer][];
        for (int l = 0; l < c.nLayer; l++) {
            blocks[l] = new P[]{
                reg(new P(2 * d)), reg(new P(d * 3 * d)), reg(new P(3 * d)),
                reg(new P(d * d)), reg(new P(d)),
                reg(new P(2 * d)), reg(new P(d * 4 * d)), reg(new P(4 * d)),
                reg(new P(4 * d * d)), reg(new P(d)),
            };
        }
        // Init: embeddings and matrices ~N(0, 0.02); LN gammas 1, betas 0; biases 0.
        initNormal(tokEmb, rng, 0.02);
        initNormal(posEmb, rng, 0.02);
        for (P[] b : blocks) {
            initNormal(b[QKVW], rng, 0.02);
            initNormal(b[PROJW], rng, 0.02 / Math.sqrt(2 * c.nLayer)); // GPT-2 residual scaling
            initNormal(b[FC1W], rng, 0.02);
            initNormal(b[FC2W], rng, 0.02 / Math.sqrt(2 * c.nLayer));
            initLn(b[LN1]); initLn(b[LN2]);
        }
        initLn(lnF);
    }

    private P reg(P p) { all.add(p); return p; }
    private static void initNormal(P p, java.util.Random rng, double sd) {
        for (int i = 0; i < p.w.length; i++) p.w[i] = rng.nextGaussian() * sd;
    }
    private static void initLn(P p) {
        int d = p.w.length / 2;
        for (int i = 0; i < d; i++) p.w[i] = 1.0;
    }

    // ---------- math helpers (row-major flat arrays) ----------

    /** out[t][:] = x[t][:] @ W[fanIn][fanOut] + b[fanOut] */
    static void linear(double[] x, double[] w, double[] b, double[] out, int T, int fanIn, int fanOut) {
        for (int t = 0; t < T; t++) {
            int xo = t * fanIn, oo = t * fanOut;
            if (b != null) System.arraycopy(b, 0, out, oo, fanOut);
            else java.util.Arrays.fill(out, oo, oo + fanOut, 0);
            for (int i = 0; i < fanIn; i++) {
                double xi = x[xo + i];
                if (xi == 0) continue;
                int wo = i * fanOut;
                for (int j = 0; j < fanOut; j++) out[oo + j] += xi * w[wo + j];
            }
        }
    }

    static void linearBackward(double[] x, double[] w, double[] dOut, double[] dx, P W, P B,
                               int T, int fanIn, int fanOut) {
        for (int t = 0; t < T; t++) {
            int xo = t * fanIn, oo = t * fanOut;
            for (int i = 0; i < fanIn; i++) {
                double s = 0;
                for (int j = 0; j < fanOut; j++) s += dOut[oo + j] * w[i * fanOut + j];
                dx[xo + i] += s;
            }
        }
        for (int i = 0; i < fanIn; i++)
            for (int j = 0; j < fanOut; j++) {
                double s = 0;
                for (int t = 0; t < T; t++) s += x[t * fanIn + i] * dOut[t * fanOut + j];
                W.g[i * fanOut + j] += s;
            }
        if (B != null)
            for (int j = 0; j < fanOut; j++) {
                double s = 0;
                for (int t = 0; t < T; t++) s += dOut[t * fanOut + j];
                B.g[j] += s;
            }
    }
    /** LayerNorm over the last dim; cache holds {mean, rstd} per row. gamma||beta packed. */
    static void layernorm(double[] x, P ln, double[] out, double[] cache, int T, int d) {
        for (int t = 0; t < T; t++) {
            int o = t * d;
            double mean = 0;
            for (int i = 0; i < d; i++) mean += x[o + i];
            mean /= d;
            double var = 0;
            for (int i = 0; i < d; i++) { double z = x[o + i] - mean; var += z * z; }
            double rstd = 1.0 / Math.sqrt(var / d + 1e-5);
            cache[t * 2] = mean; cache[t * 2 + 1] = rstd;
            for (int i = 0; i < d; i++)
                out[o + i] = (x[o + i] - mean) * rstd * ln.w[i] + ln.w[d + i];
        }
    }

    static void layernormBackward(double[] x, P ln, double[] dOut, double[] dx, double[] cache, int T, int d) {
        for (int t = 0; t < T; t++) {
            int o = t * d;
            double mean = cache[t * 2], rstd = cache[t * 2 + 1];
            double dmeanPart = 0, dvarPart = 0;
            for (int i = 0; i < d; i++) {
                double xhat = (x[o + i] - mean) * rstd;
                ln.g[i] += dOut[o + i] * xhat;
                ln.g[d + i] += dOut[o + i];
                double dxhat = dOut[o + i] * ln.w[i];
                dvarPart += dxhat * (x[o + i] - mean);
                dmeanPart += dxhat;
            }
            dvarPart *= -0.5 * rstd * rstd * rstd;
            double dvar = dvarPart;
            for (int i = 0; i < d; i++) {
                double dxhat = dOut[o + i] * ln.w[i];
                dx[o + i] += dxhat * rstd
                        + dvar * 2.0 * (x[o + i] - mean) / d
                        - (dmeanPart * rstd) / d;
            }
        }
    }

    static double gelu(double x) {
        return 0.5 * x * (1 + Math.tanh(Math.sqrt(2.0 / Math.PI) * (x + 0.044715 * x * x * x)));
    }
    static double geluGrad(double x) {
        double u = Math.sqrt(2.0 / Math.PI) * (x + 0.044715 * x * x * x);
        double th = Math.tanh(u);
        double sech2 = 1 - th * th;
        double du = Math.sqrt(2.0 / Math.PI) * (1 + 3 * 0.044715 * x * x);
        return 0.5 * (1 + th) + 0.5 * x * sech2 * du;
    }

    // ---------- forward ----------

    /** Per-sequence activation cache for the backward pass. */
    public static final class Act {
        double[] x;            // current stream, [T][d], embedding output
        double[][] ln1o, attOut, projIn, ln2o, fc1o, g, projOut, fc2o;
        double[][][] q, k, v, probs;   // per layer, per head: [T][dh] / [T][T]
        double[][] lnCache1, lnCache2, lnCacheF;
        double[] xf;           // post final-LN
        int[] tokens;
    }

    public double[] forward(int[] tokens, Act act) {
        int T = tokens.length, d = c.dModel, H = c.nHead, dh = c.headDim();
        act.tokens = tokens;
        act.x = new double[T * d];
        for (int t = 0; t < T; t++)
            for (int i = 0; i < d; i++)
                act.x[t * d + i] = tokEmb.w[tokens[t] * d + i] + posEmb.w[t * d + i];

        act.ln1o = new double[c.nLayer][]; act.attOut = new double[c.nLayer][];
        act.projIn = new double[c.nLayer][]; act.ln2o = new double[c.nLayer][];
        act.fc1o = new double[c.nLayer][]; act.g = new double[c.nLayer][];
        act.projOut = new double[c.nLayer][]; act.fc2o = new double[c.nLayer][];
        act.q = new double[c.nLayer][H][]; act.k = new double[c.nLayer][H][];
        act.v = new double[c.nLayer][H][]; act.probs = new double[c.nLayer][H][];
        act.lnCache1 = new double[c.nLayer][]; act.lnCache2 = new double[c.nLayer][];

        double[] cur = act.x;
        for (int l = 0; l < c.nLayer; l++) {
            P[] b = blocks[l];
            act.lnCache1[l] = new double[T * 2];
            act.ln1o[l] = new double[T * d];
            layernorm(cur, b[LN1], act.ln1o[l], act.lnCache1[l], T, d);

            double[] qkv = new double[T * 3 * d];
            linear(act.ln1o[l], b[QKVW].w, b[QKVB].w, qkv, T, d, 3 * d);

            double[] attnAll = new double[T * d];
            for (int h = 0; h < H; h++) {
                double[] q = new double[T * dh], k = new double[T * dh], v = new double[T * dh];
                for (int t = 0; t < T; t++)
                    for (int i = 0; i < dh; i++) {
                        q[t * dh + i] = qkv[t * 3 * d + h * dh + i];
                        k[t * dh + i] = qkv[t * 3 * d + d + h * dh + i];
                        v[t * dh + i] = qkv[t * 3 * d + 2 * d + h * dh + i];
                    }
                double[] probs = new double[T * T];
                double scale = 1.0 / Math.sqrt(dh);
                for (int t = 0; t < T; t++) {
                    double max = Double.NEGATIVE_INFINITY;
                    for (int s = 0; s <= t; s++) {
                        double dot = 0;
                        for (int i = 0; i < dh; i++) dot += q[t * dh + i] * k[s * dh + i];
                        probs[t * T + s] = dot * scale;
                        if (probs[t * T + s] > max) max = probs[t * T + s];
                    }
                    double sum = 0;
                    for (int s = 0; s <= t; s++) { probs[t * T + s] = Math.exp(probs[t * T + s] - max); sum += probs[t * T + s]; }
                    for (int s = 0; s <= t; s++) probs[t * T + s] /= sum;
                }
                double[] ctxv = new double[T * dh];
                for (int t = 0; t < T; t++)
                    for (int i = 0; i < dh; i++) {
                        double s = 0;
                        for (int u = 0; u <= t; u++) s += probs[t * T + u] * v[u * dh + i];
                        ctxv[t * dh + i] = s;
                    }
                act.q[l][h] = q; act.k[l][h] = k; act.v[l][h] = v; act.probs[l][h] = probs;
                for (int t = 0; t < T; t++)
                    for (int i = 0; i < dh; i++) attnAll[t * d + h * dh + i] = ctxv[t * dh + i];
            }
            act.projIn[l] = attnAll;
            act.projOut[l] = new double[T * d];
            linear(attnAll, b[PROJW].w, b[PROJB].w, act.projOut[l], T, d, d);
            double[] res1 = new double[T * d];
            for (int i = 0; i < T * d; i++) res1[i] = cur[i] + act.projOut[l][i];

            act.lnCache2[l] = new double[T * 2];
            act.ln2o[l] = new double[T * d];
            layernorm(res1, b[LN2], act.ln2o[l], act.lnCache2[l], T, d);
            act.fc1o[l] = new double[T * 4 * d];
            linear(act.ln2o[l], b[FC1W].w, b[FC1B].w, act.fc1o[l], T, d, 4 * d);
            act.g[l] = new double[T * 4 * d];
            for (int i = 0; i < T * 4 * d; i++) act.g[l][i] = gelu(act.fc1o[l][i]);
            act.fc2o[l] = new double[T * d];
            linear(act.g[l], b[FC2W].w, b[FC2B].w, act.fc2o[l], T, 4 * d, d);
            cur = new double[T * d];
            for (int i = 0; i < T * d; i++) cur[i] = res1[i] + act.fc2o[l][i];
        }
        double[] lnCacheF = new double[T * 2];
        act.lnCacheF = new double[][]{lnCacheF};
        act.xf = new double[T * d];
        layernorm(cur, lnF, act.xf, lnCacheF, T, d);
        // Tied head: logits = xf @ tokEmb^T
        double[] logits = new double[T * c.vocab];
        for (int t = 0; t < T; t++)
            for (int j = 0; j < c.vocab; j++) {
                double s = 0;
                for (int i = 0; i < d; i++) s += act.xf[t * d + i] * tokEmb.w[j * d + i];
                logits[t * c.vocab + j] = s;
            }
        return logits;
    }

    /** Cross-entropy loss over next-token targets; fills softmax probs into out. */
    public static double crossEntropy(double[] logits, int[] targets, double[] probsOut, int T, int vocab) {
        double loss = 0;
        for (int t = 0; t < T; t++) {
            int o = t * vocab;
            double max = Double.NEGATIVE_INFINITY;
            for (int j = 0; j < vocab; j++) if (logits[o + j] > max) max = logits[o + j];
            double sum = 0;
            for (int j = 0; j < vocab; j++) { probsOut[o + j] = Math.exp(logits[o + j] - max); sum += probsOut[o + j]; }
            for (int j = 0; j < vocab; j++) probsOut[o + j] /= sum;
            loss -= Math.log(probsOut[o + targets[t]] + 1e-12);
        }
        return loss / T;
    }

    /** Backward through the whole net. dLogits = (probs - onehot)/T conventionally precomputed by caller into probs. */
    public void backward(Act act, double[] dLogits) {
        int[] tokens = act.tokens;
        int T = tokens.length, d = c.dModel, H = c.nHead, dh = c.headDim();
        double[] dxf = new double[T * d];
        // tied head backward
        for (int t = 0; t < T; t++)
            for (int i = 0; i < d; i++) {
                double s = 0;
                for (int j = 0; j < c.vocab; j++) s += dLogits[t * c.vocab + j] * tokEmb.w[j * d + i];
                dxf[t * d + i] = s;
            }
        for (int j = 0; j < c.vocab; j++)
            for (int i = 0; i < d; i++) {
                double s = 0;
                for (int t = 0; t < T; t++) s += dLogits[t * c.vocab + j] * act.xf[t * d + i];
                tokEmb.g[j * d + i] += s;
            }

        // final LN backward needs pre-LN stream; recompute is awkward, so we
        // cache it implicitly: act.x plus residuals. Recompute the stream.
        double[][] streams = new double[c.nLayer + 1][];
        streams[0] = act.x;
        for (int l = 0; l < c.nLayer; l++) {
            double[] prev = streams[l];
            double[] cur = new double[T * d];
            for (int i = 0; i < T * d; i++) cur[i] = prev[i] + act.projOut[l][i] + act.fc2o[l][i];
            streams[l + 1] = cur;
        }

        double[] dcur = new double[T * d];
        layernormBackward(streams[c.nLayer], lnF, dxf, dcur, act.lnCacheF[0], T, d);

        for (int l = c.nLayer - 1; l >= 0; l--) {
            P[] b = blocks[l];
            // MLP residual: cur = res1 + fc2o; res1 = prev + projOut
            double[] dfc2o = dcur.clone();
            double[] dres1 = dcur.clone();

            double[] dg = new double[T * 4 * d];
            linearBackward(act.g[l], b[FC2W].w, dfc2o, dg, b[FC2W], b[FC2B], T, 4 * d, d);
            double[] dfc1o = new double[T * 4 * d];
            for (int i = 0; i < T * 4 * d; i++) dfc1o[i] = dg[i] * geluGrad(act.fc1o[l][i]);
            double[] dln2o = new double[T * d];
            linearBackward(act.ln2o[l], b[FC1W].w, dfc1o, dln2o, b[FC1W], b[FC1B], T, d, 4 * d);
            // LN2 input is res1 = prev + projOut.
            layernormBackward(addStreams(streams[l], act.projOut[l]), b[LN2], dln2o, dres1, act.lnCache2[l], T, d);

            // attention residual: res1 = prev + projOut
            double[] dprojOut = dres1.clone();
            double[] dprev = dres1.clone();
            double[] dattnAll = new double[T * d];
            linearBackward(act.projIn[l], b[PROJW].w, dprojOut, dattnAll, b[PROJW], b[PROJB], T, d, d);

            double[] dqkv = new double[T * 3 * d];
            for (int h = H - 1; h >= 0; h--) {
                double[] q = act.q[l][h], k = act.k[l][h], v = act.v[l][h], probs = act.probs[l][h];
                double[] dctx = new double[T * dh];
                for (int t = 0; t < T; t++)
                    for (int i = 0; i < dh; i++) dctx[t * dh + i] = dattnAll[t * d + h * dh + i];
                double[] dprobs = new double[T * T], dv = new double[T * dh];
                for (int t = 0; t < T; t++)
                    for (int u = 0; u <= t; u++) {
                        double s = 0;
                        for (int i = 0; i < dh; i++) s += dctx[t * dh + i] * v[u * dh + i];
                        dprobs[t * T + u] = s;
                    }
                for (int u = 0; u < T; u++)
                    for (int i = 0; i < dh; i++) {
                        double s = 0;
                        for (int t = u; t < T; t++) s += probs[t * T + u] * dctx[t * dh + i];
                        dv[u * dh + i] = s;
                    }
                // softmax backward per row (only s<=t entries are live)
                double[] dscores = new double[T * T];
                for (int t = 0; t < T; t++) {
                    double dot = 0;
                    for (int s = 0; s <= t; s++) dot += dprobs[t * T + s] * probs[t * T + s];
                    for (int s = 0; s <= t; s++)
                        dscores[t * T + s] = probs[t * T + s] * (dprobs[t * T + s] - dot);
                }
                double scale = 1.0 / Math.sqrt(dh);
                double[] dq = new double[T * dh], dk = new double[T * dh];
                for (int t = 0; t < T; t++)
                    for (int i = 0; i < dh; i++) {
                        double s = 0;
                        for (int u = 0; u <= t; u++) s += dscores[t * T + u] * scale * k[u * dh + i];
                        dq[t * dh + i] = s;
                    }
                for (int u = 0; u < T; u++)
                    for (int i = 0; i < dh; i++) {
                        double s = 0;
                        for (int t = u; t < T; t++) s += dscores[t * T + u] * scale * q[t * dh + i];
                        dk[u * dh + i] = s;
                    }
                for (int t = 0; t < T; t++)
                    for (int i = 0; i < dh; i++) {
                        dqkv[t * 3 * d + h * dh + i] = dq[t * dh + i];
                        dqkv[t * 3 * d + d + h * dh + i] = dk[t * dh + i];
                        dqkv[t * 3 * d + 2 * d + h * dh + i] = dv[t * dh + i];
                    }
            }
            double[] dln1o = new double[T * d];
            linearBackward(act.ln1o[l], b[QKVW].w, dqkv, dln1o, b[QKVW], b[QKVB], T, d, 3 * d);
            layernormBackward(streams[l], b[LN1], dln1o, dprev, act.lnCache1[l], T, d);
            dcur = dprev;
        }
        // embeddings backward
        for (int t = 0; t < T; t++) {
            for (int i = 0; i < d; i++) {
                tokEmb.g[tokens[t] * d + i] += dcur[t * d + i];
                posEmb.g[t * d + i] += dcur[t * d + i];
            }
        }
    }

    private static double[] addStreams(double[] a, double[] b) {
        double[] r = new double[a.length];
        for (int i = 0; i < a.length; i++) r[i] = a[i] + b[i];
        return r;
    }

    public void zeroGrad() { for (P p : all) java.util.Arrays.fill(p.g, 0); }

    public void adamStep(double lr, double beta1, double beta2, double eps, double weightDecay, int step) {
        double bc1 = 1 - Math.pow(beta1, step), bc2 = 1 - Math.pow(beta2, step);
        for (P p : all) {
            for (int i = 0; i < p.w.length; i++) {
                p.m[i] = beta1 * p.m[i] + (1 - beta1) * p.g[i];
                p.v[i] = beta2 * p.v[i] + (1 - beta2) * p.g[i] * p.g[i];
                double mh = p.m[i] / bc1, vh = p.v[i] / bc2;
                p.w[i] -= lr * (mh / (Math.sqrt(vh) + eps) + weightDecay * p.w[i]);
            }
        }
    }

    public void clipGrad(double maxNorm) {
        double norm = 0;
        for (P p : all) for (double g : p.g) norm += g * g;
        norm = Math.sqrt(norm);
        if (norm > maxNorm) {
            double s = maxNorm / norm;
            for (P p : all) for (int i = 0; i < p.g.length; i++) p.g[i] *= s;
        }
    }
}
