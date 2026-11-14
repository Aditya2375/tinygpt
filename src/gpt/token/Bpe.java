package gpt.token;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Byte-level BPE trained from scratch. Vocabulary: ids 0..255 are raw bytes;
 * ids 256+ are learned merges. Encoding applies merges in rank order.
 */
public final class Bpe {
    public final int vocabSize;
    /** merges[i] = {left, right, newId} in training (rank) order. */
    public final int[][] merges;
    private final Map<Long, Integer> pairToNew;
    private final String[] idToBytes;

    private Bpe(int vocabSize, int[][] merges) {
        this.vocabSize = vocabSize;
        this.merges = merges;
        this.pairToNew = new HashMap<>();
        this.idToBytes = new String[vocabSize];
        for (int i = 0; i < 256; i++) idToBytes[i] = new String(new byte[]{(byte) i}, java.nio.charset.StandardCharsets.ISO_8859_1);
        for (int[] m : merges) {
            pairToNew.put(((long) m[0] << 32) | (m[1] & 0xffffffffL), m[2]);
            idToBytes[m[2]] = idToBytes[m[0]] + idToBytes[m[1]];
        }
    }

    /** Rebuild a tokenizer from saved merges. */
    public static Bpe fromMerges(int[][] merges) {
        return new Bpe(256 + merges.length, merges);
    }

    /** Train on UTF-8 text; returns a tokenizer with 256 + nMerges symbols. */
    public static Bpe train(String text, int nMerges) {
        List<int[]> seqs = new ArrayList<>();
        // One sequence per line keeps merges from crossing sentence boundaries.
        for (String line : text.split("\n")) {
            byte[] b = line.getBytes(java.nio.charset.StandardCharsets.UTF_8);
            int[] s = new int[b.length];
            for (int i = 0; i < b.length; i++) s[i] = b[i] & 0xff;
            if (s.length > 1) seqs.add(s);
        }
        int[][] merges = new int[nMerges][];
        int nextId = 256;
        for (int step = 0; step < nMerges; step++) {
            Map<Long, long[]> counts = new HashMap<>(); // pair -> {count}
            Map<Long, Long> freq = new HashMap<>();
            for (int[] s : seqs) {
                for (int i = 0; i + 1 < s.length; i++) {
                    long k = ((long) s[i] << 32) | (s[i + 1] & 0xffffffffL);
                    freq.merge(k, 1L, Long::sum);
                }
            }
            long bestKey = -1; long bestN = 1;
            for (Map.Entry<Long, Long> e : freq.entrySet()) {
                if (e.getValue() > bestN) { bestN = e.getValue(); bestKey = e.getKey(); }
            }
            if (bestKey < 0) break; // nothing frequent enough left
            int a = (int) (bestKey >> 32), b = (int) (long) bestKey;
            merges[step] = new int[]{a, b, nextId};
            for (int i = 0; i < seqs.size(); i++) {
                seqs.set(i, mergeOnce(seqs.get(i), a, b, nextId));
            }
            nextId++;
            counts.clear();
        }
        int done = nextId - 256;
        int[][] trimmed = new int[done][];
        System.arraycopy(merges, 0, trimmed, 0, done);
        return new Bpe(256 + done, trimmed);
    }

    private static int[] mergeOnce(int[] s, int a, int b, int id) {
        List<Integer> out = new ArrayList<>(s.length);
        for (int i = 0; i < s.length; i++) {
            if (i + 1 < s.length && s[i] == a && s[i + 1] == b) { out.add(id); i++; }
            else out.add(s[i]);
        }
        return out.stream().mapToInt(Integer::intValue).toArray();
    }

    public int[] encode(String text) {
        byte[] b = text.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        int n = b.length;
        if (n == 0) return new int[0];
        int[] val = new int[n];
        int[] prev = new int[n], next = new int[n];
        boolean[] dead = new boolean[n];
        for (int i = 0; i < n; i++) { val[i] = b[i] & 0xff; prev[i] = i - 1; next[i] = i + 1; }
        next[n - 1] = -1;
        // Priority queue of (rank, pos), lazy deletion.
        java.util.PriorityQueue<long[]> pq = new java.util.PriorityQueue<>(
                (x, y) -> x[0] != y[0] ? Long.compare(x[0], y[0]) : Long.compare(x[1], y[1]));
        for (int i = 0; i + 1 < n; i++) {
            Integer id = pairToNew.get(((long) val[i] << 32) | (val[i + 1] & 0xffffffffL));
            if (id != null) pq.add(new long[]{id, i});
        }
        while (!pq.isEmpty()) {
            long[] top = pq.poll();
            int rank = (int) top[0], pos = (int) top[1];
            if (dead[pos]) continue;
            int nxt = next[pos];
            if (nxt < 0) continue;
            Integer id = pairToNew.get(((long) val[pos] << 32) | (val[nxt] & 0xffffffffL));
            if (id == null || id != rank) continue; // stale entry
            val[pos] = id;
            dead[nxt] = true;
            int after = next[nxt];
            next[pos] = after;
            if (after >= 0) prev[after] = pos;
            int p = prev[pos];
            if (p >= 0 && !dead[p]) {
                Integer lid = pairToNew.get(((long) val[p] << 32) | (val[pos] & 0xffffffffL));
                if (lid != null) pq.add(new long[]{lid, p});
            }
            if (after >= 0) {
                Integer rid = pairToNew.get(((long) val[pos] << 32) | (val[after] & 0xffffffffL));
                if (rid != null) pq.add(new long[]{rid, pos});
            }
        }
        java.util.List<Integer> out = new java.util.ArrayList<>();
        for (int i = 0; i < n; i++) if (!dead[i]) out.add(val[i]);
        return out.stream().mapToInt(Integer::intValue).toArray();
    }

    public String decode(int[] ids) {
        StringBuilder sb = new StringBuilder();
        for (int id : ids) sb.append(idToBytes[id]);
        return new String(sb.toString().getBytes(java.nio.charset.StandardCharsets.ISO_8859_1), java.nio.charset.StandardCharsets.UTF_8);
    }
}
