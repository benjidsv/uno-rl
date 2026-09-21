k2 approximate KL is better here because it is computed in log-space and remains
nonnegative. Unlike the previous ratio-space quadratic, it does not cap the
diagnostic when the new selected-action probability approaches zero. The k3 one led to policy collapse.

T/G scaling, while a sound idea in principle (scale to prevent the model to learn to stall games), is counter productive: it inflates the gradients, which directly contributes to the policy collapse described above.

desktop compute budget
after empirical measurements and many optimizations:
the best config is 32 minibatches (roughly 512) at 16 workers, achieves 17.4k presentations/s on cuda
with amp: 