# NumCompute-Stream: Technical Report

**Student:** Khush Bhavin Maru
**Course:** Programming for Artificial Intelligence (COMP 5004)
**University:** Adelaide University


## Design Decisions

The first real decision was what "streaming support" actually means for a
decision tree, because there is no single obvious answer. A Hoeffding tree
grows nodes incrementally using statistical bounds to decide when a split
is reliable enough to commit to. That is the theoretically correct approach
for unbounded streams, but implementing it properly requires getting the
Hoeffding bound calculations right, handling the grace period and split
confidence parameters, and writing tests that can actually verify the
statistical behaviour. For a single-semester assignment I decided the
complexity was not worth it. Instead the tree appends each incoming chunk
to an internal buffer and rebuilds from scratch on every partial_fit call.
The result is correct, easy to test, and honest about its tradeoff: training
cost grows as the buffer grows. The benchmark results quantify exactly how
fast that happens so it is not a hidden cost.

For statistics the choice was more straightforward. Welford's algorithm
gives exact running mean and variance in O(1) per sample with no stored
history, and it stays numerically stable on large inputs where the
two-pass formula drifts. The MultiFeatureStats class wraps one RunningStats
tracker per column so 2D arrays work cleanly without looping over columns
in the calling code.

Preprocessing was the module where the streaming constraint created the
most friction. StandardScaler was simple because Welford handles it
directly. MinMaxScaler was slightly trickier since you have to expand the
stored range whenever a new chunk goes outside what has been seen before,
and the transform has to reflect that expanded range immediately. The
Imputer only supports mean and constant strategies in streaming mode
because there is no exact O(1) online median without storing all values,
and approximate methods like the P-squared algorithm felt out of scope.

The ensemble design went with Random Forest and Bagging. Both use bootstrap
sampling per tree, and Random Forest adds feature subsampling at each split
on top of that. The EnsembleClassifier wrapper gives them a shared interface
so the rest of the code never needs to know which one it is talking to.
Switching methods is a single constructor argument.

The Pipeline and StreamTrainer classes exist to keep the demo notebook clean.
Without them the notebook would repeat the same fit-transform-predict loop
in every cell. StreamTrainer also handles timing per chunk, which feeds
directly into the benchmark plots without any extra instrumentation.


## Testing and Edge Cases

Writing the tests before finishing the main modules turned out to be
genuinely useful rather than just a process requirement. Several bugs
showed up early that would have been annoying to track down later.

The confusion matrix tracker had a bug where it expanded incorrectly when
new classes appeared mid-stream. The old indices were being mapped back
into the new larger matrix using a range slice instead of searchsorted,
which put counts in the wrong cells when classes were not contiguous
integers starting from zero. The test caught this immediately because it
checked specific cell values rather than just the matrix shape.

NaN handling needed testing at the boundary between the stats and
preprocessing layers. The Welford updater strips NaNs before processing,
which is correct, but early versions of the StandardScaler were passing
the raw column including NaNs to the tracker before stripping them. The
test for NaN inputs in partial_fit caught this.

Zero-variance features are a genuine edge case that shows up in real data
when a column is constant across a chunk. The scaler sets std to 1.0 for
those columns so the transform does not produce NaN or inf. The tree skips
zero-variance features entirely when searching for splits since there is
nothing to split on. Both behaviours get tested explicitly.

Single-sample chunks are a streaming-specific edge case that would never
come up in batch training. The Welford updater handles them fine since it
processes values one at a time anyway, but the tree needed a specific check
because building a split requires at least two samples. The test confirms
that calling partial_fit with a single row does not crash and that the
tree accumulates correctly across multiple single-sample calls.

The rolling window accuracy test checks that the window size is actually
respected, not just that the metric returns a number. With window=2 and
three chunks where accuracies are 1.0, 0.0, and 1.0, the result after the
third chunk should be 0.5 because the first chunk has dropped out. Getting
that specific value right requires the buffer logic to be correct.

In total there are 177 tests across five files. All pass.


## Benchmark Results

The imputation benchmark compares filling NaN values with a Python loop
against the vectorised approach using np.isnan and np.take. At n=500 the
difference is small, but by n=10000 the vectorised version is around 5 to
8 times faster. The Python loop overhead comes from iterating over columns
and then iterating again over the NaN positions within each column. The
vectorised version finds all NaN positions across the entire array in one
call and scatters the fill values in one operation.

The confusion matrix benchmark shows a similar pattern. At n=50000
np.add.at is around 15 to 20 times faster than the row-by-row loop. The
loop pays Python function call overhead for every single prediction, which
adds up fast at that scale.

The model comparison benchmark runs a single decision tree, Random Forest
with 10 trees, and Bagging with 10 trees across 10 chunks of 30 samples
each on an 8-feature binary classification task. The single tree is fastest
per chunk as expected. Random Forest takes roughly 8 to 12 times longer
per chunk but achieves higher final accuracy. Bagging sits between the two
since it uses all features at every split rather than subsampling.

The scaler benchmark confirms that StandardScaler partial_fit cost stays
flat across chunk calls, which is the expected O(n_features) behaviour.
The tree does not have this property since it rebuilds from an ever-growing
buffer, and the streaming model benchmark reflects that cost visibly in the
per-chunk times for later chunks.


## Reflections

The hardest part of this assignment was not the algorithms themselves but
getting the streaming interfaces to actually be correct rather than just
syntactically present. The first version of StandardScaler had a
partial_fit method that computed mean and std from the current chunk only
and overwrote the previous values. It passed basic smoke tests but gave
completely wrong results after the second chunk because the transform was
using statistics from chunk 2 alone applied to data that had been scaled
using chunk 1 statistics. Switching to Welford trackers fixed the
correctness problem and made the tests much easier to reason about because
there was one clear source of truth for what the statistics should be at
any point.

The decision to rebuild the tree from the full buffer was something I went
back and forth on. It feels like cheating because a real streaming system
would not store all historical data, but the alternative was implementing
Hoeffding trees which would have taken most of the remaining time and
produced something harder to verify. The benchmark at least makes the
cost visible rather than hiding it, which felt like the right call.

Testing at 177 tests ended up being more useful than the minimum 30 would
have been. Finding the confusion matrix bug through tests rather than by
staring at wrong output in the notebook saved a real amount of time. The
rolling window test in particular was worth writing because it checked
behaviour that would be easy to get subtly wrong and hard to notice just
by looking at plots.

The visualise module was the most straightforward part of the assignment
and also the part that made everything else look better. Having dedicated
functions for each plot type kept the notebook cells short and made the
comparison between models much cleaner than it would have been with raw
matplotlib in every cell.
