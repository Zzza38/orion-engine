/**
 * Iris: classify 150 flowers into 3 species from 4 measurements.
 *
 * Demonstrates multi-class classification end to end: a train/test split, feature
 * standardization (fitted on the training set only), a softmax output trained with sparse
 * categorical cross-entropy on integer labels, a validation split with early stopping, and
 * evaluation on held-out data with a confusion matrix.
 *
 *   npx tsx examples/iris.ts
 *
 * Expected output (seeded): progress lines every 25 epochs, then "Early stopping after epoch 55;
 * restored the weights of epoch 35", a test accuracy of 93.3% (28/30), a confusion matrix whose
 * only errors are versicolor/virginica mix-ups (the two species overlap; setosa is always
 * right), and the class probabilities of three test flowers.
 */
import { argmax, dense, earlyStopping, Sequential, StandardScaler, trainTestSplit } from "../src/index.js";
import { IRIS, IRIS_SPECIES } from "./data/iris.js";

// Features are the first four columns; the label is the species index (0, 1 or 2).
const features = IRIS.map((row) => row.slice(0, 4));
const labels = IRIS.map((row) => row[4]);

// Hold out 20% for the final test. The split shuffles with its own seed; the dataset is sorted by
// species, so shuffling matters.
const { xTrain, xTest, yTrain, yTest } = trainTestSplit(features, labels, { testSize: 0.2, seed: 39 });

// Standardize each feature to mean 0 / std 1. Fit on the training data only, then apply the same
// transform to the test data, so no information leaks from the test set.
const scaler = new StandardScaler();
const xTrainScaled = scaler.fitTransform(xTrain);
const xTestScaled = scaler.transform(xTest);

// 4 inputs → 16 ReLU units → 3 softmax probabilities.
const model = new Sequential({
    inputSize: 4,
    seed: 1,
    name: "iris",
    layers: [dense(16, "relu"), dense(3, "softmax")],
});
// "scce" (sparse categorical cross-entropy) takes integer labels directly; with one-hot targets
// you would use "cce" instead.
model.compile({ loss: "scce", optimizer: { name: "adam", learningRate: 0.01 }, metrics: ["accuracy"] });

// Hold out the last 20% of the (already shuffled) training data for validation, and stop once the
// validation loss has not improved for 20 epochs, rolling back to the best weights.
const stopper = earlyStopping({ monitor: "valLoss", patience: 20, restoreBestWeights: true });
const history = model.fit(xTrainScaled, yTrain, {
    epochs: 500,
    batchSize: 16,
    validationSplit: 0.2,
    callbacks: [stopper],
    verbose: 25,
});

const best = history.best("valLoss");
if (stopper.stoppedEpoch !== null && best !== undefined) {
    console.log(
        `\nEarly stopping after epoch ${stopper.stoppedEpoch + 1}; restored the weights of epoch ${best.epoch + 1} ` +
            `(valLoss ${best.value.toFixed(4)}).`,
    );
}

// Evaluate on data the model has never seen.
const { loss, accuracy } = model.evaluate(xTestScaled, yTest);
const predicted = argmax(model.predict(xTestScaled));
const correct = predicted.filter((p, i) => p === yTest[i]).length;
console.log(
    `\nTest accuracy: ${(accuracy * 100).toFixed(1)}% (${correct}/${yTest.length}), test loss ${loss.toFixed(4)}`,
);

// Confusion matrix: rows are the true species, columns the predicted one.
const confusion = IRIS_SPECIES.map(() => IRIS_SPECIES.map(() => 0));
for (const [i, p] of predicted.entries()) confusion[yTest[i]][p]++;
console.log(`\n${"true \\ predicted".padEnd(18)}${IRIS_SPECIES.map((s) => s.padStart(12)).join("")}`);
for (const [i, row] of confusion.entries()) {
    console.log(`${IRIS_SPECIES[i].padEnd(18)}${row.map((n) => String(n).padStart(12)).join("")}`);
}

// Class probabilities for a few test flowers (predict() on one sample returns number[]).
console.log();
for (let i = 0; i < 3; i++) {
    const probabilities = model.predict(xTestScaled[i]);
    const shown = probabilities.map((p, c) => `${IRIS_SPECIES[c]} ${(p * 100).toFixed(1)}%`).join(", ");
    const measurements = xTest[i].map((v) => v.toFixed(1)).join(", ");
    console.log(`[${measurements}] (${IRIS_SPECIES[yTest[i]]}): ${shown}`);
}
