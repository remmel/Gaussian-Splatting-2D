import * as tf from 'tensorflow';

export default class GS2D {
    constructor(imgSize = [256, 256, 3], numSamples = 100) {
        this.imgSize = imgSize;
        this.numSamples = numSamples;

        const [h, w] = this.imgSize;
        this.x = tf.linspace(0, w - 1, w).expandDims(0).tile([h, 1]);
        this.y = tf.linspace(0, h - 1, h).expandDims(1).tile([1, w]);
    }

    drawGaussian(sigma, rho, mean, color, alpha) {
        return tf.tidy(() => {
            const [h, w] = this.imgSize;
            const r = rho.reshape([-1, 1, 1]);
            const sx = sigma.slice([0, 0], [-1, 1]).reshape([-1, 1, 1]);
            const sy = sigma.slice([0, 1], [-1, 1]).reshape([-1, 1, 1]);
            const dx = tf.sub(this.x, mean.slice([0, 0], [-1, 1]).reshape([-1, 1, 1]));
            const dy = tf.sub(this.y, mean.slice([0, 1], [-1, 1]).reshape([-1, 1, 1]));

            const v = tf.exp(tf.div(
                tf.mul(-0.5, tf.sub(
                    tf.add(tf.pow(tf.mul(sx, dx), 2), tf.pow(tf.mul(sy, dy), 2)),
                    tf.mul(2, tf.mul(tf.mul(dx, dy), tf.mul(r, tf.mul(sy, sx))))
                )),
                tf.add(tf.mul(tf.pow(sy, 2), tf.mul(tf.pow(sx, 2), tf.sub(1, tf.pow(r, 2)))), 1e-8)
            ));

            const img = tf.sum(tf.mul(tf.mul(v.expandDims(3), color.reshape([-1, 1, 1, 3])), alpha.reshape([-1, 1, 1, 1])), 0);
            return tf.clipByValue(img, 0, 1);
        });
    }

    randomInitParam() {
        return tf.tidy(() => {
            const sigma = tf.sub(tf.randomUniform([this.numSamples, 2]), 3);
            const rho = tf.mul(tf.randomUniform([this.numSamples, 1]), 2);
            const mean = tf.atanh(tf.sub(tf.mul(tf.randomUniform([this.numSamples, 2]), 2), 1));
            const color = tf.atanh(tf.randomUniform([this.numSamples, 3]));
            const alpha = tf.sub(tf.zeros([this.numSamples, 1]), 0.01);
            return tf.concat([sigma, rho, mean, color, alpha], 1);
        });
    }

    parseParam(w) {
        return tf.tidy(() => {
            const size = tf.tensor(this.imgSize.slice(0, 2).reverse());
            const sigma = tf.mul(tf.sigmoid(w.slice([0, 0], [-1, 2])), tf.mul(size, 0.25));
            const rho = tf.tanh(w.slice([0, 2], [-1, 1]));
            const mean = tf.mul(tf.add(tf.mul(0.5, tf.tanh(w.slice([0, 3], [-1, 2]))), 0.5), size);
            const color = tf.add(tf.mul(0.5, tf.tanh(w.slice([0, 5], [-1, 3]))), 0.5);
            const alpha = tf.add(tf.mul(0.5, tf.tanh(w.slice([0, 8], [-1, 1]))), 0.5);
            return [sigma, rho, mean, color, alpha];
        });
    }

    async train(target, numEpochs = 10, lr = 0.005) {
        console.time('epoch');
        const w = tf.variable(this.randomInitParam());
        const optimizer = tf.train.adam(lr);

        for (let epoch = 0; epoch < numEpochs; epoch++) {
            console.time(`Epoch ${epoch}`);
            for (let i = 0; i < 30; i++) {
                optimizer.minimize(() => {
                    const predicted = this.drawGaussian(...this.parseParam(w));
                    return tf.losses.meanSquaredError(target, predicted);
                });
            }

            // Display progress
            const predicted = this.drawGaussian(...this.parseParam(w));
            await this.displayProgress(predicted);
            console.timeEnd(`Epoch ${epoch}`);
        }
    }

    async displayProgress(predicted, predictedCanvas = null) {
        const canvas = predictedCanvas ?? document.getElementById('sourceCanvas');
        await tf.browser.toPixels(predicted, canvas);
    }
}
