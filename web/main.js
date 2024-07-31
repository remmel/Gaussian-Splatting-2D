import * as tf from 'tensorflow';
import GS2D from "./GS2D.js";
import { RangeInputElt } from "./RangeInputElt.js";
import { hexToRGB } from "./utils.js";

async function initTrain() {

    // await tf.setBackend('cpu');
    // await tf.setBackend('webgl');
    await tf.ready();

    console.log(`Using TensorFlow.js backend: ${tf.getBackend()}`);

    // load target image
    const targetCanvas = document.getElementById('targetCanvas');
    const ctx = targetCanvas.getContext('2d');
    const img = new Image();
    img.src = 'a.png'; // Replace with your image path
    await new Promise(resolve => img.onload = resolve);
    ctx.drawImage(img, 0, 0, 256, 256);

    async function startTrain() {
        this.disabled = true;
        const targetImageData = ctx.getImageData(0, 0, 256, 256);
        const targetTensor = tf.browser.fromPixels(targetImageData).div(255);
        const gs = new GS2D();
        await gs.train(targetTensor);
    }

    document.getElementById('startButton').addEventListener('click', startTrain)
}



async function initDemo() {
    RangeInputElt.defineCustomElt()
    const gs = new GS2D()
    const demoParams = [
        // sx,  sy, rho, m_x, m_y, c_r, c_g, c_b,   a
        // [-3, -3, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],  // Red circle
        [-1, -3, 0.0, -1, -1, 0.0, 1.0, 0.0, 1.0],  // Green ellipse
        [-3, -3, 0.75, 1, 1, 0.0, 0.0, 1.0, 1.0],  // Blue tilted ellipse
        // [-3, -3, 0.75, 1, 1, 0.0, 0.0, 1.0, 1.0],  // Blue tilted ellipse
    ]

    function readSplatParams() {
        const sx = document.getElementById('sx').value
        const sy = document.getElementById('sy').value
        const rho = document.getElementById('rho').value
        const mx = document.getElementById('mx').value
        const my = document.getElementById('my').value
        const color = hexToRGB(document.getElementById('color').value)
        const alpha = document.getElementById('alpha').value

        return [sx, sy, rho, mx, my, color.r, color.g, color.b, alpha]
    }


    // will draw the hardcoded splats and the one from input form
    async function drawDemoSplats() {
        const customParams = readSplatParams()
        const w = tf.tensor([...demoParams, customParams]);
        const [sigma, rho, mean, color, alpha] = gs.parseParam(w);
        const img = gs.drawGaussian(sigma, rho, mean, color, alpha);
        await gs.displayProgress(img, document.getElementById('demoCanvas'));
    }

    document.querySelectorAll('input, range-input').forEach(input => {
        input.addEventListener('input', drawDemoSplats);
    })

    drawDemoSplats()
}

document.addEventListener('DOMContentLoaded', () => {
    initTrain()
    initDemo()
})

