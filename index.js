const convnetjs = require("convnetjs");
const jimp = require('jimp');
const fs = require('fs');
const join = require('path').join;

// If this works even slightly well, break everything out into modules and organize

// a small Convolutional Neural Network if you wish to predict on images:
const createNetwork = labels => {
  const layer_defs = [];
  layer_defs.push({type:'input', out_sx:64, out_sy:64, out_depth:3}); // declare size of input
  // output Vol is of size 32x32x3 here
  layer_defs.push({type:'conv', sx:5, filters:16, stride:1, pad:2, activation:'relu'});
  // the layer will perform convolution with 16 kernels, each of size 5x5.
  // the input will be padded with 2 pixels on all sides to make the output Vol of the same size
  // output Vol will thus be 32x32x16 at this point
  layer_defs.push({type:'pool', sx:2, stride:2});
  // output Vol is of size 16x16x16 here
  layer_defs.push({type:'conv', sx:5, filters:20, stride:1, pad:2, activation:'relu'});
  // output Vol is of size 16x16x20 here
  layer_defs.push({type:'pool', sx:2, stride:2});
  // output Vol is of size 8x8x20 here
  layer_defs.push({type:'conv', sx:5, filters:20, stride:1, pad:2, activation:'relu'});
  // output Vol is of size 8x8x20 here
  layer_defs.push({type:'pool', sx:2, stride:2});
  // output Vol is of size 4x4x20 here
  layer_defs.push({type:'softmax', num_classes: labels.length});
  // output Vol is of size 1x1x10 here

  const net = new convnetjs.Net();
  net.makeLayers(layer_defs);

  return net;
}

async function imageToVol(path) {
  const image = await jimp.read(path);
  var p = image.bitmap.data;
  var pv = []
  for(var i=0;i<p.length;i++) {
    pv.push(p[i]/255.0-0.5); // normalize image pixels to [-0.5, 0.5]
  }
  var x = new convnetjs.Vol(image.bitmap.width, image.bitmap.height, 4, 0.0); //input volume (image)
  x.w = pv;
  return x;
}

const pathToTrainingData = './training-data';
const images = fs
  .readdirSync(pathToTrainingData)
  .filter(name => name.indexOf('.') !== 0);

const getLabel = filename => filename.split('_')[0];

const byFindingLabels = (memo, filename) => Object.assign(memo, { [getLabel(filename)]: true });
const labels = Object.keys(images.reduce(byFindingLabels, {}));

console.log('Labels', labels);

// Load network
const networkData = JSON.parse(fs.readFileSync('./networks/1510153941998.json', 'utf8'));
const net = new convnetjs.Net();
net.fromJSON(networkData);

// Create new
// const net = createNetwork(labels);

const trainer = new convnetjs.SGDTrainer(net, {
  method: 'adadelta',
  batch_size: 10,
  l2_decay: 0.001
});

(async function () {
  console.log('Getting ready to train', images.length, 'images');
  let timeSum = 0;
  let xLossSum = 0;
  let runs = 2000;

  for (let index = 0; index < runs; index++) {
    const randomFilename = images[Math.floor(Math.random() * images.length)];
    const label = getLabel(randomFilename);
    const labelIndex = labels.indexOf(label);
    const started = Date.now();
    const x = await imageToVol(join(pathToTrainingData, randomFilename));
    const stats = trainer.train(x, labelIndex);
    const lossX = stats.cost_loss;
    xLossSum += lossX;
    const lossW = stats.l2_decay_loss;
    const took = Date.now() - started;
    timeSum += took;
    const num = index + 1;
    const averageTrainingTime = timeSum / num / 1000;
    const averageLossX = xLossSum / num;
    const left = (runs - num) * averageTrainingTime;
    const data = [
      'Labels', label, labelIndex,
      'Loss', averageLossX.toFixed(3), lossW.toFixed(3),
      'Train Time', averageTrainingTime.toFixed(2),
      'Est Left', left.toFixed(2)
    ].map(s => (s + '').padStart(10));
    console.log(...data);
  }

  console.log('Saving network');
  fs.writeFileSync(`./networks/${Date.now()}.json`, JSON.stringify(net.toJSON()));
})();
