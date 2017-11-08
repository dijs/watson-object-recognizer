const convnetjs = require("convnetjs");
const jimp = require('jimp');
const fs = require('fs');
const join = require('path').join;
const maxBy = require('lodash/maxBy');

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

console.log('Possible Labels', labels);

const networkData = JSON.parse(fs.readFileSync('./networks/1510143349594.json', 'utf8'));
const net = new convnetjs.Net();
net.fromJSON(networkData);

const filename = process.argv[2];
console.log('Trying to recognize', filename);

(async function () {
  const vol = await imageToVol(join(pathToTrainingData, filename));
  const results = net.forward(vol).w;
  const bestGuess = maxBy(labels, (label, index) => results[index]);
  const bestGuessScore = Math.max(...results);
  console.log('Guessing this is', bestGuess);
  console.log('With propability', Math.round(bestGuessScore * 100) + '%');
})();