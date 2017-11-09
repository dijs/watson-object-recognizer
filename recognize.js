const convnetjs = require("convnetjs");
const jimp = require('jimp');
const fs = require('fs');
const join = require('path').join;
const maxBy = require('lodash/maxBy');
const getTrainingImage = require('./normalizer');

function imageToVol(image) {
  var p = image.bitmap.data;
  var pv = []
  for(var i=0;i<p.length;i++) {
    pv.push(p[i]/255.0-0.5); // normalize image pixels to [-0.5, 0.5]
  }
  var x = new convnetjs.Vol(image.bitmap.width, image.bitmap.height, 4, 0.0); //input volume (image)
  x.w = pv;
  return x;
}

const pathToTrainingData = join(__dirname, 'training-data');
const images = fs
  .readdirSync(pathToTrainingData)
  .filter(name => name.indexOf('.') !== 0);

const getLabel = filename => filename.split('_')[0];

const byFindingLabels = (memo, filename) => Object.assign(memo, { [getLabel(filename)]: true });
const labels = Object.keys(images.reduce(byFindingLabels, {}));

console.log('Possible Labels', labels);

const networkData = JSON.parse(fs.readFileSync(join(__dirname, 'networks/1510154893606.json'), 'utf8'));
const net = new convnetjs.Net();
net.fromJSON(networkData);

// Full captured image
function recognize(path) {
  return getTrainingImage(path)
    .then(imageToVol)
    .then(vol => {
      const results = net.forward(vol).w;
      const score = Math.max(...results);
      const guess = labels[results.indexOf(score)];
      return {
        guess,
        score
      };
    });
};

// const path = process.argv[2];
// console.log('Trying to recognize', path);
// recognize(path).then(console.log).catch(console.error);

module.exports = recognize;
