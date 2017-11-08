var convnetjs = require("convnetjs");
var jimp = require('jimp');

// If this works even slightly well, break everything out into modules and organize

const labels = [
  'thatch',
  'langos',
  'vw'
];

// a small Convolutional Neural Network if you wish to predict on images:
var layer_defs = [];
layer_defs.push({type:'input', out_sx:32, out_sy:32, out_depth:3}); // declare size of input
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
layer_defs.push({type:'softmax', num_classes:labels.length});
// output Vol is of size 1x1x10 here

net = new convnetjs.Net();
net.makeLayers(layer_defs);

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

(async function () {
  // const x = await imageToVol('http://192.168.1.103:9101/untagged/dog-1509431268384-83212171041426510.jpg');
  const x1 = await imageToVol('./output/left.jpg');
  const x2 = await imageToVol('./output/right.jpg');
  const x3 = await imageToVol('./output/vw_left.jpg');
  const x4 = await imageToVol('./output/vw_right.jpg');

  // forward a random data point through the network
  // prob is a Vol. Vols have a field .w that stores the raw data, and .dw that stores gradients
  console.log('probability that x1 is class 0: ' + net.forward(x1).w[0]); // prints 0.50101

  var trainer = new convnetjs.SGDTrainer(net, {learning_rate:0.01, l2_decay:0.001});
  // train the network, specifying that x is class zero
  trainer.train(x1, 0);
  trainer.train(x2, 0);
  trainer.train(x3, 1);
  trainer.train(x4, 1);

  console.log('probability that x1 is class 0: ' + net.forward(x1).w[0]);
  console.log('probability that x2 is class 0: ' + net.forward(x2).w[0]);
  console.log('probability that x3 is class 1: ' + net.forward(x3).w[1]);
  console.log('probability that x4 is class 1: ' + net.forward(x4).w[1]);
  
})();