const jimp = require('jimp');
const basename = require('path').basename;
const join = require('path').join;
const fs = require('fs');

const pathToTagged = '../watson-sight/tagged';
const pathToTrainingData = './training-data';
const size = 64;

function getImageInfo(filename) {
  const [tag, timestamp, left, right, top, bottom] = basename(filename, '.png').split('_');
  const x = parseInt(left, 10);
  const y = parseInt(top, 10);
  return {
    tag,
    age: Date.now() - parseInt(timestamp, 0),
    x,
    y,
    w: parseInt(right, 10) - x,
    h: parseInt(bottom, 10) - y,
  };
}

function getTrainingImage(path) {
  const { x, y, w, h } = getImageInfo(basename(path));
  return new Promise(resolve => {
    jimp.read(path).then(image => {
      return image
        .crop(x, y, w, h)
        .contain(size, size, (err, newImage) => {
          resolve(newImage);
        });
    });
  });
}

function writeTrainingImage(filename) {
  const { tag, age, x, y, w, h } = getImageInfo(filename);
  return jimp.read(join(pathToTagged, filename)).then(image => {
    return image
      .crop(x, y, w, h)
      .contain(size, size)
      .write(join(pathToTrainingData, `${tag}_${age}.jpg`));
  });
}

module.exports = getTrainingImage;

if (require.main === module) {
  const taggedFiles = fs.readdirSync('../watson-sight/tagged');
  Promise.all(taggedFiles.map(writeTrainingImage)).then(() => console.log('done'));
}
