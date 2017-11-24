const jimp = require('jimp');
const basename = require('path').basename;
const join = require('path').join;
const fs = require('fs');

const pathToTagged = '../watson-sight/tagged';
const pathToTrainingData = './training-data';
const size = 128;

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
  return new Promise((resolve, reject) => {
    jimp.read(path).then(image => {
      return image
        .crop(x, y, w, h)
        .contain(size, size, (err, newImage) => {
          if (err) return reject(err);
          resolve(newImage);
        });
    }).catch(err => reject(err));
  });
}

function writeTrainingImage(filename) {
  const { tag, age, x, y, w, h } = getImageInfo(filename);
  return jimp.read(join(pathToTagged, filename)).then(image => {
    return image
      .crop(x, y, w, h)
      .contain(size, size)
      .write(join(pathToTrainingData, `${tag}_${age}.jpg`))
      .flip(true, false)
      .write(join(pathToTrainingData, `${tag}_${age}_flipped.jpg`));
  });
}

module.exports = getTrainingImage;

if (require.main === module) {
  const taggedFiles = fs
    .readdirSync('../watson-sight/tagged')
    .filter(name => name.indexOf('.') !== 0);
  (async function () {
    console.log('Getting ready to train', taggedFiles.length, 'images');
    let timeSum = 0;
    for (let index = 0; index < taggedFiles.length; index++) {
      const started = Date.now();
      const filename = taggedFiles[index];
      try {
        await writeTrainingImage(filename);
      } catch (e) {
        console.log(filename, e.message);
        continue;
      }
      const took = Date.now() - started;
      timeSum += took;
      const num = index + 1;
      const averageTrainingTime = timeSum / num / 1000;
      const left = (taggedFiles.length - num) * averageTrainingTime;
      const data = [
        'Train Time', averageTrainingTime.toFixed(2),
        'Est Left', left.toFixed(2)
      ].map(s => (s + '').padStart(10));
      console.log(...data);
    }
  })();

}
