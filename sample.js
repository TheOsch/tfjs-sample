/*
AN EXAMPLE OF TFJS

(https://github.com/tensorflow/tfjs) 
reproduced from https://www.kaggle.com/robertbm/extreme-learning-machine-example

This is a quick example of a TJFS implementation/solution to the MNIST handwritten
digit digital recognizer problem.

Let's chose this dataset since a high accuracy on MNIST is regarded as a basic requirement 
of credibility in a classification algorithm.

I think this example could be especially interesting to novice machine learning students
mainly because of its simplicity.
*/

// Import the TFJS 
// This module is optimized for Node.js using CPU. If you've got a good GPU you can try
// '@tensorflow/tfjs-node-gpu' instead.
const tf = require('@tensorflow/tfjs-node');

// Some helper functions
const f = require('./functions');

/*
READING THE MNIST DATASET

The MNIST dataset contains a series of monochrome images 28x28 of handwritten digits,
on each row of the dataset stored as a vector with 784 values, each representing a
pixel value, the training data has an additional column containing the label associated
with each image.
*/

const path = require('path');
const fs = require('fs');
const parseCSV = require('csv-parse/lib/sync');

console.log('Reading MINST data');
const minstText = fs.readFileSync(path.resolve(__dirname, 'input/train.csv')).toString();
console.log('Parsing CSV (warning: takes a while)');
const minstArray = parseCSV(minstText);
const minstData = minstArray.slice(1);

f.displayMatrix(minstData);

/*
As we can see, each row has 785 columns, with the first being the label and the rest of them 
representing the pixel values (28x28) of the image.

Next, we will need to separate the labels from the pixel values. 
*/

const xTrain = minstData.map(line => line.slice(1));
console.log('Training images:');
f.displayMatrix(xTrain);

const labels = minstData.map(line => Number.parseInt(line[0]));
console.log('Correct labels:');
f.displayVector(labels);

/*
It's time to transfer our data to TF.js tensors
*/

const xTensor = tf.tensor2d(xTrain, [xTrain.length, xTrain[0].length], 'float32');

/*
Since this is a multiclass classification problem, we will One Hot Encode
the labels. This simply means that we will use vectors to represent each
class, instead of the label value. Each vector contains the value 1 at the
index corresponding to the class it represents, with the rest of the values
set to 0.
*/

const yTensor = tf.oneHot(labels, 10);

console.log('One hot encoded labels:')
f.displayMatrix(yTensor.arraySync());

/*
The next step is to split the data into training and testing parts, since we 
would like to test our accuracy of our model at the end. We will use around 10%
of our training data for testing.
*/

const trainSize = Math.floor(xTrain.length * 0.9);
console.log('Train size: ', trainSize);
const testSize = xTrain.length - trainSize;
console.log('Test size: ', testSize);

const xTrainTensor = xTensor.slice([0],  [trainSize]);
const yTrainTensor = yTensor.slice([0], [trainSize]);
const xTestTensor = xTensor.slice([trainSize], [-1]);
const yTestTensor = yTensor.slice([trainSize], [-1]);

/*
Now, our data is ready for both training and testing our neural network. 
Next, we will take a look at the implementation of the 
Extreme Learning Machine. 

EXTREME LEARNING MACHINE IMPLEMENTATION

The ELM algorithm is similar to other neural networks with 3 key differences:

1. The number of hidden units is usually larger than in other neural networks that are 
   trained using backpropagation.
2. The weights from input to hidden layer are randomly generated, usually using values
   from a continuous uniform distribution.
3. The output neurons are linear rather than sigmoidal, this means we can use least square
   errors regression to solve the output weights.

Let's start by defining some constants and generate the input to hidden layer weights:
*/


const INPUT_LENGTH = xTrain[0].length;
const HIDDEN_UNITS = 1000;
const wIn = tf.randomNormal([INPUT_LENGTH, HIDDEN_UNITS]);

/*
The next step is to compute our hidden layer to output weights. 
This is done in the following way:

- Compute the dot product between the input and input-to-hidden layer weights, and apply
  some activation function. Here we will use ReLU, since it is simple and in this case it
  gives us a good result:
*/

function inputToHidden(x) {
  let a = tf.dot(x, wIn);
  a = tf.maximum(a, tf.scalar(0)); // ReLU
  return a;
}

/*
- Compute output weights, this is a standard least square error regression problem, since we 
  try to minimize the least square error between the predicted labels and the training labels.
  The solution to this is: 

      T   -1  T
  β=(X  X)   X  y

  Where X is our input to hidden layer matrix computed using the function from the previous 
  step, and y is our training labels.
                            -1  
  Unfortunately there's no X   method in TensorFlow, we have to code it ourselves.
*/

function invertedMatrix(tensorX) {
  //Gauss method
  
  let size = tensorX.shape[0];
  
  // 1. Create a matrix consisting of X and an identity matrix to the right
  // For now it will be an ordinary JS array (needed for the next operation)
  let x = tensorX.concat(tf.eye(size), 1).arraySync();

  // 2. Turn the left half of the matrix into an identity matrix
  // by dividing and summing rows. After that the right half (which
  // was initially identity) will turn into an inverted matrix.
  let size2 = size * 2;
  for(let i = 0; i < size; ++ i) {
    // Make sure there's no zero on the main diagonal
    if (x[i][i] === 0.0) {
      let j;
      for(j = i + 1; j < size && x[j][i] === 0.0; ++ j);
      // If we didn't find a nonzero element in this column below the current line
      // then the matrix is degenerate
      if (j === size) {
        throw new Error('The matrix is degenerate');
      }
      let t = x[i];
      x[i] = x[j];
      x[j] = t;
    }        
    if (x[i][i] !== 1.0) {
      let d = x[i][i];
      for(let k = i; k < size2; ++k) {
        x[i][k] /= d;
      }
    }
    for (let j = 0; j < size; ++j) {
      if (j !== i && x[j][i] !== 0.0) {
        let m = x[j][i];
        for (let k = i; k < size2; ++k) {
          x[j][k] -= x[i][k] * m;
        }
      }
    }
  }
 
  return tf.tensor(x).split(2, 1)[1]; 
}

/*
Let me remind the formula:
      T   -1  T
  β=(X  X)   X  y
*/

console.log('Learning');
const wOut = (() => { 
  // Just to aviod polluting the global namespace
  // with local variables
  const x = inputToHidden(xTrainTensor);
  const xt = x.transpose();
  let xtx = tf.dot(xt, x);
  return tf.dot(invertedMatrix(xtx), tf.dot(xt, yTrainTensor));
})();

/*
Now that we have our trained model, let's create a function that predicts
the output, this is done simply by computing the dot product between the
result from the inputToHidden function we defined earlier, with the output 
weights:
*/

function predict(x, wout) {
  return tf.dot(inputToHidden(x), wout)
}

/*
Next, we can test our model:
*/

console.log('Predicting');
const predicted = predict(xTestTensor, wOut).argMax(-1).arraySync();
console.log('Predicted:')
f.displayVector(predicted);
const actual = yTestTensor.argMax(-1).arraySync();
console.log('Actual:');
f.displayVector(actual);

/*
Let's see what we've got
*/

const TEST_DATA_SIZE = actual.length;
  
const digits = Array.from(Array(10), () => ({correct: 0, mistaken: 0}))
const total = {
  correct: 0,
  mistaken: 0
};
for (let i = 0; i < TEST_DATA_SIZE; ++i) {
  if (predicted[i] === actual[i]) {
    ++ digits[predicted[i]].correct;
    ++ total.correct;
  } else {
    ++ digits[predicted[i]].mistaken;
    ++ digits[actual[i]].mistaken;
    ++ total.mistaken;
  }
}

console.log('Digit\t\tCorrect\t\tMistaken\tAccuracy');
for (let n = 0; n < 10; ++n) {
  let v = digits[n];
  console.log(
    n, '\t\t',
    v.correct, '\t\t',
    v.mistaken, '\t\t',
    v.correct + v.mistaken === 0 ? 
      '' 
    : 
      Math.floor(v.correct * 100 / (v.correct + v.mistaken)), '%'
  );
}
