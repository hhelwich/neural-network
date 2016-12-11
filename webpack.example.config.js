var path = require("path");

module.exports = {
  devServer: { inline: true },
  entry: './ts-built/src/example/predict-sinus/main.js',
  output: {
    filename: "predict-sinus.bundle.js",
  }
};
