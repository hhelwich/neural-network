var JasmineWebpackPlugin = require('jasmine-webpack-plugin');

module.exports = {
  devServer: { inline: true },
  entry: ['./ts-built/test/main.js'],
  plugins: [new JasmineWebpackPlugin({
    filename: 'index.html'
  })]
};
