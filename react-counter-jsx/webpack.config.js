module.exports = {
  entry: "./src/index.jsx",
  output: {
    filename: "bundle.js",
    path: __dirname + "/dist",
    clean: true,
  },
  resolve: {
    extensions: [".js", ".jsx"],
  },
  module: {
    rules: [
      {
        test: /\.(js|jsx)$/,
        exclude: /node_modules/,
        use: {
          loader: "babel-loader",
          options: {
            presets: ["@babel/preset-env", "@babel/preset-react"],
          },
        },
      },
      {
        test: /\.css$/,
        use: ["style-loader", "css-loader"],
      },
    ],
  },
  devServer: {
    static: "./public",
    port: 3000,
    hot: true,
  },
  plugins: [
    new (require("html-webpack-plugin"))({
      template: "./public/index.html",
    }),
  ],
};
