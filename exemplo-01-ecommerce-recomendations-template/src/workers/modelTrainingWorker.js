import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
import { workerEvents } from '../events/constants.js';

let _model = null;
let _globalCtx = {};

const WEIGHTS = {
    category: 0.4,
    color: 0.3,
    price: 0.2,
    age: 0.1
}

const normalize = (value, min, max) => (value - min) / (max - min);
const oneHotWeighted = (index, length, weight) => tf.oneHot(index, length).cast('float32').mul(weight);

function encodeProduct (product, context) {
    // normalizando dados do produto para ficar entre 0 e 1, e aplicando pesos para cada dimensão
    const age = tf.tensor1d([context.productAvgAgeProduct[product.name] ?? 0.5 * WEIGHTS.age]);
    const price = tf.tensor1d([normalize(product.price, context.minPrice, context.maxPrice) * WEIGHTS.price]);

    const color = oneHotWeighted(context.colorsIndex[product.color], context.numColors, WEIGHTS.color);
    const category = oneHotWeighted(context.categoryIndex[product.category], context.numCategories, WEIGHTS.category);

    return tf.concat1d([age, price, category, color]);
}

function createContext(products, users) {
    const ages = users.map(u => u.age);
    const prices = products.map(p => p.price);
    const colors = [...new Set(products.map(p => p.color))];
    const categories = [...new Set(products.map(p => p.category))];

    const minAge = Math.min(...ages);
    const maxAge = Math.max(...ages);

    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);

    const colorsIndex = Object.fromEntries(
        colors.map((color, index) => [color, index])
    )

    const categoryIndex = Object.fromEntries(
        categories.map((category, index) => [category, index])
    )

    // Computar a média de idade dos usuários por produto comprado (ajuda a personalizar recomendações)
    const midAge = (minAge + maxAge) / 2;
    const ageSums = {}
    const ageCounts = {}

    users.forEach(user => {
        user.purchases.forEach(product => {
            ageSums[product.name] = (ageSums[product.name] || 0) + user.age;
            ageCounts[product.name] = (ageCounts[product.name] || 0) + 1;
        })
    })

    const productAvgAgeProduct = Object.fromEntries(
        products.map(product => {
            const avg = ageCounts[product.name] ? ageSums[product.name] / ageCounts[product.name] : midAge;
            return [product.name, normalize(avg, minAge, maxAge)];
        })
    )

    return {
        products,
        users,
        colorsIndex,
        categoryIndex,
        minAge,
        maxAge,
        minPrice,
        maxPrice,
        numCategories: categories.length,
        numColors: colors.length,
        dimentions: 2 + colors.length + categories.length, // idade + price + one-hot colors + one-hot categories
        productAvgAgeProduct
    }
}

function encodeUser(user, context) {
    if(user.purchases.length) {
       return tf.stack(user.purchases.map(product => encodeProduct(product, context)))
        .mean(0) // média dos vetores dos produtos comprados para representar o usuário
        .reshape([1, context.dimentions]); // reshape para garantir que seja um vetor 2D (1, N) para compatibilidade com operações de ML
    }

    return tf.concat1d(
        [
            tf.zeros([1]), // preço é ignorado,
            tf.tensor1d([
                normalize(user.age, context.minAge, context.maxAge)
                * WEIGHTS.age
            ]),
            tf.zeros([context.numCategories]), // categoria ignorada,
            tf.zeros([context.numColors]), // color ignorada,

        ]
    ).reshape([1, context.dimentions])
}

function createTrainingData(context) {
    const labels = [];
    const inputs = [];

    context.users
        .filter(user => user.purchases.length > 0)
        .forEach(user => {
            const userVector = encodeUser(user, context).dataSync();
            context.products.forEach(product => {
                const productVector = encodeProduct(product, context).dataSync();
                const label = user.purchases.some(p => p.name === product.name) ? 1 : 0; // 1 se o usuário comprou o produto, caso contrário 0
                
                labels.push(label);
                inputs.push([...userVector, ...productVector]);
            })
        });
    return {
        xs: tf.tensor2d(inputs),
        ys: tf.tensor2d(labels, [labels.length, 1]),
        inputDimensions: context.dimentions * 2, // tamanho do vetor de entrada (userVector + productVector)
    };
}

async function configureNeuralNetAndTrain(trainingData) {
    const model = tf.sequential();

    model.add(tf.layers.dense({
        inputShape: [trainingData.inputDimensions],
        units: 128,
        activation: 'relu',
    }));

    model.add(tf.layers.dense({
        units: 64,
        activation: 'relu',
    }));

    model.add(tf.layers.dense({
        units: 32,
        activation: 'relu',
    }));

    model.add(tf.layers.dense({
        units: 1,
        activation: 'sigmoid'
    }))

    model.compile({
        optimizer: tf.train.adam(0.01),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    })

    await model.fit(trainingData.xs, trainingData.ys, {
        epochs: 100,
        batch: 32,
        shuffle: true,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                postMessage({
                    type: workerEvents.trainingLog,
                    epoch: epoch,
                    loss: logs.loss,
                    accuracy: logs.acc
                });
            }
        },
    })

    return model;
}

async function trainModel({ users }) {
    console.log('Training model with users:', users)
    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 50 } });

    const products = await (await fetch('/data/products.json')).json();
    const context = createContext(products, users);
    context.productVectors = products.map(product => {
        return {
            name: product.name,
            meta: {...product},
            vector: encodeProduct(product, context).dataSync()
        }
    });

    _globalCtx = context;

    const trainedData = createTrainingData(context);
    _model = await configureNeuralNetAndTrain(trainedData);

    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 100 } });
    postMessage({ type: workerEvents.trainingComplete });
}

function recommend(user, context) {
    if (_model === null) return;

    const userVector = encodeUser(user, context).dataSync();
    const inputs = context.productVectors.map(({ vector }) => [...userVector, ...vector]);

    const inputTensor = tf.tensor2d(inputs);
    const predictions = _model.predict(inputTensor);

    const scores = predictions.dataSync();

    const recommendations = context.productVectors
        .map((product, index) => ({
            ...product.meta,
            name: product.name,
            score: scores[index]
        }))

    const sortedRecommendations = recommendations.sort((a, b) => b.score - a.score);

    console.log('will recommend for user:', user)
    postMessage({
        type: workerEvents.recommend,
        user,
        recommendations: sortedRecommendations
    });
}

const handlers = {
    [workerEvents.trainModel]: trainModel,
    [workerEvents.recommend]: d => recommend(d.user, _globalCtx),
};

self.onmessage = e => {
    const { action, ...data } = e.data;
    if (handlers[action]) handlers[action](data);
};
