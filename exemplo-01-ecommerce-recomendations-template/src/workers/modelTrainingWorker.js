import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
import { workerEvents } from '../events/constants.js';

let _globalCtx = {};
const WEIGHTS = {
    category: 0.4,
    color: 0.3,
    price: 0.2,
    age: 0.1
}

const normalize = (value, min, max) => (value - min) / (max - min);
const oneHotWeighted = (index, length, weight) => tf.oneHot(index, length).cast('float32').mul(weight);

function encodeProduct (product, ctx) {
    // normalizando dados do produto para ficar entre 0 e 1, e aplicando pesos para cada dimensão
    const age = tf.tensor1d([ctx.productAvgAgeProduct[product.name] ?? 0.5 * WEIGHTS.age]);
    const price = tf.tensor1d([normalize(product.price, ctx.minPrice, ctx.maxPrice) * WEIGHTS.price]);

    const color = oneHotWeighted(ctx.colorsIndex[product.color], ctx.numColors, WEIGHTS.color);
    const category = oneHotWeighted(ctx.categoryIndex[product.category], ctx.numCategories, WEIGHTS.category);

    return tf.concat([age, price, category, color]);
}

function createContext(catalog, users) {
    const ages = users.map(u => u.age);
    const prices = catalog.map(p => p.price);
    const colors = [...new Set(catalog.map(p => p.color))];
    const categories = [...new Set(catalog.map(p => p.category))];

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
        catalog.map(product => {
            const avg = ageCounts[product.name] ? ageSums[product.name] / ageCounts[product.name] : midAge;
            return [product.name, normalize(avg, minAge, maxAge)];
        })
    )

    return {
        catalog,
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

async function trainModel({ users }) {
    console.log('Training model with users:', users)

    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 50 } });

    const catalog = await (await fetch('/data/products.json')).json();
    const context = createContext(catalog, users);
    context.productVectors = catalog.map(product => {
        return {
            name: product.name,
            meta: {...product},
            vector: encodeProduct(product, context).dataSync()
        }
    });

    _globalCtx = context;

    postMessage({
        type: workerEvents.trainingLog,
        epoch: 1,
        loss: 1,
        accuracy: 1
    });

    setTimeout(() => {
        postMessage({ type: workerEvents.progressUpdate, progress: { progress: 100 } });
        postMessage({ type: workerEvents.trainingComplete });
    }, 1000);


}

function recommend(user, ctx) {
    console.log('will recommend for user:', user)
    // postMessage({
    //     type: workerEvents.recommend,
    //     user,
    //     recommendations: []
    // });
}

const handlers = {
    [workerEvents.trainModel]: trainModel,
    [workerEvents.recommend]: d => recommend(d.user, _globalCtx),
};

self.onmessage = e => {
    const { action, ...data } = e.data;
    if (handlers[action]) handlers[action](data);
};
