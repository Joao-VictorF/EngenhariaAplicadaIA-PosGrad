import tf from '@tensorflow/tfjs-node';

async function trainModel(inputXs, outputYs) {
    const model = tf.sequential();
    
    // Camada de entrada com 7 neurônios (correspondente às 7 features: idade, cor (3) e localização (3))
    // 80 neurônios na camada oculta para capturar relações complexas
    // 80 neuroios pois a base de dados é pequena, então não queremos uma rede muito grande para evitar overfitting
    // quanto mais neurônios, mais complexa a rede, e mais dados são necessários para treinar e mais processamento é necessário
    // A ReLU age como um filtro, permitindo que apenas os sinais mais fortes passem para a próxima camada, o que ajuda a rede a aprender padrões mais complexos.
    // Se a info chegou nesse neuronio, é porque ela é relevante para a decisão final, e a ReLU ajuda a destacar essas informações importantes.
    model.add(tf.layers.dense({ inputShape: [7], units: 80, activation: 'relu' }));

    // Camada de saída com 3 neurônios (correspondente às 3 categorias: premium, medium, basic)
    // A softmax é usada para converter as saídas em probabilidades, ou seja, a soma das saídas será igual a 1, e cada saída representará a probabilidade de pertencer a cada categoria.
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));

    // Compilamos o modelo com o otimizador Adam (Adaptive Moment Estimation): 
    // Ajuda os pesos da rede a serem ajustados de forma eficiente durante o treinamento, adaptando a taxa de aprendizado para cada peso com base nas primeiras e segundas derivadas dos gradientes.
    // Loss: A função de perda 'categoricalCrossentropy' é usada para problemas de classificação multiclasse, onde as saídas são categorias exclusivas. Ela mede a diferença entre as distribuições de probabilidade previstas pelo modelo e as distribuições reais (labels). O objetivo do treinamento é minimizar essa perda, o que significa que o modelo está aprendendo a prever as categorias corretas com maior precisão.
    // Metrics: A métrica 'accuracy' é usada para avaliar a precisão do modelo durante o treinamento e a avaliação. Ela calcula a proporção de previsões corretas em relação ao total de previsões feitas, fornecendo uma medida intuitiva de quão bem o modelo está performando.
    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy', // A perda de entropia cruzada é usada para problemas de classificação multiclasse, onde as saídas são categorias exclusivas.
        metrics: ['accuracy']
    });

    await model.fit(inputXs, outputYs, {
        verbose: 0, // Exibe o progresso do treinamento
        epochs: 100, // Número de Vezes que vai reler a base de dados para treinar o modelo
        shuffle: true, // Embaralha os dados a cada época para melhorar o treinamento
        // callbacks: {
        //     onEpochEnd: (epoch, logs) => {
        //         console.log(`Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(4)}, accuracy = ${logs.acc ? logs.acc.toFixed(4) : 'N/A'}`);
        //     }
        // }
    });

    return model;
}

async function predict(model, pessoaNormalizada) {
    const inputTensor = tf.tensor2d(pessoaNormalizada);
    const prediction = model.predict(inputTensor);
    const predArray = await prediction.array();

    return predArray[0].map((prob, index) => ({prob, index}));
}

// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = [
    [0.33, 1, 0, 0, 1, 0, 0], // Erick
    [0, 0, 1, 0, 0, 1, 0],    // Ana
    [1, 0, 0, 1, 0, 0, 1]     // Carlos
]

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
    [1, 0, 0], // premium - Erick
    [0, 1, 0], // medium - Ana
    [0, 0, 1]  // basic - Carlos
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado)
const outputYs = tf.tensor2d(tensorLabels)

const model = await trainModel(inputXs, outputYs)

// Normalizamos os dados de entrada da pessoa para o formato esperado pelo modelo
const pessoaNormalizada = [[
    0.2, // idade normalizada (28/40)
    0,   // azul
    0,   // vermelho
    1,   // verde
    1,   // São Paulo
    0,   // Rio
    0    // Curitiba
]]

const predictions = await predict(model, pessoaNormalizada) 
const result = predictions
    .sort((a, b) => b.prob - a.prob)
    .map(p => `${labelsNomes[p.index]}: ${(p.prob * 100).toFixed(2)}%`)
    .join("\n");

console.log("Resultados das previsões:\n", result);
