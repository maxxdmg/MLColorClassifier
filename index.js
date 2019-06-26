let model;
let labelList = [
	'red-ish',
	'green-ish',
	'blue-ish',
	'orange-ish',
	'yellow-ish',
	'pink-ish',
	'purple-ish',
	'brown-ish',
	'grey-ish'
]
let data;
let xs;
let ys;
let lossElem = document.getElementById("loss")
let epochElem = document.getElementById("epoch")
let predElem = document.getElementById("prediction")


readJSON = (path) => {
	return new Promise((res, rej) => {
		var xhr = new XMLHttpRequest();
	    xhr.open('GET', path, true);
	    xhr.responseType = 'blob';
	    xhr.onload = function(e) { 
	      if (this.status == 200) {
	          var file = new File([this.response], 'temp');
	          var fileReader = new FileReader();
	          fileReader.addEventListener('load', function(){
	              res(fileReader.result);
	          });
	          fileReader.readAsText(file);
	      } 
	    }
	    xhr.send();
	})
}

normalize = (data, labelList) => {
	let colors = []
	let labels = []
	for (let x of data.entries) {
		let col = [x.r / 255, x.g / 255, x.b / 255]
		colors.push(col)
		labels.push(labelList.indexOf(x.label))
	}

	// initialize tensor inputs
	let xs = tf.tensor2d(colors)
	
	// initialize tensor training outputs w/ one hot encoding
	let labelsTensor = tf.tensor1d(labels, 'int32')
	
	let ys = tf.oneHot(labelsTensor, 9);

	return [xs, ys]
}

createModel = async (xs, ys) => {
	// create model and layers
	model = tf.sequential()

	let hidden = tf.layers.dense({
		units: 16,
		activation: 'sigmoid',
		inputDim: 3
	})
	let output = tf.layers.dense({
		units: 9,
		activation: 'softmax'
	})

	// add layers 
	model.add(hidden)
	model.add(output)

	// create optimizer
	const learningRate = 0.2
	const optimizer = tf.train.sgd(learningRate)

	// compile model
	model.compile({
		optimizer,
		loss: 'categoricalCrossentropy'
	})

	// config training options and train the model
	let epochNum = 10
	const options = {
		epochs: epochNum,
		validationSplit: 0.1,
		shuffle: true,
		callbacks: {
			onTrainBegin: () => console.log('training started'),
			onTrainEnd: () => console.log('finished <3'),
			onBatchEnd: tf.nextFrame,
			onEpochEnd: (num, logs) => {
				lossElem.innerHTML = "loss: " + logs.loss
				epochElem.innerHTML = "epoch # " + ( num + 1 ) + " / " + epochNum
				// console.log('Loss: ' + logs.loss)
			}
		}
	}

	return await model.fit(xs, ys, options)
}


predict = (r, g, b) => {
	// normalize the values
	r = r/255
	g = g/255
	b = b/255

	tf.tidy(() => {
		// create input data
		const xs = tf.tensor2d([
			[r, g, b]
		])

		let results = model.predict(xs)
		let index = results.argMax(1).dataSync()[0]
		let label = labelList[index]
		predElem.innerHTML = label
	})
}

changeBG = (r, g, b) => {
	let bg = document.getElementById("bg")
	let col = "rgb(" + r + ", " + g + ", " + b + ")"
	console.log(col)
	bg.style.backgroundColor = col
}

onTrainingEnd = () => {
	let load = document.getElementById("load")
	load.hidden = true
	let container = document.getElementById("container")
	container.hidden = false
	changeBG(0, 0, 0)	

	let r = document.getElementById("red")
	let g = document.getElementById("green")
	let b = document.getElementById("blue")

	r.addEventListener('change', () => {
		changeBG(r.value, g.value, b.value)
		predict(r.value, g.value, b.value)
	})
	g.addEventListener('change', () => {
		changeBG(r.value, g.value, b.value)
		predict(r.value, g.value, b.value)
	})
	b.addEventListener('change', () => {
		changeBG(r.value, g.value, b.value)
		predict(r.value, g.value, b.value)
	})
}

run = async () => {
	let json = await readJSON('./data.json')
	data = JSON.parse(json)
	let normArr = normalize(data, labelList)
	xs = normArr[0]
	ys = normArr[1]
	
	createModel(xs, ys).then(res => {
		console.log(res.history.loss)
		onTrainingEnd()
	})

}

run()
